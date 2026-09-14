"""Fresh 4B LoRA learning fixed teacher Top-10 distributions; no student rollouts."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import hashlib
import math
from pathlib import Path
import random
import re
from time import perf_counter

import numpy as np

from .evaluate import digest, evaluate, read_jsonl
from .inference import PROMPT
from .kd_collect import TOP_K, cache_entry, cache_lock, json_hash
from .kd_targets import normalize_batch, summarize_soft, topk_diagnostics
from .metrics import VERSION
from .official import OfficialScorer, REVISION
from .sft import (
    MODEL,
    TRAIN_RATE,
    FORWARD_RATE,
    SAMPLE_RATE,
    evaluation_steps,
    generate,
    prepare_examples,
    scheduled_lr,
    validate_records,
    write_json,
)
from .sft_metrics import summarize_nll
from .tinker_inference import (
    COOKBOOK_REVISION,
    PROCESSOR_REVISION,
    TEACHER_MODEL,
    TEACHER_REVISION,
    image_message,
    load_renderer,
)
from .training_logging import TrainingLogger, evaluation_metrics


def load_train(args, tokenizer, renderer):
    """A training run consumes an immutable prefix of the preselected cache."""
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "teacher_model": TEACHER_MODEL,
        "teacher_processor_revision": TEACHER_REVISION,
        "student_model": MODEL,
        "student_processor_revision": PROCESSOR_REVISION,
        "cookbook_revision": COOKBOOK_REVISION,
        "top_k": TOP_K,
        "loss_temperature": 1,
        "rollout_temperature": 0,
        "train_manifest_sha256": digest(args.train_manifest),
        "dev_manifest_sha256": digest(args.dev_manifest),
        "tokenizer_sha256": json_hash(json.loads(tokenizer.backend_tokenizer.to_str())),
        "max_pixels": args.max_pixels,
        "max_sequence_tokens": args.max_sequence_tokens,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Teacher cache model, data or processing protocol differs")
    if not 1 <= args.train_examples <= len(manifest["records"]):
        raise ValueError("Invalid cached training subset size")
    by_id = {row["id"]: row for row in read_jsonl(args.train_manifest)}
    datums, probabilities, hashes = [], [], []
    records = manifest["records"][: args.train_examples]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate cached examples")
    for index, record in enumerate(records):
        row = by_id[record["id"]]
        if digest(args.data_root / row["image"]) != record["image_sha256"]:
            raise ValueError("Cached image content changed")
        prompt = renderer.build_generation_prompt(
            [image_message(args.data_root / row["image"], args.max_pixels)]
        )
        datum, logprobs = cache_entry(
            args.cache_dir, index, record, prompt, len(tokenizer)
        )
        if datum.model_input.length + 1 > args.max_sequence_tokens:
            raise ValueError("Cached sequence exceeds context limit")
        datums.append(datum)
        probabilities.append(logprobs)
        hashes.append(digest(args.cache_dir / f"{index:04d}.json"))
    return (
        datums,
        topk_diagnostics(np.concatenate(probabilities)),
        {
            "cache_sha256": json_hash(
                {"manifest": digest(manifest_path), "entries": hashes}
            ),
            "tokenizer_sha256": manifest["tokenizer_sha256"],
        },
    )


def estimate_cost(train, dev, args):
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    training = sum(d.model_input.length for d in train) * args.epochs
    forward = 2 * sum(d.model_input.length for d in train)
    forward += len(evaluation_steps(steps, args.eval_every)) * sum(
        d.model_input.length for _, _, d in dev
    )
    prefill = sum(p.length for _, p, _ in dev) if args.generate_dev else 0
    output_bound = len(dev) * args.max_new_tokens if args.generate_dev else 0
    return {
        "training_tokens": training,
        "nll_forward_tokens": forward,
        "generation_prefill_tokens": prefill,
        "generation_output_token_bound": output_bound,
        "estimated_usd_bound": (
            training * TRAIN_RATE
            + (forward + prefill) * FORWARD_RATE
            + output_bound * SAMPLE_RATE
        )
        / 1e6,
    }


def assess(client, train, dev, stage, step, args, report, logger, train_endpoints):
    """Gold Dev likelihood and teacher-target fit measure different objectives."""
    result = {"step": step}
    gold = [d for _, _, d in dev]
    output = client.forward(gold, loss_fn="cross_entropy").result(timeout=600)
    summary = summarize_nll(output, gold)
    likelihoods = []
    for (row, _, datum), prediction in zip(dev, output.loss_fn_outputs, strict=True):
        mask = datum.loss_fn_inputs["weights"].data
        likelihoods.append(
            {
                "id": row["id"],
                "token_logprobs": [
                    lp
                    for lp, w in zip(prediction["logprobs"].data, mask, strict=True)
                    if w
                ],
                "target_tokens": [
                    t
                    for t, w in zip(
                        datum.loss_fn_inputs["target_tokens"].data, mask, strict=True
                    )
                    if w
                ],
            }
        )
    write_json(args.output_dir / f"{stage}_dev_likelihoods.json", likelihoods)
    summary.pop("per_example")
    result["dev"] = summary
    if train_endpoints:
        output = client.forward(train, loss_fn="cross_entropy").result(timeout=600)
        result["train_kd"] = summarize_soft(output, train)
    report[stage] = result
    report.setdefault("evaluation_stages", []).append(stage)
    write_json(args.output_dir / "run.json", report)
    public = evaluation_metrics(result)
    public.update({f"train_kd/{k}": v for k, v in result.get("train_kd", {}).items()})
    logger.log(public)
    with (args.output_dir / "evaluations.jsonl").open("a") as stream:
        stream.write(json.dumps(public, allow_nan=False) + "\n")
    print(json.dumps({"evaluation": public}), flush=True)


def run(args, train, dev, tokenizer, renderer, scorer, report, logger):
    import tinker

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=MODEL,
        rank=args.rank,
        seed=args.seed,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    report["status"] = "running"
    report["training_info"] = client.get_info().model_dump(mode="json")
    write_json(args.output_dir / "run.json", report)
    assess(client, train, dev, "before", 0, args, report, logger, True)
    started = perf_counter()
    rng, step = random.Random(args.seed), 0
    with (args.output_dir / "steps.jsonl").open("x") as stream:
        for epoch in range(args.epochs):
            order = list(train)
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                step += 1
                batch = order[start : start + args.batch_size]
                tick = perf_counter()
                output = client.forward_backward(
                    normalize_batch(batch), loss_fn="cross_entropy"
                ).result(timeout=600)
                metrics = summarize_soft(output, batch)  # Validate before optimizer.
                lr = scheduled_lr(step, args.learning_rate, report["warmup_steps"])
                client.optim_step(
                    tinker.AdamParams(
                        learning_rate=lr,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                        weight_decay=0,
                        grad_clip_norm=1.0,
                    )
                ).result(timeout=600)
                event = {
                    "training/optimizer_step": step,
                    "training/epoch": epoch + 1,
                    "training/learning_rate": lr,
                    "training/step_seconds": perf_counter() - tick,
                    **{f"training/{k}": v for k, v in metrics.items()},
                }
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                logger.log(event)
                print(json.dumps(event), flush=True)
                if step < report["optimizer_steps"] and step % args.eval_every == 0:
                    assess(
                        client,
                        train,
                        dev,
                        f"step_{step:04d}",
                        step,
                        args,
                        report,
                        logger,
                        False,
                    )
    report["training_loop_seconds"] = perf_counter() - started
    # Private checkpoint addresses never enter W&B config or source control.
    report["state_path"] = (
        client.save_state("final", ttl_seconds=7 * 86400).result().path
    )
    saved = client.save_weights_for_sampler("after", ttl_seconds=7 * 86400).result()
    report["sampler_paths"] = {"after": saved.path}
    write_json(args.output_dir / "run.json", report)
    assess(client, train, dev, "after", step, args, report, logger, True)
    if args.generate_dev:
        sampler = service.create_sampling_client(model_path=saved.path)
        predictions = generate(
            sampler,
            dev,
            tokenizer,
            renderer.get_stop_sequences(),
            args,
            args.output_dir / "after_dev_predictions.jsonl",
        )
        details, metrics = evaluate(
            [r for r, _, _ in dev], args.data_root, predictions, scorer
        )
        write_json(args.output_dir / "after_dev_details.json", details)
        report["after"]["dev"].update(metrics)
        logger.log(evaluation_metrics(report["after"]))
    output_tokens = report["after"]["dev"].get("eval/output_tokens_total", 0)
    report["estimated_compute_usd"] = (
        report["cost"]["estimated_usd_bound"]
        - (report["cost"]["generation_output_token_bound"] - output_tokens)
        * SAMPLE_RATE
        / 1e6
    )
    report["checks"] = {
        "completed_optimizer_steps": step,
        "teacher_target_kl_decreased": report["after"]["train_kd"][
            "truncated_forward_kl"
        ]
        < report["before"]["train_kd"]["truncated_forward_kl"],
        "full_dev_examples": len(dev),
    }
    report["status"] = "completed"
    write_json(args.output_dir / "run.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cache-dir",
        "train-manifest",
        "dev-manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
        "official-repo",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--train-examples", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--generate-dev", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-sequence-tokens", type=int, default=16384)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--max-estimated-usd", type=float, default=0.2)
    p.add_argument("--wandb-mode", choices=["disabled", "online"], default="disabled")
    p.add_argument("--wandb-project", default="vlm-table-extraction")
    p.add_argument("--dataset-label", default="rd-kd-smoke8")
    p.add_argument("--inference-concurrency", type=int, default=4)
    args = p.parse_args()
    for key, value in vars(args).items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key != "warmup_ratio"
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
    if (
        not math.isfinite(args.warmup_ratio)
        or not 0 <= args.warmup_ratio <= 1
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.dataset_label)
    ):
        raise ValueError("Invalid warmup or public dataset label")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Use a fresh output directory; paid training does not auto-resume"
        )
    train_rows, dev_rows = read_jsonl(args.train_manifest), read_jsonl(
        args.dev_manifest
    )
    if len(train_rows) != 800 or len(dev_rows) != 100:
        raise ValueError("Expected frozen Train800 and full Dev100")
    validate_records(train_rows, dev_rows, args.data_root)
    tokenizer, renderer = load_renderer(
        MODEL, PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    with cache_lock(args.cache_dir):
        train, mass, cache = load_train(args, tokenizer, renderer)
    if mass["retained_mass_mean"] < 0.98 or mass["retained_mass_p05"] < 0.90:
        raise ValueError(f"Top-10 retained mass below planned gate: {mass}")
    dev = prepare_examples(
        dev_rows, args.data_root, renderer, args.max_pixels, args.max_sequence_tokens
    )
    scorer = OfficialScorer(args.official_repo)
    cost = estimate_cost(train, dev, args)
    if 1.1 * cost["estimated_usd_bound"] > args.max_estimated_usd:
        raise ValueError("Training plus full-Dev checks exceeds reserved budget")
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    # Reuse the existing explicit telemetry allowlist and checkpoint evaluator.
    args.generate_train = False
    args.train_nll_endpoints_only = True
    args.generate_every = steps
    report = {
        "status": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "off_policy_topk_kd",
        "model": MODEL,
        "initialization": "fresh_lora_on_hosted_model",
        "hosted_weight_revision": None,
        "teacher_model": TEACHER_MODEL,
        "teacher_processor_revision": TEACHER_REVISION,
        "teacher_hosted_weight_revision": None,
        "processor_revision": PROCESSOR_REVISION,
        "cookbook_revision": COOKBOOK_REVISION,
        "official_revision": REVISION,
        "metrics_version": VERSION,
        "tinker_version": version("tinker"),
        "prompt": PROMPT,
        "train_manifest_sha256": digest(args.train_manifest),
        "dev_manifest_sha256": digest(args.dev_manifest),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "optimizer_steps": steps,
        "top_k": TOP_K,
        "loss_temperature": 1,
        "rollout_temperature": 0,
        "retained_mass": mass,
        **cache,
        "lora": {
            "rank": args.rank,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": False,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "weight_decay": 0,
            "grad_clip_norm": 1.0,
        },
        "loss_reduction": "batch_supervised_position_mean_soft_ce",
        "learning_rate_schedule": "linear_warmup_then_constant",
        "warmup_steps": math.ceil(args.warmup_ratio * steps),
        "rates_usd_per_million": {
            "training": TRAIN_RATE,
            "forward": FORWARD_RATE,
            "sample": SAMPLE_RATE,
        },
        "implementation_sha256": json_hash(
            {
                f: digest(Path(__file__).with_name(f))
                for f in ("kd.py", "kd_targets.py", "kd_collect.py")
            }
        ),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cost": cost,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run.json", report)
    print(
        json.dumps(
            {
                "status": "preflight",
                "optimizer_steps": steps,
                "retained_mass": mass,
                **cost,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    logger = None
    try:
        logger = TrainingLogger(report)
        run(args, train, dev, tokenizer, renderer, scorer, report, logger)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        if logger:
            logger.log({"training/status": report["status"]})
            if "estimated_compute_usd" in report:
                logger.log(
                    {
                        "runtime/estimated_training_compute_usd": report[
                            "estimated_compute_usd"
                        ]
                    }
                )
            report["wandb_receipt"] = logger.finish()
        write_json(args.output_dir / "run.json", report)


if __name__ == "__main__":
    main()
