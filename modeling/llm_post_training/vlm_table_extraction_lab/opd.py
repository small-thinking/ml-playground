"""Synchronous on-policy distillation: fresh student rollout, teacher score, one update."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
from time import perf_counter

from .evaluate import digest, evaluate, read_jsonl
from .inference import PROMPT
from .kd import assess
from .kd_collect import (
    CollectionBudget,
    TEACHER_FORWARD_RATE,
    TEACHER_SAMPLE_RATE,
    json_hash,
    tokenizer_identity,
)
from .metrics import VERSION, parse_table
from .official import OfficialScorer, REVISION
from .opd_targets import native_batch, policy_datum, summarize_policy
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
from .tinker_inference import (
    COOKBOOK_REVISION,
    PROCESSOR_REVISION,
    TEACHER_MODEL,
    TEACHER_REVISION,
    image_message,
    load_renderer,
)
from .training_logging import TrainingLogger, evaluation_metrics


def prepare_prompts(rows, root, renderer, teacher_renderer, args):
    prompts = []
    for row in rows:
        message = image_message(root / row["image"], args.max_pixels)
        prompt = renderer.build_generation_prompt([message])
        teacher_prompt = teacher_renderer.build_generation_prompt([message])
        if json_hash(prompt.model_dump(mode="json")) != json_hash(
            teacher_prompt.model_dump(mode="json")
        ):
            raise ValueError("Teacher/student multimodal prompts differ")
        if prompt.length + args.max_new_tokens > args.max_sequence_tokens:
            raise ValueError("Prompt plus rollout cap exceeds context")
        prompts.append((row, prompt))
    return prompts


def estimate_cost(train, dev, args):
    """Report both a length scenario and the worst output-cap bound, not a quote."""
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    prefill = sum(p.length for _, p in train) * args.epochs
    forward = len(evaluation_steps(steps, args.eval_every)) * sum(
        d.model_input.length for _, _, d in dev
    )
    dev_prefill = sum(p.length for _, p, _ in dev) if args.generate_dev else 0
    dev_output = len(dev) * args.max_new_tokens if args.generate_dev else 0

    def cost(length):
        output = len(train) * args.epochs * length
        return (
            prefill * FORWARD_RATE
            + output * SAMPLE_RATE
            + (prefill + output) * TEACHER_FORWARD_RATE
            + len(train) * args.epochs * TEACHER_SAMPLE_RATE
            + (prefill + output - len(train) * args.epochs) * TRAIN_RATE
            + (forward + dev_prefill) * FORWARD_RATE
            + dev_output * SAMPLE_RATE
        ) / 1e6

    return {
        "training_tokens": prefill
        + len(train) * args.epochs * args.max_new_tokens
        - len(train) * args.epochs,
        "nll_forward_tokens": forward,
        "generation_prefill_tokens": dev_prefill,
        "generation_output_token_bound": dev_output,
        "estimated_usd_bound": cost(args.max_new_tokens),
        "estimated_usd_length_scenario": cost(args.expected_output_tokens),
        "expected_output_tokens_per_rollout": args.expected_output_tokens,
        "execution_budget_usd": args.max_estimated_usd,
    }


def collect_one(sampler, teacher, row, prompt, tokenizer, stop, args, step, position):
    """Persist each paid response before the next request; no application retries."""
    import tinker

    path = args.output_dir / "rollouts" / f"{step:04d}_{position:02d}.json"
    sequence = (
        sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1,
                top_p=1,
                top_k=-1,
                stop=stop,
                max_tokens=args.max_new_tokens,
                seed=args.seed + 1000 * step + position,
            ),
        )
        .result(timeout=600)
        .sequences[0]
    )
    record = {
        "id": row["id"],
        "image_sha256": row["image_sha256"],
        "prompt_sha256": json_hash(prompt.model_dump(mode="json")),
        "optimizer_step_before_rollout": step - 1,
        "tokens": sequence.tokens,
        "sampling_logprobs": sequence.logprobs,
        "stop_reason": sequence.stop_reason,
        "prompt_tokens": prompt.length,
    }
    write_json(path, record)
    if not sequence.tokens or sequence.stop_reason not in {"stop", "length"}:
        raise ValueError("Empty or unknown student completion")
    # Validate sampled IDs/logprobs before spending on teacher scoring.
    policy_datum(
        prompt, sequence.tokens, sequence.logprobs, sequence.logprobs, len(tokenizer)
    )
    if len(sequence.tokens) > args.max_new_tokens:
        raise ValueError("Student exceeded output cap")
    full = prompt.append(tinker.EncodedTextChunk(tokens=sequence.tokens))
    teacher_logprobs = teacher.compute_logprobs(full).result(timeout=600)
    record["teacher_full_logprobs"] = teacher_logprobs
    write_json(path, record)
    if len(teacher_logprobs) != full.length:
        raise ValueError("Teacher logprob length differs from full sequence")
    datum = policy_datum(
        prompt,
        sequence.tokens,
        sequence.logprobs,
        teacher_logprobs[prompt.length :],
        len(tokenizer),
    )
    html = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
    try:
        parse_table(html)
        valid = True
    except ValueError:
        valid = False
    record.update(html=html, format_valid=valid)
    write_json(path, record)
    return datum, {
        "rollout_tokens": len(sequence.tokens),
        "prefill_tokens": prompt.length,
        "teacher_forward_tokens": full.length,
        "training_tokens": datum.model_input.length,
        "format_invalid": int(not valid),
        "truncated": int(sequence.stop_reason == "length"),
    }


def generation_cost(prefill, output):
    return (prefill * FORWARD_RATE + output * SAMPLE_RATE) / 1e6


def run(args, train, dev, tokenizer, renderer, scorer, report, logger):
    import tinker
    from tinker.lib.retry_handler import RetryConfig

    budget = CollectionBudget(args.output_dir / "usage.json", args.max_estimated_usd)
    service = tinker.ServiceClient()
    retry = RetryConfig(enable_retry_logic=False)
    teacher = service.create_sampling_client(
        base_model=TEACHER_MODEL, retry_config=retry
    )
    client = service.create_lora_training_client(
        base_model=MODEL,
        rank=args.rank,
        seed=args.seed,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    report.update(
        status="running", training_info=client.get_info().model_dump(mode="json")
    )
    report["intermediate_sampler_paths"] = {}
    write_json(args.output_dir / "run.json", report)

    def dev_assess(stage, step):
        amount = sum(d.model_input.length for _, _, d in dev) * FORWARD_RATE / 1e6
        budget.reserve(f"{stage}:dev_nll", amount)
        assess(client, [], dev, stage, step, args, report, logger, False)
        budget.settle(amount)

    dev_assess("before", 0)
    rng, step = random.Random(args.seed), 0
    totals = dict.fromkeys(
        (
            "rollout_tokens",
            "prefill_tokens",
            "teacher_forward_tokens",
            "training_tokens",
            "format_invalid",
            "truncated",
        ),
        0,
    )
    started = perf_counter()
    with (args.output_dir / "steps.jsonl").open("x") as stream:
        for epoch in range(args.epochs):
            order = list(train)
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                step += 1
                batch = order[start : start + args.batch_size]
                tick = perf_counter()
                prefill = sum(p.length for _, p in batch)
                cap = len(batch) * args.max_new_tokens
                bound = (
                    prefill * FORWARD_RATE
                    + cap * SAMPLE_RATE
                    + (prefill + cap) * TEACHER_FORWARD_RATE
                    + len(batch) * TEACHER_SAMPLE_RATE
                    + (prefill + cap - len(batch)) * TRAIN_RATE
                ) / 1e6
                budget.reserve(f"step_{step}:rollout_teacher_backward", bound)
                # No update or prefetch across batches: this snapshot is current.
                sampler = client.save_weights_and_get_sampling_client(
                    retry_config=retry
                )
                with ThreadPoolExecutor(max_workers=args.inference_concurrency) as pool:
                    futures = [
                        pool.submit(
                            collect_one,
                            sampler,
                            teacher,
                            row,
                            prompt,
                            tokenizer,
                            renderer.get_stop_sequences(),
                            args,
                            step,
                            j,
                        )
                        for j, (row, prompt) in enumerate(batch)
                    ]
                    try:
                        collected = [f.result() for f in futures]
                    except Exception:
                        for future in futures:
                            future.cancel()
                        raise
                datums = [d for d, _ in collected]
                output = client.forward_backward(
                    native_batch(datums), loss_fn="importance_sampling"
                ).result(timeout=600)
                # Persist learner evidence before optimizer; a failure never retries.
                write_json(
                    args.output_dir / "rollouts" / f"{step:04d}_learner.json",
                    [r["logprobs"].to_numpy().tolist() for r in output.loss_fn_outputs],
                )
                metrics = summarize_policy(output, datums)
                lr = scheduled_lr(step, args.learning_rate, report["warmup_steps"])
                client.optim_step(
                    tinker.AdamParams(
                        learning_rate=lr,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                        weight_decay=0,
                        grad_clip_norm=1,
                    )
                ).result(timeout=600)
                usage = {key: sum(u[key] for _, u in collected) for key in totals}
                amount = (
                    usage["prefill_tokens"] * FORWARD_RATE
                    + usage["rollout_tokens"] * SAMPLE_RATE
                    + usage["teacher_forward_tokens"] * TEACHER_FORWARD_RATE
                    + len(batch) * TEACHER_SAMPLE_RATE
                    + usage["training_tokens"] * TRAIN_RATE
                ) / 1e6
                budget.settle(amount)
                for key, value in usage.items():
                    totals[key] += value
                event = {
                    "training/optimizer_step": step,
                    "training/epoch": epoch + 1,
                    "training/learning_rate": lr,
                    "training/step_seconds": perf_counter() - tick,
                    **{f"training/{k}": v for k, v in metrics.items()},
                    **{f"training/{k}": v for k, v in usage.items()},
                    "runtime/estimated_compute_usd": budget.state[
                        "estimated_compute_usd"
                    ],
                }
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                logger.log(event)
                print(json.dumps(event), flush=True)
                report.update(
                    completed_optimizer_steps=step,
                    usage=totals,
                    estimated_compute_usd=budget.state["estimated_compute_usd"],
                )
                write_json(args.output_dir / "run.json", report)
                if step < report["optimizer_steps"] and step % args.eval_every == 0:
                    stage = f"step_{step:04d}"
                    saved = client.save_weights_for_sampler(
                        stage, ttl_seconds=7 * 86400
                    ).result()
                    report["intermediate_sampler_paths"][stage] = saved.path
                    dev_assess(stage, step)
    report["training_loop_seconds"] = perf_counter() - started
    report["state_path"] = (
        client.save_state("final", ttl_seconds=7 * 86400).result().path
    )
    saved = client.save_weights_for_sampler("after", ttl_seconds=7 * 86400).result()
    report["sampler_paths"] = {"after": saved.path}
    dev_assess("after", step)
    if args.generate_dev:
        prefill = sum(p.length for _, p, _ in dev)
        budget.reserve(
            "after:dev_generation",
            generation_cost(prefill, len(dev) * args.max_new_tokens),
        )
        sampler = service.create_sampling_client(
            model_path=saved.path, retry_config=retry
        )
        predictions = generate(
            sampler,
            dev,
            tokenizer,
            renderer.get_stop_sequences(),
            args,
            args.output_dir / "after_dev_predictions.jsonl",
        )
        output_tokens = sum(p["output_tokens"] for p in predictions.values())
        budget.settle(generation_cost(prefill, output_tokens))
        details, metrics = evaluate(
            [r for r, _, _ in dev], args.data_root, predictions, scorer
        )
        write_json(args.output_dir / "after_dev_details.json", details)
        report["after"]["dev"].update(metrics)
        logger.log(evaluation_metrics(report["after"]))
    report.update(
        status="completed",
        estimated_compute_usd=budget.state["estimated_compute_usd"],
        checks={
            "completed_optimizer_steps": step,
            "full_dev_examples": len(dev),
            "rollouts": len(train) * args.epochs,
        },
    )
    write_json(args.output_dir / "run.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--generate-dev", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-sequence-tokens", type=int, default=16384)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--expected-output-tokens", type=int, default=1600)
    p.add_argument("--max-estimated-usd", type=float, default=0.3)
    p.add_argument("--wandb-mode", choices=["disabled", "online"], default="disabled")
    p.add_argument("--wandb-project", default="vlm-table-extraction")
    p.add_argument("--dataset-label", default="rd-opd-smoke8")
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
    if not 0 <= args.warmup_ratio <= 1 or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,80}", args.dataset_label
    ):
        raise ValueError("Invalid warmup or public label")
    if args.expected_output_tokens > args.max_new_tokens:
        raise ValueError("Expected length exceeds output cap")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Fresh output directory required; paid runs never auto-resume")
    rows, dev_rows = read_jsonl(args.train_manifest), read_jsonl(args.dev_manifest)
    if len(rows) != 800 or len(dev_rows) != 100 or not 1 <= args.train_examples <= 800:
        raise ValueError("Expected fixed Train800 and complete Dev100")
    validate_records(rows, dev_rows, args.data_root)
    tokenizer, renderer = load_renderer(
        MODEL, PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    teacher_tokenizer, teacher_renderer = load_renderer(
        TEACHER_MODEL, TEACHER_REVISION, args.tinker_cookbook_dir
    )
    identity = tokenizer_identity(tokenizer, teacher_tokenizer)
    train = prepare_prompts(
        rows[: args.train_examples], args.data_root, renderer, teacher_renderer, args
    )
    dev = prepare_examples(
        dev_rows, args.data_root, renderer, args.max_pixels, args.max_sequence_tokens
    )
    scorer = OfficialScorer(args.official_repo)
    cost = estimate_cost(train, dev, args)
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    args.generate_train, args.train_nll_endpoints_only, args.generate_every = (
        False,
        True,
        steps,
    )
    args.rollouts_per_example, args.updates_per_rollout = 1, 1
    report = {
        "status": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "on_policy_reverse_kl_opd",
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
        "tokenizer_sha256": identity,
        "rollout_temperature": 1.0,
        "loss_temperature": 1.0,
        "opd_objective": "sampled_reverse_kl",
        "advantage_discount": 0.0,
        "rollout_seed_policy": "seed_plus_1000_step_plus_position",
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
        "loss_reduction": "batch_supervised_position_mean_importance_sampling",
        "learning_rate_schedule": "linear_warmup_then_constant",
        "warmup_steps": math.ceil(args.warmup_ratio * steps),
        "rates_usd_per_million": {
            "training": TRAIN_RATE,
            "forward": FORWARD_RATE,
            "sample": SAMPLE_RATE,
            "teacher_forward": TEACHER_FORWARD_RATE,
            "teacher_scoring_output": TEACHER_SAMPLE_RATE,
        },
        "implementation_sha256": json_hash(
            {
                f: digest(Path(__file__).with_name(f))
                for f in (
                    "opd.py",
                    "opd_targets.py",
                    "kd.py",
                    "sft.py",
                    "kd_collect.py",
                )
            }
        ),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cost": cost,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rollouts").mkdir()
    write_json(args.output_dir / "run.json", report)
    print(
        json.dumps({"status": "preflight", "optimizer_steps": steps, **cost}),
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
