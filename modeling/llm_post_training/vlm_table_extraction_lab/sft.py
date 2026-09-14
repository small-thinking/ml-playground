"""Small, local-logged Tinker LoRA SFT. Default: preflight only; --execute trains."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
from time import perf_counter

from .evaluate import digest, evaluate, read_jsonl
from .inference import PROMPT
from .metrics import VERSION, parse_table
from .official import OfficialScorer, REVISION
from .sft_metrics import summarize_nll
from .tinker_inference import (
    COOKBOOK_REVISION,
    PROCESSOR_REVISION,
    image_message,
    load_renderer,
)

MODEL = "Qwen/Qwen3.5-4B"
# USD / million tokens, verified 2026-09-14. Estimates, not billing guarantees.
TRAIN_RATE, FORWARD_RATE, SAMPLE_RATE = 0.737, 0.33, 1.005


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_records(train, dev, root):
    """Verify labels/hashes and reject exact cross-split contamination before API use."""
    fingerprints = []
    for rows, split in ((train, "train"), (dev, "dev")):
        images, labels = set(), set()
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"Expected explicitly marked {split} records")
            for field in ("image", "label"):
                path = (root / row[field]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Data path escapes data root")
                actual = digest(path)
                if actual != row.get(f"{field}_sha256"):
                    raise ValueError(f"Missing or mismatched {field} hash")
                (images if field == "image" else labels).add(actual)
            parse_table((root / row["label"]).read_text())
        fingerprints.append((images, labels))
    if (
        {r["id"] for r in train} & {r["id"] for r in dev}
        or fingerprints[0][0] & fingerprints[1][0]
        or fingerprints[0][1] & fingerprints[1][1]
    ):
        raise ValueError("Train/dev overlap in IDs, image bytes, or labels")


def next_token_datum(full, weights):
    """Shift once for causal prediction, retaining image chunks and masking them."""
    import tinker

    chunks = list(full.chunks)
    if not isinstance(chunks[-1], tinker.EncodedTextChunk):
        raise ValueError("Expected text at the end of the supervised sequence")
    targets = []
    for chunk in chunks:
        targets.extend(
            chunk.tokens
            if isinstance(chunk, tinker.EncodedTextChunk)
            else [0] * chunk.length
        )
    last = chunks.pop()
    if last.length > 1:
        chunks.append(tinker.EncodedTextChunk(tokens=last.tokens[:-1]))
    return tinker.Datum(
        model_input=tinker.ModelInput(chunks=chunks),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=targets[1:], dtype="int64"),
            "weights": tinker.TensorData(data=weights[1:].tolist(), dtype="float32"),
        },
    )


def prepare_examples(rows, root, renderer, max_pixels, max_length):
    from tinker_cookbook.renderers import TrainOnWhat

    examples = []
    for row in rows:
        user = image_message(root / row["image"], max_pixels)
        prompt = renderer.build_generation_prompt([user])
        full, weights = renderer.build_supervised_example(
            [user, {"role": "assistant", "content": (root / row["label"]).read_text()}],
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        # Reject rather than truncate: truncated HTML is an incorrect SFT target.
        if full.length > max_length:
            raise ValueError("Training example exceeds max sequence tokens")
        if weights[: prompt.length].sum().item() != 0:
            raise ValueError("Prompt/image tokens must have zero loss weight")
        datum = next_token_datum(full, weights)
        mask = datum.loss_fn_inputs["weights"].data
        if not any(mask) or any(w not in (0, 1) for w in mask):
            raise ValueError("Expected a nonempty binary assistant mask")
        examples.append((row, prompt, datum))
    return examples


def token_mean_batch(datums):
    """Tinker sums losses; normalize across supervised tokens in this whole batch."""
    import tinker

    tokens = sum(sum(d.loss_fn_inputs["weights"].data) for d in datums)
    if tokens <= 0:
        raise ValueError("Batch has no supervised tokens")
    return [
        tinker.Datum(
            model_input=d.model_input,
            loss_fn_inputs={
                **d.loss_fn_inputs,
                "weights": tinker.TensorData(
                    data=[w / tokens for w in d.loss_fn_inputs["weights"].data],
                    dtype="float32",
                ),
            },
        )
        for d in datums
    ]


def mean_nll(output, datums):
    return summarize_nll(output, datums)["assistant_nll"]


def evaluation_steps(total_steps, every):
    """Include initial/final evaluations once, even off the interval boundary."""
    return sorted({0, total_steps, *range(every, total_steps, every)})


def scheduled_lr(step, peak_lr, warmup_steps):
    return peak_lr * min(1.0, step / warmup_steps) if warmup_steps else peak_lr


def estimate_cost(
    train, dev, epochs, max_new_tokens, batch_size=2, eval_every=4, generate_every=8
):
    total_steps = math.ceil(len(train) / batch_size) * epochs
    nll_passes = len(evaluation_steps(total_steps, eval_every))
    extra_generations = len(evaluation_steps(total_steps, generate_every)) - 2
    train_tokens = sum(d.model_input.length for _, _, d in train) * epochs
    forward_tokens = nll_passes * sum(d.model_input.length for _, _, d in train + dev)
    # Endpoints generate Train+Dev; periodic generation is Dev-only.
    prefill_tokens = 2 * sum(p.length for _, p, _ in train + dev)
    prefill_tokens += extra_generations * sum(p.length for _, p, _ in dev)
    generation_examples = 2 * len(train + dev) + extra_generations * len(dev)
    output_bound = generation_examples * max_new_tokens
    return {
        "training_tokens": train_tokens,
        "nll_forward_tokens": forward_tokens,
        "nll_evaluations": nll_passes,
        "generation_examples": generation_examples,
        "generation_prefill_tokens": prefill_tokens,
        "generation_output_token_bound": output_bound,
        "training_usd": train_tokens * TRAIN_RATE / 1e6,
        "estimated_usd_bound": (
            train_tokens * TRAIN_RATE
            + (forward_tokens + prefill_tokens) * FORWARD_RATE
            + output_bound * SAMPLE_RATE
        )
        / 1e6,
    }


def generate(client, examples, tokenizer, stop, args, path):
    import tinker

    predictions = {}
    with path.open("x") as stream:
        for row, prompt, _ in examples:
            start = perf_counter()
            response = client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=0,
                    seed=args.seed,
                    stop=stop,
                    max_tokens=args.max_new_tokens,
                ),
            ).result(timeout=600)
            seq = response.sequences[0]
            prediction = {
                "id": row["id"],
                "html": tokenizer.decode(seq.tokens, skip_special_tokens=True),
                "input_tokens": prompt.length,
                "output_tokens": len(seq.tokens),
                "stop_reason": seq.stop_reason,
                "latency_seconds": perf_counter() - start,
            }
            stream.write(json.dumps(prediction) + "\n")
            stream.flush()
            predictions[row["id"]] = prediction
    return predictions


def run(args, train, dev, tokenizer, renderer, scorer, report):
    import tinker

    report["status"] = "running"
    write_json(args.output_dir / "run.json", report)
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=MODEL,
        rank=args.rank,
        seed=args.seed,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    report["training_info"] = client.get_info().model_dump(mode="json")
    write_json(args.output_dir / "run.json", report)

    def assess(stage, step, generate_train=False, generate_dev=False):
        start = perf_counter()
        sampler = None
        if generate_train or generate_dev:
            saved = client.save_weights_for_sampler(
                stage, ttl_seconds=7 * 86400
            ).result()
            report.setdefault("sampler_paths", {})[stage] = saved.path
            write_json(args.output_dir / "run.json", report)
            sampler = service.create_sampling_client(model_path=saved.path)
        results = {"step": step}
        for name, examples in (("train", train), ("dev", dev)):
            datums = [d for _, _, d in examples]
            output = client.forward(datums, loss_fn="cross_entropy").result(timeout=600)
            summary = summarize_nll(output, datums)
            # Keep raw supervised probabilities locally for independent recalculation.
            likelihoods = []
            for (row, _, datum), result, per_example in zip(
                examples,
                output.loss_fn_outputs,
                summary.pop("per_example"),
                strict=True,
            ):
                mask = datum.loss_fn_inputs["weights"].data
                likelihoods.append(
                    {
                        "id": row["id"],
                        **per_example,
                        "token_logprobs": [
                            lp
                            for lp, w in zip(result["logprobs"].data, mask, strict=True)
                            if w
                        ],
                        "target_tokens": [
                            t
                            for t, w in zip(
                                datum.loss_fn_inputs["target_tokens"].data,
                                mask,
                                strict=True,
                            )
                            if w
                        ],
                    }
                )
            write_json(
                args.output_dir / f"{stage}_{name}_likelihoods.json", likelihoods
            )
            if (name == "train" and generate_train) or (name == "dev" and generate_dev):
                predictions = generate(
                    sampler,
                    examples,
                    tokenizer,
                    renderer.get_stop_sequences(),
                    args,
                    args.output_dir / f"{stage}_{name}_predictions.jsonl",
                )
                details, metrics = evaluate(
                    [r for r, _, _ in examples], args.data_root, predictions, scorer
                )
                write_json(args.output_dir / f"{stage}_{name}_details.json", details)
                summary.update(metrics)
            results[name] = summary
        results["wall_seconds"] = perf_counter() - start
        results["dev_train_nll_gap"] = (
            results["dev"]["assistant_nll"] - results["train"]["assistant_nll"]
        )
        report[stage] = results
        report.setdefault("evaluation_stages", []).append(stage)
        write_json(args.output_dir / "run.json", report)
        curve = {"step": step, "dev_train_nll_gap": results["dev_train_nll_gap"]}
        for name in ("train", "dev"):
            for metric in (
                "assistant_nll",
                "assistant_perplexity",
                "example_nll_max",
                "eval/cell_f1",
                "eval/numeric_f1",
                "eval/parse_success",
                "eval/truncated",
            ):
                if metric in results[name]:
                    curve[f"{name}/{metric}"] = results[name][metric]
        with (args.output_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(curve, allow_nan=False) + "\n")
        print(json.dumps({"evaluation": curve}), flush=True)

    total_steps = math.ceil(len(train) / args.batch_size) * args.epochs
    warmup_steps = math.ceil(args.warmup_ratio * total_steps)
    assess("before", 0, generate_train=True, generate_dev=True)
    start = perf_counter()
    optimization_seconds = 0.0
    rng = random.Random(args.seed)
    with (args.output_dir / "steps.jsonl").open("x") as log:
        step = 0
        for epoch in range(args.epochs):
            order = list(train)
            rng.shuffle(order)
            for offset in range(0, len(order), args.batch_size):
                step += 1
                datums = [d for _, _, d in order[offset : offset + args.batch_size]]
                started = perf_counter()
                result = client.forward_backward(
                    token_mean_batch(datums), loss_fn="cross_entropy"
                ).result(timeout=600)
                batch_metrics = summarize_nll(result, datums)
                nll = batch_metrics[
                    "assistant_nll"
                ]  # Validate before applying gradients.
                learning_rate = scheduled_lr(step, args.learning_rate, warmup_steps)
                optim = client.optim_step(
                    tinker.AdamParams(
                        learning_rate=learning_rate,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                        weight_decay=0,
                        grad_clip_norm=1.0,
                    )
                ).result(timeout=600)
                entry = {
                    "step": step,
                    "epoch": epoch + 1,
                    "assistant_nll": nll,
                    "assistant_perplexity": batch_metrics["assistant_perplexity"],
                    "perplexity_overflow": batch_metrics["perplexity_overflow"],
                    "input_tokens": sum(d.model_input.length for d in datums),
                    "supervised_tokens": sum(
                        sum(d.loss_fn_inputs["weights"].data) for d in datums
                    ),
                    "learning_rate": learning_rate,
                    "seconds": perf_counter() - started,
                    "optimizer_metrics": optim.metrics,
                }
                optimization_seconds += entry["seconds"]
                log.write(json.dumps(entry, allow_nan=False) + "\n")
                log.flush()
                print(
                    json.dumps(
                        {k: entry[k] for k in ("step", "assistant_nll", "seconds")}
                    ),
                    flush=True,
                )
                if step < total_steps and step % args.eval_every == 0:
                    assess(
                        f"step_{step:04d}",
                        step,
                        generate_dev=step % args.generate_every == 0,
                    )
    report["training_loop_seconds"] = perf_counter() - start
    report["training_seconds"] = optimization_seconds
    report["state_path"] = (
        client.save_state("final", ttl_seconds=7 * 86400).result().path
    )
    write_json(args.output_dir / "run.json", report)
    assess("after", step, generate_train=True, generate_dev=True)
    output_tokens = sum(
        report[stage][split].get("eval/output_tokens_total", 0)
        for stage in report["evaluation_stages"]
        for split in ("train", "dev")
    )
    report["estimated_compute_usd"] = (
        report["cost"]["estimated_usd_bound"]
        - (report["cost"]["generation_output_token_bound"] - output_tokens)
        * SAMPLE_RATE
        / 1e6
    )
    report["checks"] = {
        "train_nll_decreased": report["after"]["train"]["assistant_nll"]
        < report["before"]["train"]["assistant_nll"],
        "completed_optimizer_steps": step,
        "saved_sampler_changed": report["sampler_paths"]["before"]
        != report["sampler_paths"]["after"],
    }
    # A decrease verifies learning on these examples; it is not a generalization claim.
    report["status"] = "completed"
    write_json(args.output_dir / "run.json", report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "train-manifest",
        "dev-manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
        "official-repo",
    ):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=4)
    p.add_argument("--generate-every", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--max-sequence-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--max-train-examples", type=int, default=8)
    p.add_argument("--max-dev-examples", type=int, default=4)
    p.add_argument("--max-estimated-usd", type=float, default=0.5)
    return p


def main():
    args = parser().parse_args()
    for key, value in vars(args).items():
        if key == "warmup_ratio":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if args.generate_every % args.eval_every:
        raise ValueError("generate_every must be a multiple of eval_every")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            "Use a fresh output directory; automatic training resume is unsupported"
        )
    train_rows, dev_rows = read_jsonl(args.train_manifest), read_jsonl(
        args.dev_manifest
    )
    if (
        len(train_rows) > args.max_train_examples
        or len(dev_rows) > args.max_dev_examples
    ):
        raise ValueError("Manifest exceeds the explicitly configured example limits")
    validate_records(train_rows, dev_rows, args.data_root)
    scorer = OfficialScorer(args.official_repo)
    tokenizer, renderer = load_renderer(
        MODEL, PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    train, dev = [
        prepare_examples(
            rows, args.data_root, renderer, args.max_pixels, args.max_sequence_tokens
        )
        for rows in (train_rows, dev_rows)
    ]
    cost = estimate_cost(
        train,
        dev,
        args.epochs,
        args.max_new_tokens,
        args.batch_size,
        args.eval_every,
        args.generate_every,
    )
    if cost["estimated_usd_bound"] > args.max_estimated_usd:
        raise ValueError(
            "Estimated token-cost bound exceeds budget; no Tinker calls made"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "model": MODEL,
        "initialization": "fresh_lora_on_hosted_model",
        "hosted_weight_revision": None,
        "processor_revision": PROCESSOR_REVISION,
        "cookbook_revision": COOKBOOK_REVISION,
        "official_revision": REVISION,
        "metrics_version": VERSION,
        "tinker_version": version("tinker"),
        "train_manifest_sha256": digest(args.train_manifest),
        "dev_manifest_sha256": digest(args.dev_manifest),
        "prompt": PROMPT,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "optimizer_steps": math.ceil(len(train) / args.batch_size) * args.epochs,
        "wandb_enabled": False,
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
        "loss_reduction": "batch_supervised_token_mean",
        "learning_rate_schedule": "linear_warmup_then_constant",
        "warmup_steps": math.ceil(
            args.warmup_ratio * math.ceil(len(train) / args.batch_size) * args.epochs
        ),
        "cost": cost,
        "rates_usd_per_million": {
            "training": TRAIN_RATE,
            "forward": FORWARD_RATE,
            "sample": SAMPLE_RATE,
        },
    }
    write_json(args.output_dir / "run.json", report)
    print(
        json.dumps({"status": "preflight", "steps": report["optimizer_steps"], **cost}),
        flush=True,
    )
    if args.execute:
        if args.env_file:
            from dotenv import load_dotenv

            load_dotenv(args.env_file, override=False)
        try:
            run(args, train, dev, tokenizer, renderer, scorer, report)
        except Exception as exc:
            report["status"] = "failed"
            report["error_type"] = type(exc).__name__
            write_json(args.output_dir / "run.json", report)
            raise


if __name__ == "__main__":
    main()
