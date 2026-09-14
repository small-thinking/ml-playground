"""Compare saved SFT samplers on a complete external Dev manifest, locally logged."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import SimpleNamespace

from .evaluate import digest, evaluate, read_jsonl
from .official import OfficialScorer
from .sft import FORWARD_RATE, MODEL, SAMPLE_RATE, prepare_examples, write_json
from .sft_metrics import summarize_nll
from .tinker_inference import COOKBOOK_REVISION, PROCESSOR_REVISION, SEED, load_renderer


def estimate_cost(examples, max_new_tokens):
    # compute_logprobs processes the full sequence and generates one ignored token.
    nll_tokens = 2 * sum(d.model_input.length + 1 for _, _, d in examples)
    input_tokens = 2 * sum(p.length for _, p, _ in examples)
    output_tokens = 2 * len(examples) * max_new_tokens
    return {
        "nll_input_tokens": nll_tokens,
        "generation_input_tokens": input_tokens,
        "generation_output_token_bound": output_tokens,
        "estimated_compute_usd_bound": (
            (nll_tokens + input_tokens) * FORWARD_RATE
            + (output_tokens + 2 * len(examples)) * SAMPLE_RATE
        )
        / 1e6,
    }


def sample_one(sampler, example, tokenizer, stop, max_new_tokens, lock):
    import tinker

    row, prompt, _ = example
    start = perf_counter()
    response = sampler.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0, seed=SEED, stop=stop, max_tokens=max_new_tokens
        ),
    ).result(timeout=600)
    seq = response.sequences[0]
    with lock:
        html = tokenizer.decode(seq.tokens, skip_special_tokens=True)
    return {
        "id": row["id"],
        "html": html,
        "input_tokens": prompt.length,
        "output_tokens": len(seq.tokens),
        "stop_reason": seq.stop_reason,
        "latency_seconds": perf_counter() - start,
    }


def likelihood_one(sampler, example):
    row, _, datum = example
    targets = datum.loss_fn_inputs["target_tokens"].data
    full = datum.model_input.append_int(targets[-1])
    values = sampler.compute_logprobs(full).result(timeout=600)
    if len(values) != full.length:
        raise ValueError("Sampler probability sequence length mismatch")
    mask = datum.loss_fn_inputs["weights"].data
    selected = [lp for lp, w in zip(values[1:], mask, strict=True) if w]
    if not selected or any(
        lp is None or not math.isfinite(lp) or lp > 0 for lp in selected
    ):
        raise ValueError("Invalid supervised log probability")
    return {
        "id": row["id"],
        "token_logprobs": selected,
        "target_tokens": [t for t, w in zip(targets, mask, strict=True) if w],
    }


def collect(examples, function, path, concurrency):
    """Persist completions as they arrive; do not publish private rows to stdout."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool, path.open("x") as stream:
        # At most concurrency network requests execute simultaneously.
        futures = [pool.submit(function, example) for example in examples]
        rows = []
        try:
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if len(rows) % 10 == 0:
                    print(
                        json.dumps({"completed": len(rows), "total": len(examples)}),
                        flush=True,
                    )
        except BaseException:
            for future in futures:
                future.cancel()  # Do not run queued paid requests after a failure.
            raise
    return rows


def likelihood_summary(rows):
    # Adapt stored supervised positions to the same pure NLL/PPL calculation.
    datums = [
        SimpleNamespace(
            loss_fn_inputs={
                "weights": SimpleNamespace(data=[1] * len(r["token_logprobs"]))
            }
        )
        for r in rows
    ]
    output = SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": SimpleNamespace(data=r["token_logprobs"])} for r in rows
        ]
    )
    summary = summarize_nll(output, datums)
    summary.pop("per_example")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-run",
        "manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
        "official-repo",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--expected-examples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-estimated-usd", type=float, default=2.0)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if any(
        x <= 0
        for x in (
            args.expected_examples,
            args.max_new_tokens,
            args.max_pixels,
            args.concurrency,
        )
    ):
        raise ValueError("Counts and limits must be positive")
    if not math.isfinite(args.max_estimated_usd) or args.max_estimated_usd <= 0:
        raise ValueError("Budget must be positive and finite")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory")
    source = json.loads(args.source_run.read_text())
    if source["status"] != "completed" or source["model"] != MODEL:
        raise ValueError("Expected a completed compatible SFT run")
    if (
        source["processor_revision"] != PROCESSOR_REVISION
        or source["cookbook_revision"] != COOKBOOK_REVISION
    ):
        raise ValueError("Source run renderer revisions differ from this evaluator")
    records = read_jsonl(args.manifest)
    if len(records) != args.expected_examples or any(
        r.get("split") != "dev" for r in records
    ):
        raise ValueError("Expected the entire explicitly sized Dev manifest")
    for row in records:
        for field in ("image", "label"):
            data_path = (args.data_root / row[field]).resolve()
            if not data_path.is_relative_to(args.data_root.resolve()):
                raise ValueError("Data path escapes data root")
            if digest(data_path) != row.get(field + "_sha256"):
                raise ValueError("Image/label hash mismatch")
    scorer = OfficialScorer(args.official_repo)
    evaluate(records, args.data_root, {}, scorer)  # Validate all references first.
    tokenizer, renderer = load_renderer(
        MODEL, PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    examples = prepare_examples(
        records, args.data_root, renderer, args.max_pixels, 65536
    )
    cost = estimate_cost(examples, args.max_new_tokens)
    if cost["estimated_compute_usd_bound"] > args.max_estimated_usd:
        raise ValueError("Preflight exceeds the estimated compute budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "preflight",
        "source_run_sha256": digest(args.source_run),
        "manifest_sha256": digest(args.manifest),
        "examples": len(records),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "sampler_paths": {k: source["sampler_paths"][k] for k in ("before", "after")},
        "processor_revision": PROCESSOR_REVISION,
        "seed": SEED,
        "nll_method": "saved_sampler.compute_logprobs, full original reference HTML, assistant mask",
        "max_reference_sequence_tokens": max(
            d.model_input.length + 1 for _, _, d in examples
        ),
        "wandb_enabled": False,
        "cost": cost,
    }
    path = args.output_dir / "run.json"
    write_json(path, report)
    print(
        json.dumps({"status": "preflight", "examples": len(records), **cost}),
        flush=True,
    )
    if not args.execute:
        return
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    import tinker

    start = perf_counter()
    service = tinker.ServiceClient()
    lock = Lock()
    report["status"] = "running"
    write_json(path, report)
    try:
        for stage in ("before", "after"):
            sampler = service.create_sampling_client(
                model_path=report["sampler_paths"][stage]
            )
            likelihoods = collect(
                examples,
                lambda ex: likelihood_one(sampler, ex),
                args.output_dir / f"{stage}_likelihoods.jsonl",
                args.concurrency,
            )
            report[stage] = likelihood_summary(likelihoods)
            write_json(path, report)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "nll": report[stage]["assistant_nll"],
                        "ppl": report[stage]["assistant_perplexity"],
                    }
                ),
                flush=True,
            )
            predictions = collect(
                examples,
                lambda ex: sample_one(
                    sampler,
                    ex,
                    tokenizer,
                    renderer.get_stop_sequences(),
                    args.max_new_tokens,
                    lock,
                ),
                args.output_dir / f"{stage}_predictions.jsonl",
                args.concurrency,
            )
            details, metrics = evaluate(
                records, args.data_root, {r["id"]: r for r in predictions}, scorer
            )
            write_json(args.output_dir / f"{stage}_details.json", details)
            report[stage].update(metrics)
            write_json(path, report)
        output_tokens = sum(
            report[s]["eval/output_tokens_total"] for s in ("before", "after")
        )
        report["estimated_compute_usd"] = (
            cost["estimated_compute_usd_bound"]
            - (cost["generation_output_token_bound"] - output_tokens)
            * SAMPLE_RATE
            / 1e6
        )
        report["wall_seconds"] = perf_counter() - start
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        write_json(path, report)


if __name__ == "__main__":
    main()
