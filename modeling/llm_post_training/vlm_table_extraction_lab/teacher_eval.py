"""Capped, locally recorded Dev evaluation of an explicitly priced visual teacher.

Every paid request is reserved before submission and persisted before settlement.
There are no retries or resume: a failed/uncertain request keeps its reservation.
The worst-case total is informational; the actual cap is enforced per request.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from time import perf_counter
from threading import Event, Lock

from .checkpoint_eval import likelihood_one, likelihood_summary
from .evaluate import digest, evaluate, read_jsonl
from .inference import PROMPT
from .kd_collect import cache_lock, json_hash
from .kd_collection_budget import ConcurrentBudget
from .metrics import VERSION
from .official import OfficialScorer, REVISION
from .sft import prepare_examples
from .teacher_config import LARGE_TEACHER_MODEL, TEACHERS, teacher_spec
from .tinker_inference import COOKBOOK_REVISION, SEED, load_renderer


def private_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def paid_request(budget, key, bound, path, function, actual_cost):
    """A saved response without a settled journal deliberately blocks replay."""
    if path.exists():
        raise ValueError("Request output exists; do not replay paid requests")
    budget.reserve(key, bound)
    row = function()
    amount = actual_cost(row)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Invalid request accounting")
    private_json(path, row)
    budget.settle(amount)
    return row


def validate_manifest(manifest, data_root, expected):
    records = read_jsonl(manifest)
    if len(records) != expected or any(r.get("split") != "dev" for r in records):
        raise ValueError("Expected the entire explicitly sized Dev manifest")
    for row in records:
        for field in ("image", "label"):
            path = (data_root / row[field]).resolve()
            if not path.is_relative_to(data_root.resolve()):
                raise ValueError("Data path escapes data root")
            if digest(path) != row.get(field + "_sha256"):
                raise ValueError("Image/label hash mismatch")
    return records


def costs(examples, spec, max_new_tokens):
    nll_tokens = sum(d.model_input.length + 1 for _, _, d in examples)
    input_tokens = sum(p.length for _, p, _ in examples)
    bound = len(examples) * max_new_tokens
    return {
        "nll_input_tokens": nll_tokens,
        "generation_input_tokens": input_tokens,
        "generation_output_token_bound": bound,
        # Tinker compute_logprobs also samples one ignored token per call.
        "estimated_compute_usd_bound": (
            (nll_tokens + input_tokens) * spec.forward_rate
            + (bound + len(examples)) * spec.sample_rate
        )
        / 1e6,
    }


def sample_one(sampler, example, tokenizer, stop, max_tokens, seed, lock):
    import tinker

    row, prompt, _ = example
    started = perf_counter()
    response = sampler.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0, seed=seed, stop=stop, max_tokens=max_tokens
        ),
    ).result(timeout=600)
    sequence = response.sequences[0]
    tokens = list(sequence.tokens)
    if (
        not tokens
        or len(tokens) > max_tokens
        or sequence.stop_reason not in {"stop", "length"}
    ):
        raise ValueError("Invalid generation completion")
    logprobs = sequence.logprobs
    if logprobs is not None and (
        len(logprobs) != len(tokens)
        or any(not math.isfinite(lp) or lp > 0 for lp in logprobs)
    ):
        raise ValueError("Invalid generation log probabilities")
    with lock:
        html = tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "id": row["id"],
        "html": html,
        "tokens": tokens,
        "token_logprobs": logprobs,
        "input_tokens": prompt.length,
        "output_tokens": len(tokens),
        "stop_reason": sequence.stop_reason,
        "latency_seconds": perf_counter() - started,
    }


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory; no automatic paid resume")
    spec = teacher_spec(args.model)
    records = validate_manifest(args.manifest, args.data_root, args.expected_examples)
    scorer = OfficialScorer(args.official_repo)
    evaluate(records, args.data_root, {}, scorer)
    tokenizer, renderer = load_renderer(
        args.model, spec.revision, args.tinker_cookbook_dir
    )
    examples = prepare_examples(
        records, args.data_root, renderer, args.max_pixels, 65536
    )
    if any(prompt.length + args.max_new_tokens > 65536 for _, prompt, _ in examples):
        raise ValueError("Generation exceeds the priced 64K context")
    cost = costs(examples, spec, args.max_new_tokens)
    report = {
        "status": "preflight",
        "model": args.model,
        "hosted_weight_revision": None,
        "processor_revision": spec.revision,
        "cookbook_revision": COOKBOOK_REVISION,
        "manifest_sha256": digest(args.manifest),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "tokenizer_sha256": json_hash(json.loads(tokenizer.backend_tokenizer.to_str())),
        "official_revision": REVISION,
        "metrics_version": VERSION,
        "script_sha256": digest(Path(__file__)),
        "sdk_version": version("tinker"),
        "examples": len(examples),
        "seed": args.seed,
        "renderer": "qwen3_5_disable_thinking",
        "temperature": 0,
        "wandb_enabled": False,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "rates_usd_per_million_tokens": {
            "prefill": spec.forward_rate,
            "sample": spec.sample_rate,
        },
        "cost": cost,
        "nll_method": "compute_logprobs, full original reference HTML, assistant mask",
    }
    with cache_lock(args.output_dir):
        args.output_dir.chmod(0o700)
        report_path = args.output_dir / "run.json"
        private_json(report_path, report)
        print(
            json.dumps({"status": "preflight", "examples": len(examples), **cost}),
            flush=True,
        )
        if not args.execute:
            return
        if args.env_file:
            from dotenv import load_dotenv

            load_dotenv(args.env_file, override=False)
        import tinker
        from tinker.lib.retry_handler import RetryConfig

        budget = ConcurrentBudget(
            args.output_dir / "usage.json", args.max_estimated_usd
        )
        started = perf_counter()
        report["status"] = "running"
        private_json(report_path, report)
        try:
            sampler = tinker.ServiceClient().create_sampling_client(
                base_model=args.model,
                retry_config=RetryConfig(enable_retry_logic=False),
            )
            likelihoods, predictions = [], []
            stopped, token_lock = Event(), Lock()

            def collect_one(index, example):
                request_budget = budget.for_example()
                try:
                    if stopped.is_set():
                        raise RuntimeError("Cancelled before paid request")
                    nll_cost = (
                        (example[2].model_input.length + 1) * spec.forward_rate
                        + spec.sample_rate
                    ) / 1e6
                    likelihood = paid_request(
                        request_budget,
                        f"nll_{index:04d}",
                        nll_cost,
                        args.output_dir / f"{index:04d}_likelihood.json",
                        lambda: likelihood_one(sampler, example),
                        lambda row: nll_cost,
                    )
                    if stopped.is_set():
                        raise RuntimeError("Cancelled before paid request")
                    bound = (
                        example[1].length * spec.forward_rate
                        + args.max_new_tokens * spec.sample_rate
                    ) / 1e6
                    prediction = paid_request(
                        request_budget,
                        f"generation_{index:04d}",
                        bound,
                        args.output_dir / f"{index:04d}_prediction.json",
                        lambda: sample_one(
                            sampler,
                            example,
                            tokenizer,
                            renderer.get_stop_sequences(),
                            args.max_new_tokens,
                            args.seed,
                            token_lock,
                        ),
                        lambda row: (
                            row["input_tokens"] * spec.forward_rate
                            + row["output_tokens"] * spec.sample_rate
                        )
                        / 1e6,
                    )
                    return likelihood, prediction
                except BaseException:
                    stopped.set()
                    raise

            # Submit only one bounded chunk at a time; failure cancels future chunks.
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for start in range(0, len(examples), args.concurrency):
                    futures = [
                        pool.submit(collect_one, index, examples[index])
                        for index in range(
                            start, min(start + args.concurrency, len(examples))
                        )
                    ]
                    try:
                        for future in as_completed(futures):
                            likelihood, prediction = future.result()
                            likelihoods.append(likelihood)
                            predictions.append(prediction)
                    except BaseException:
                        stopped.set()
                        for future in futures:
                            future.cancel()
                        raise
                    report["completed_examples"] = len(predictions)
                    report["estimated_compute_usd"] = budget.state[
                        "estimated_compute_usd"
                    ]
                    private_json(report_path, report)
                    print(
                        json.dumps(
                            {
                                "completed_examples": len(predictions),
                                "estimated_compute_usd": report[
                                    "estimated_compute_usd"
                                ],
                            }
                        ),
                        flush=True,
                    )
            details, metrics = evaluate(
                records, args.data_root, {r["id"]: r for r in predictions}, scorer
            )
            private_json(args.output_dir / "details.json", details)
            report["metrics"] = {**likelihood_summary(likelihoods), **metrics}
            report["output_hashes"] = {
                p.name: digest(p)
                for p in sorted(args.output_dir.glob("*.json"))
                if p.name not in {"run.json", "usage.json"}
            }
            report["status"] = "completed"
        except BaseException as exc:
            report["status"] = "failed"
            report["error_type"] = type(exc).__name__
            raise
        finally:
            report["estimated_compute_usd"] = budget.state["estimated_compute_usd"]
            report["pending_request"] = budget.state["pending"]
            report["wall_seconds"] = perf_counter() - started
            private_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
        "official-repo",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", choices=TEACHERS, default=LARGE_TEACHER_MODEL)
    parser.add_argument("--expected-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-estimated-usd", type=float, default=2.5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.expected_examples,
            args.max_new_tokens,
            args.max_pixels,
            args.concurrency,
        )
        <= 0
    ):
        raise ValueError("Counts must be positive")
    if not math.isfinite(args.max_estimated_usd) or args.max_estimated_usd <= 0:
        raise ValueError("Budget must be positive and finite")
    previous_umask = os.umask(0o077)
    try:
        run(args)
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
