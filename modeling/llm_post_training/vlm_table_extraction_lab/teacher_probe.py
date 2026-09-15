"""Compare teachers on fixed historical student prefixes; never update a model."""

import argparse
import json
import math
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .evaluate import digest, read_jsonl
from .kd_collect import CollectionBudget, json_hash, tokenizer_identity
from .opd_targets import policy_datum
from .sft import write_json
from .teacher_config import LARGE_TEACHER_MODEL, TEACHERS, teacher_spec
from .tinker_inference import (
    PROCESSOR_REVISION,
    TEACHER_MODEL,
    image_message,
    load_renderer,
)


def feedback_summary(student, old, new):
    """Token-weighted diagnostics on identical prefixes, not a quality ranking."""
    student, old, new = [np.asarray(v, dtype=float) for v in (student, old, new)]
    if (
        student.ndim != 1
        or not student.size
        or old.shape != student.shape
        or new.shape != student.shape
        or any(not np.all(np.isfinite(v)) or np.any(v > 0) for v in (student, old, new))
    ):
        raise ValueError("Expected aligned finite token log probabilities")
    a, b = old - student, new - student
    informative = (np.abs(a) > 0.001) & (np.abs(b) > 0.001)
    return {
        "completion_tokens": int(student.size),
        "old_teacher_nll": float(-old.mean()),
        "new_teacher_nll": float(-new.mean()),
        "old_sampled_reverse_kl": float(-a.mean()),
        "new_sampled_reverse_kl": float(-b.mean()),
        "old_advantage_std": float(a.std()),
        "new_advantage_std": float(b.std()),
        "old_near_zero_fraction": float((np.abs(a) <= 0.001).mean()),
        "new_near_zero_fraction": float((np.abs(b) <= 0.001).mean()),
        "mean_abs_teacher_logprob_difference": float(np.abs(new - old).mean()),
        "informative_tokens": int(informative.sum()),
        "informative_sign_disagreement": (
            float((np.sign(a[informative]) != np.sign(b[informative])).mean())
            if informative.any()
            else None
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-run",
        "manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--model", choices=TEACHERS, default=LARGE_TEACHER_MODEL)
    p.add_argument("--examples", type=int, default=32)
    p.add_argument("--max-estimated-usd", type=float, default=0.6)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not math.isfinite(args.max_estimated_usd) or args.max_estimated_usd <= 0:
        raise ValueError("Budget must be positive and finite")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh output directory is required")
    source = json.loads(args.source_run.read_text())
    if source["status"] != "completed" or source["teacher_model"] != TEACHER_MODEL:
        raise ValueError("Expected completed original35B teacher run")
    if digest(args.manifest) != source["train_manifest_sha256"]:
        raise ValueError("Training manifest differs from source run")
    paths = sorted(
        path
        for path in (args.source_run.parent / "rollouts").glob("*.json")
        if re.fullmatch(r"\d{4}_\d{2}\.json", path.name)
    )
    if not 1 <= args.examples <= len(paths):
        raise ValueError("Invalid fixed prefix count")
    spec = teacher_spec(args.model)
    student, renderer = load_renderer(
        "Qwen/Qwen3.5-4B", PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    teacher, teacher_renderer = load_renderer(
        spec.model, spec.revision, args.tinker_cookbook_dir
    )
    identity = tokenizer_identity(student, teacher)
    records = {r["id"]: r for r in read_jsonl(args.manifest)}
    prepared = []
    for path in paths[: args.examples]:
        raw = json.loads(path.read_text())
        row = records[raw["id"]]
        image = (args.data_root / row["image"]).resolve()
        if (
            not image.is_relative_to(args.data_root.resolve())
            or digest(image) != raw["image_sha256"]
            or raw["image_sha256"] != row["image_sha256"]
        ):
            raise ValueError("Source image differs from manifest")
        message = image_message(image, source["config"]["max_pixels"])
        prompt = renderer.build_generation_prompt([message])
        other = teacher_renderer.build_generation_prompt([message])
        if (
            json_hash(prompt.model_dump(mode="json")) != raw["prompt_sha256"]
            or json_hash(other.model_dump(mode="json")) != raw["prompt_sha256"]
        ):
            raise ValueError("Historical prompt differs")
        if len(raw["teacher_full_logprobs"]) != prompt.length + len(raw["tokens"]):
            raise ValueError("Historical probabilities differ in length")
        old = raw["teacher_full_logprobs"][prompt.length :]
        policy_datum(prompt, raw["tokens"], raw["sampling_logprobs"], old, len(student))
        prepared.append((path, raw, prompt, old))
    cost = (
        sum(
            (prompt.length + len(raw["tokens"])) * spec.forward_rate + spec.sample_rate
            for _, raw, prompt, _ in prepared
        )
        / 1e6
    )
    if cost * 1.1 > args.max_estimated_usd:
        raise ValueError("Fixed-prefix estimate exceeds budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "preflight",
        "source_run_sha256": digest(args.source_run),
        "manifest_sha256": digest(args.manifest),
        "old_teacher_model": TEACHER_MODEL,
        "model": spec.model,
        "processor_revision": spec.revision,
        "tokenizer_sha256": identity,
        "examples": len(prepared),
        "selection": "first saved rollouts in original training order; no quality filtering",
        "estimated_compute_usd_bound": cost,
        "rates_usd_per_million": {
            "forward": spec.forward_rate,
            "sample": spec.sample_rate,
        },
    }
    write_json(args.output_dir / "run.json", report)
    if not args.execute:
        print(json.dumps(report))
        return
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    import tinker
    from tinker.lib.retry_handler import RetryConfig

    sampler = tinker.ServiceClient().create_sampling_client(
        base_model=spec.model, retry_config=RetryConfig(enable_retry_logic=False)
    )
    budget = CollectionBudget(args.output_dir / "usage.json", args.max_estimated_usd)
    started = perf_counter()
    report["status"] = "running"
    write_json(args.output_dir / "run.json", report)
    all_student, all_old, all_new = [], [], []
    try:
        for index, (path, raw, prompt, old) in enumerate(prepared):
            full = prompt.append(tinker.EncodedTextChunk(tokens=raw["tokens"]))
            amount = (full.length * spec.forward_rate + spec.sample_rate) / 1e6
            budget.reserve(f"prefix_{index}", amount)
            new = sampler.compute_logprobs(full).result(timeout=600)
            write_json(
                args.output_dir / f"prefix_{index:03d}.json",
                {
                    "id": raw["id"],
                    "source_rollout_sha256": digest(path),
                    "prompt_tokens": prompt.length,
                    "new_teacher_full_logprobs": new,
                },
            )
            if len(new) != full.length:
                raise ValueError("Teacher probability length differs")
            new = new[prompt.length :]
            feedback_summary(raw["sampling_logprobs"], old, new)
            budget.settle(amount)
            all_student.extend(raw["sampling_logprobs"])
            all_old.extend(old)
            all_new.extend(new)
        report.update(
            status="completed",
            metrics=feedback_summary(all_student, all_old, all_new),
            estimated_compute_usd=budget.state["estimated_compute_usd"],
            wall_seconds=perf_counter() - started,
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        write_json(args.output_dir / "run.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "examples": report["examples"],
                "estimated_compute_usd": report["estimated_compute_usd"],
            }
        )
    )


if __name__ == "__main__":
    previous_umask = os.umask(0o077)
    try:
        main()
    finally:
        os.umask(previous_umask)
