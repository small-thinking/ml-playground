"""Cache fixed MoE-teacher rollouts and Top-10 targets. No API calls by default."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .evaluate import digest, read_jsonl
from .inference import PROMPT
from .kd_targets import soft_targets, topk_diagnostics
from .metrics import parse_table
from .sft import MODEL, validate_records
from .tinker_inference import (
    COOKBOOK_REVISION,
    PROCESSOR_REVISION,
    TEACHER_MODEL,
    TEACHER_REVISION,
    image_message,
    load_renderer,
)

TOP_K = 10
TEACHER_FORWARD_RATE, TEACHER_SAMPLE_RATE = 0.54, 1.335


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def cache_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def tokenizer_identity(student, teacher):
    """Compare the entire tokenizer, not just vocabulary size or model family."""
    left = json.loads(student.backend_tokenizer.to_str())
    right = json.loads(teacher.backend_tokenizer.to_str())
    if left != right or student.special_tokens_map != teacher.special_tokens_map:
        raise ValueError(
            "Teacher/student tokenization differs; token-level KD is unsafe"
        )
    return json_hash(left)


def completion_topk(response, prompt_length, completion_length):
    """Prompt-scoring position P predicts the first sampled token at position P."""
    arrays = response.topk_prompt_logprobs_np
    if arrays is None:
        raise ValueError("Teacher did not return Top-K prompt probabilities")
    expected = (prompt_length + completion_length, TOP_K)
    if arrays.token_ids.shape != expected or arrays.logprobs.shape != expected:
        raise ValueError("Unexpected full-sequence Top-K shape")
    # Image/prompt positions may be missing; only the completion is supervised.
    return arrays.token_ids[prompt_length:], arrays.logprobs[prompt_length:]


class CollectionBudget:
    """Reserve before submission; uncertain calls block automatic paid retries."""

    def __init__(self, path, maximum):
        self.path, self.maximum = path, maximum
        self.state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "estimated_compute_usd": 0.0,
                "pending": None,
                "calls": 0,
            }
        )
        if self.state["pending"] is not None:
            raise ValueError(
                "Uncertain paid request: reconcile the local journal before resuming"
            )

    def reserve(self, key, bound):
        # Reserve 10% for token accounting variance. Storage is budgeted separately.
        if (self.state["estimated_compute_usd"] + bound) * 1.1 > self.maximum:
            raise ValueError("Collection budget reached before next paid request")
        self.state["pending"] = {"key": key, "estimated_usd_bound": bound}
        atomic_json(self.path, self.state)

    def settle(self, amount):
        if (
            self.state["pending"] is None
            or amount > self.state["pending"]["estimated_usd_bound"] + 1e-9
        ):
            raise ValueError("Paid usage exceeds the reserved estimate")
        self.state["estimated_compute_usd"] += amount
        self.state["calls"] += 1
        self.state["pending"] = None
        atomic_json(self.path, self.state)


def cache_entry(directory, index, record, prompt, vocab_size):
    """Read back immutable targets; reject stale, edited or incomplete records."""
    metadata = json.loads((directory / f"{index:04d}.json").read_text())
    arrays_path = directory / f"{index:04d}.npz"
    rollout_path = directory / f"{index:04d}.rollout.json"
    if (
        metadata["record"] != record
        or metadata["arrays_sha256"] != digest(arrays_path)
        or metadata["rollout_sha256"] != digest(rollout_path)
        or record["prompt_sha256"] != json_hash(prompt.model_dump(mode="json"))
    ):
        raise ValueError("Cache content or rendered prompt identity mismatch")
    rollout = json.loads(rollout_path.read_text())
    if rollout["stop_reason"] != "stop":
        raise ValueError("Truncated teacher rollout is not a valid target")
    parse_table(rollout["html"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if arrays["token_ids"].shape[1] != TOP_K:
            raise ValueError("Expected Top-10 targets")
        datum = soft_targets(
            prompt,
            rollout["tokens"],
            arrays["token_ids"],
            arrays["logprobs"],
            vocab_size,
        )
        return datum, arrays["logprobs"].copy()


def collect_one(index, row, record, prompt, tokenizer, renderer, sampler, args, budget):
    import tinker

    directory = args.cache_dir
    output_path = directory / f"{index:04d}.rollout.json"
    if not output_path.exists():
        bound = (
            prompt.length * TEACHER_FORWARD_RATE
            + args.max_new_tokens * TEACHER_SAMPLE_RATE
        ) / 1e6
        budget.reserve(f"{index}:rollout", bound)
        result = sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=0,
                seed=args.seed,
                stop=renderer.get_stop_sequences(),
            ),
        ).result(timeout=600)
        sequence = result.sequences[0]
        rollout = {
            "record": record,
            "tokens": sequence.tokens,
            "html": tokenizer.decode(sequence.tokens, skip_special_tokens=True),
            "stop_reason": sequence.stop_reason,
        }
        atomic_json(output_path, rollout)
        budget.settle(
            (
                prompt.length * TEACHER_FORWARD_RATE
                + len(sequence.tokens) * TEACHER_SAMPLE_RATE
            )
            / 1e6
        )
    rollout = json.loads(output_path.read_text())
    if rollout["record"] != record:
        raise ValueError("Partial rollout belongs to a different prompt")
    if rollout["stop_reason"] != "stop":
        raise ValueError(
            "Teacher output truncated; preserved locally, no silent dropping/retry"
        )
    parse_table(rollout["html"])
    full = tinker.ModelInput(
        chunks=[*prompt.chunks, tinker.EncodedTextChunk(tokens=rollout["tokens"])]
    )
    if full.length > args.max_sequence_tokens:
        raise ValueError("Teacher sequence exceeds configured training context")
    cost = (full.length * TEACHER_FORWARD_RATE + TEACHER_SAMPLE_RATE) / 1e6
    budget.reserve(f"{index}:topk", cost)
    result = sampler.sample(
        prompt=full,
        num_samples=1,
        include_prompt_logprobs=True,
        topk_prompt_logprobs=TOP_K,
        sampling_params=tinker.SamplingParams(
            max_tokens=1, temperature=0, seed=args.seed
        ),
    ).result(timeout=600)
    # Keep the reservation until validated targets are durable. A crash or bad
    # response must not authorize an automatic repeat of this paid request.
    if result.topk_prompt_logprobs_np is not None:
        np.savez_compressed(
            directory / f"{index:04d}.raw-topk.npz",
            token_ids=result.topk_prompt_logprobs_np.token_ids,
            logprobs=result.topk_prompt_logprobs_np.logprobs,
        )
    ids, probabilities = completion_topk(result, prompt.length, len(rollout["tokens"]))
    soft_targets(prompt, rollout["tokens"], ids, probabilities, len(tokenizer))
    arrays_path = directory / f"{index:04d}.npz"
    np.savez_compressed(arrays_path, token_ids=ids, logprobs=probabilities)
    atomic_json(
        directory / f"{index:04d}.json",
        {
            "record": record,
            "arrays_sha256": digest(arrays_path),
            "rollout_sha256": digest(output_path),
            "retained_mass": topk_diagnostics(probabilities),
        },
    )
    budget.settle(cost)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-manifest",
        "dev-manifest",
        "data-root",
        "cache-dir",
        "tinker-cookbook-dir",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--max-new-examples", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-sequence-tokens", type=int, default=16384)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--max-estimated-usd", type=float, default=0.25)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if any(
        not np.isfinite(v) or v <= 0
        for v in (
            args.limit,
            args.max_new_examples,
            args.max_new_tokens,
            args.max_sequence_tokens,
            args.max_pixels,
            args.max_estimated_usd,
        )
    ):
        raise ValueError("Limits must be finite and positive")
    rows, dev = read_jsonl(args.train_manifest), read_jsonl(args.dev_manifest)
    if (
        len(rows) != 800
        or len(dev) != 100
        or len({r["id"] for r in rows}) != 800
        or not 1 <= args.limit <= 800
    ):
        raise ValueError("Expected frozen Train800 / Dev100 and a valid subset limit")
    validate_records(rows, dev, args.data_root)
    rows = random.Random(args.seed).sample(rows, args.limit)
    student_tokenizer, student_renderer = load_renderer(
        MODEL, PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    tokenizer, renderer = load_renderer(
        TEACHER_MODEL, TEACHER_REVISION, args.tinker_cookbook_dir
    )
    token_hash = tokenizer_identity(student_tokenizer, tokenizer)
    prompts, records = [], []
    for row in rows:
        user = image_message(args.data_root / row["image"], args.max_pixels)
        prompt = renderer.build_generation_prompt([user])
        student = student_renderer.build_generation_prompt([user])
        fingerprint = json_hash(prompt.model_dump(mode="json"))
        if fingerprint != json_hash(student.model_dump(mode="json")):
            raise ValueError("Teacher/student image preprocessing or prompt differs")
        prompts.append(prompt)
        records.append(
            {
                "id": row["id"],
                "image_sha256": row["image_sha256"],
                "prompt_sha256": fingerprint,
            }
        )
    identity = {
        "teacher_model": TEACHER_MODEL,
        "teacher_processor_revision": TEACHER_REVISION,
        "student_model": MODEL,
        "student_processor_revision": PROCESSOR_REVISION,
        "cookbook_revision": COOKBOOK_REVISION,
        "tokenizer_sha256": token_hash,
        "train_manifest_sha256": digest(args.train_manifest),
        "dev_manifest_sha256": digest(args.dev_manifest),
        "top_k": TOP_K,
        "loss_temperature": 1,
        "rollout_temperature": 0,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "max_sequence_tokens": args.max_sequence_tokens,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "records": records,
    }
    with cache_lock(args.cache_dir):
        manifest = args.cache_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Cache identity changed; use a new cache directory")
        atomic_json(manifest, identity)
        budget = CollectionBudget(args.cache_dir / "usage.json", args.max_estimated_usd)
        missing = []
        for index, (record, prompt) in enumerate(zip(records, prompts, strict=True)):
            if (args.cache_dir / f"{index:04d}.json").exists():
                cache_entry(args.cache_dir, index, record, prompt, len(tokenizer))
            else:
                missing.append(index)
        print(
            json.dumps(
                {
                    "status": "preflight",
                    "selected": len(rows),
                    "cached": len(rows) - len(missing),
                    "next_examples": min(len(missing), args.max_new_examples),
                }
            ),
            flush=True,
        )
        if not args.execute:
            return
        if args.env_file:
            from dotenv import load_dotenv

            load_dotenv(args.env_file, override=False)
        import tinker

        sampler = tinker.ServiceClient().create_sampling_client(
            base_model=TEACHER_MODEL
        )
        for index in missing[: args.max_new_examples]:
            collect_one(
                index,
                rows[index],
                records[index],
                prompts[index],
                tokenizer,
                renderer,
                sampler,
                args,
                budget,
            )
            print(
                json.dumps(
                    {
                        "cached": index + 1,
                        "estimated_compute_usd": budget.state["estimated_compute_usd"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
