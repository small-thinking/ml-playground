"""Create deterministic, disjoint GSM8K split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv


DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_SEED = 20260901
DEFAULT_SFT_COUNT = 512
DEFAULT_RL_COUNT = 1500
DEFAULT_EVAL_COUNT = 256
DEFAULT_MANIFEST_PATH = Path(__file__).parent / "manifests" / "gsm8k_splits.json"
DEFAULT_PROFILE_PATH = Path(__file__).parent / "manifests" / "gsm8k_profile.json"
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class ManifestError(ValueError):
    """Raised when a split manifest cannot be constructed safely."""


@dataclass(frozen=True)
class SplitManifest:
    """Content IDs and provenance for one immutable data partition."""

    schema_version: int
    dataset_id: str
    dataset_config: str
    dataset_revision: str
    seed: int
    sft_ids: Tuple[str, ...]
    rl_ids: Tuple[str, ...]
    eval_ids: Tuple[str, ...]
    manifest_hash: str

    def validate(self) -> None:
        all_ids = self.sft_ids + self.rl_ids + self.eval_ids
        if len(set(all_ids)) != len(all_ids):
            raise ManifestError("SFT, RL, and evaluation IDs must be disjoint")
        if not all(self.sft_ids) or not all(self.rl_ids) or not all(self.eval_ids):
            raise ManifestError("each split must contain at least one example")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def content_id(row: Mapping[str, object]) -> str:
    """Return a stable ID without storing GSM8K text in the manifest."""
    try:
        payload = {
            "answer": str(row["answer"]).strip(),
            "question": str(row["question"]).strip(),
        }
    except KeyError as exc:
        raise ManifestError("GSM8K rows require question and answer fields") from exc
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"gsm8k-{hashlib.sha256(encoded).hexdigest()}"


def _ordered_ids(
    rows: Iterable[Mapping[str, object]], seed: int, label: str
) -> Tuple[str, ...]:
    ids = tuple(content_id(row) for row in rows)
    if len(set(ids)) != len(ids):
        raise ManifestError("source rows produced duplicate content IDs")
    return tuple(
        sorted(
            ids,
            key=lambda example_id: hashlib.sha256(
                f"{seed}:{label}:{example_id}".encode()
            ).hexdigest(),
        )
    )


def _manifest_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    train_rows: Iterable[Mapping[str, object]],
    test_rows: Iterable[Mapping[str, object]],
    dataset_revision: str,
    seed: int = DEFAULT_SEED,
    sft_count: int = DEFAULT_SFT_COUNT,
    rl_count: int = DEFAULT_RL_COUNT,
    eval_count: int = DEFAULT_EVAL_COUNT,
) -> SplitManifest:
    """Partition official train/test rows without retaining their raw text."""
    if not dataset_revision:
        raise ManifestError("dataset revision is required")
    if min(sft_count, rl_count, eval_count) <= 0:
        raise ManifestError("split counts must be positive")

    train_ids = _ordered_ids(train_rows, seed, "train")
    eval_ids = _ordered_ids(test_rows, seed, "test")
    required_train = sft_count + rl_count
    if len(train_ids) < required_train:
        raise ManifestError(f"need {required_train} train rows, found {len(train_ids)}")
    if len(eval_ids) < eval_count:
        raise ManifestError(f"need {eval_count} test rows, found {len(eval_ids)}")

    payload = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "sft_ids": train_ids[:sft_count],
        "rl_ids": train_ids[sft_count:required_train],
        "eval_ids": eval_ids[:eval_count],
    }
    manifest = SplitManifest(manifest_hash=_manifest_hash(payload), **payload)
    manifest.validate()
    return manifest


def write_manifest(manifest: SplitManifest, path: Path) -> None:
    """Write the small, reviewable manifest without raw questions or answers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> SplitManifest:
    """Load a manifest only when its content hash and split invariants agree."""
    try:
        payload = json.loads(path.read_text())
        manifest = SplitManifest(
            schema_version=int(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            dataset_config=str(payload["dataset_config"]),
            dataset_revision=str(payload["dataset_revision"]),
            seed=int(payload["seed"]),
            sft_ids=tuple(payload["sft_ids"]),
            rl_ids=tuple(payload["rl_ids"]),
            eval_ids=tuple(payload["eval_ids"]),
            manifest_hash=str(payload["manifest_hash"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest at {path}") from exc

    manifest.validate()
    expected_hash = _manifest_hash(
        {
            "schema_version": manifest.schema_version,
            "dataset_id": manifest.dataset_id,
            "dataset_config": manifest.dataset_config,
            "dataset_revision": manifest.dataset_revision,
            "seed": manifest.seed,
            "sft_ids": manifest.sft_ids,
            "rl_ids": manifest.rl_ids,
            "eval_ids": manifest.eval_ids,
        }
    )
    if manifest.manifest_hash != expected_hash:
        raise ManifestError("manifest hash does not match its content")
    return manifest


def select_rows(
    rows: Iterable[Mapping[str, object]], example_ids: Sequence[str]
) -> Tuple[Mapping[str, object], ...]:
    """Return source rows in manifest order, failing on any provenance drift."""
    indexed = {content_id(row): row for row in rows}
    if len(indexed) < len(example_ids):
        raise ManifestError("source rows are incomplete or contain duplicate IDs")
    missing = [example_id for example_id in example_ids if example_id not in indexed]
    if missing:
        raise ManifestError(f"source rows are missing manifest ID {missing[0]}")
    return tuple(indexed[example_id] for example_id in example_ids)


def load_official_eval_rows(
    manifest: SplitManifest,
) -> Tuple[Mapping[str, object], ...]:
    """Fetch the pinned GSM8K test revision and recover the held-out rows."""
    from datasets import load_dataset

    load_dotenv(ENV_FILE, override=False)
    test_rows = load_dataset(
        manifest.dataset_id,
        manifest.dataset_config,
        split="test",
        revision=manifest.dataset_revision,
    )
    return select_rows(test_rows, manifest.eval_ids)


def dataset_profile(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, Union[float, int]]:
    """Summarize lengths and answer-marker coverage without retaining text."""
    if not rows:
        raise ManifestError("cannot profile an empty split")
    question_lengths = sorted(len(str(row["question"])) for row in rows)
    answer_lengths = sorted(len(str(row["answer"])) for row in rows)

    def percentile(values: Sequence[int], fraction: float) -> int:
        return values[round((len(values) - 1) * fraction)]

    return {
        "examples": len(rows),
        "answer_marker_fraction": sum("####" in str(row["answer"]) for row in rows)
        / len(rows),
        "question_chars_p50": percentile(question_lengths, 0.5),
        "question_chars_p90": percentile(question_lengths, 0.9),
        "answer_chars_p50": percentile(answer_lengths, 0.5),
        "answer_chars_p90": percentile(answer_lengths, 0.9),
    }


def write_profile(
    dataset_revision: str,
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Write reviewable aggregate data facts next to the split manifest."""
    profile = {
        "dataset_config": DATASET_CONFIG,
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "test": dataset_profile(test_rows),
        "train": dataset_profile(train_rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")


def prepare_official_manifest(
    output_path: Path,
    profile_path: Path,
    seed: int = DEFAULT_SEED,
    sft_count: int = DEFAULT_SFT_COUNT,
    rl_count: int = DEFAULT_RL_COUNT,
    eval_count: int = DEFAULT_EVAL_COUNT,
) -> SplitManifest:
    """Fetch a pinned official revision and write its deterministic manifest."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    load_dotenv(ENV_FILE, override=False)
    revision = HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=revision)
    manifest = build_manifest(
        dataset["train"],
        dataset["test"],
        dataset_revision=revision,
        seed=seed,
        sft_count=sft_count,
        rl_count=rl_count,
        eval_count=eval_count,
    )
    write_manifest(manifest, output_path)
    write_profile(revision, dataset["train"], dataset["test"], profile_path)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a deterministic GSM8K split manifest."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--profile-output", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = prepare_official_manifest(
        args.output, args.profile_output, seed=args.seed
    )
    print(
        "wrote "
        f"{args.output} sft={len(manifest.sft_ids)} rl={len(manifest.rl_ids)} "
        f"eval={len(manifest.eval_ids)} overlap=0 hash={manifest.manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
