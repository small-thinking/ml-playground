"""Import verified teacher outputs by identity, without additional API calls."""

import json
import shutil

from .evaluate import digest
from .kd_collect import atomic_json, cache_entry, cache_lock, rejected_entry


def reuse_cache(source, destination, identity, prompts, vocab_size):
    if source.resolve() == destination.resolve():
        raise ValueError("Reuse requires a different, new cache directory")
    with cache_lock(source):
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key, value in identity.items():
            if (
                key not in {"records", "include_invalid_rollouts"}
                and manifest.get(key) != value
            ):
                raise ValueError(f"Source cache protocol differs: {key}")
        usage_path = source / "usage.json"
        usage = json.loads(usage_path.read_text())
        if usage["pending"] is not None:
            raise ValueError("Cannot reuse cache with unresolved paid requests")
        source_records = manifest["records"]
        by_id = {r["id"]: (i, r) for i, r in enumerate(source_records)}
        if len(by_id) != len(source_records):
            raise ValueError("Source cache has duplicate records")
        imports = []
        for index, (record, prompt) in enumerate(
            zip(identity["records"], prompts, strict=True)
        ):
            if record["id"] not in by_id:
                continue
            old_index, old_record = by_id[record["id"]]
            if old_record != record:
                raise ValueError("Source record image/prompt identity differs")
            accepted = (source / f"{old_index:04d}.json").exists()
            rejection_path = source / f"{old_index:04d}.rejected.json"
            if accepted:
                if rejection_path.exists():
                    raise ValueError("Source entry is both accepted and rejected")
                cache_entry(
                    source,
                    old_index,
                    record,
                    prompt,
                    vocab_size,
                    include_invalid=bool(manifest.get("include_invalid_rollouts")),
                )
                suffixes = [".rollout.json", ".npz", ".raw-topk.npz", ".json"]
            elif rejection_path.exists() and identity.get("include_invalid_rollouts"):
                rejected_entry(source, old_index, record)
                suffixes = [".rollout.json"]
            else:
                raise ValueError("Source cache entry is incomplete")
            hashes = {}
            for suffix in suffixes:
                original = source / f"{old_index:04d}{suffix}"
                target = destination / f"{index:04d}{suffix}"
                hashes[suffix] = digest(original)
                if target.exists():
                    if digest(target) != hashes[suffix]:
                        raise ValueError("Existing imported cache differs")
                else:
                    temporary = target.with_suffix(target.suffix + ".importing")
                    shutil.copyfile(original, temporary)
                    temporary.replace(target)
            imports.append(
                {
                    "index": index,
                    "source_index": old_index,
                    "complete": accepted,
                    "hashes": hashes,
                    "source_rejection_sha256": (
                        digest(rejection_path) if rejection_path.exists() else None
                    ),
                }
            )
        provenance = {
            "source_manifest_sha256": digest(manifest_path),
            "source_usage_sha256": digest(usage_path),
            "source_estimated_compute_usd": usage["estimated_compute_usd"],
            "imports": imports,
        }
        path = destination / "reuse.json"
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError("Cache reuse provenance changed")
        atomic_json(path, provenance)
