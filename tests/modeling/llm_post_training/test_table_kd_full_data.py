"""Full-data KD keeps valid token trajectories and imports caches by identity."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import kd, kd_collect as mod
from modeling.llm_post_training.vlm_table_extraction_lab.kd_cache_reuse import (
    reuse_cache,
)
from .test_table_kd_cache import cache_fixture, response
from .test_table_kd_rejections import loader_fixture, reject_fixture


def snapshot(directory):
    return {p.name: mod.digest(p) for p in directory.iterdir() if p.is_file()}


def rewrite_rollout(directory, index, **changes):
    path = directory / f"{index:04d}.rollout.json"
    rollout = json.loads(path.read_text())
    rollout.update(changes)
    mod.atomic_json(path, rollout)
    metadata_path = directory / f"{index:04d}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["rollout_sha256"] = mod.digest(path)
        mod.atomic_json(metadata_path, metadata)
    return rollout


@pytest.mark.parametrize("stop", ["stop", "length"])
def test_invalid_trajectory_keeps_tokens_and_collects_only_missing_topk(tmp_path, stop):
    prompt, record = cache_fixture(tmp_path)
    rewrite_rollout(tmp_path, 0, html="unfinished table", stop_reason=stop)
    original = (tmp_path / "0000.rollout.json").read_bytes()
    (tmp_path / "0000.json").unlink()
    (tmp_path / "0000.npz").unlink()
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        assert kwargs["topk_prompt_logprobs"] == mod.TOP_K
        assert kwargs["sampling_params"].max_tokens == 1
        # Teacher-forcing preserves the sampled prefix; no EOS/repair is appended.
        assert kwargs["prompt"].chunks[-1].tokens == [4, 5]
        assert kwargs["prompt"].length == prompt.length + 2
        return SimpleNamespace(result=lambda **kw: response(prompt.length))

    class Tokenizer:
        def __len__(self):
            return 20

    args = SimpleNamespace(
        cache_dir=tmp_path,
        max_sequence_tokens=100,
        seed=42,
        include_invalid_rollouts=True,
    )
    budget = mod.CollectionBudget(tmp_path / "usage.json", maximum=1)
    assert mod.collect_one(
        0,
        {},
        record,
        prompt,
        Tokenizer(),
        None,
        SimpleNamespace(sample=sample),
        args,
        budget,
    )
    assert len(calls) == budget.state["calls"] == 1
    assert (tmp_path / "0000.rollout.json").read_bytes() == original
    assert not (tmp_path / "0000.rejected.json").exists()
    metadata = json.loads((tmp_path / "0000.json").read_text())
    assert metadata["quality"] == {"format_valid": False, "truncated": stop == "length"}
    datum, lps = mod.cache_entry(tmp_path, 0, record, prompt, 20, include_invalid=True)
    assert datum.model_input.length == prompt.length + 1
    assert datum.model_input.chunks[-1].tokens == [4]
    assert lps.shape == (2, mod.TOP_K)
    assert datum.loss_fn_inputs["weights"].to_numpy().sum() == pytest.approx(2)
    with pytest.raises(ValueError):
        mod.cache_entry(tmp_path, 0, record, prompt, 20)


@pytest.mark.parametrize(
    "changes", [{"tokens": []}, {"stop_reason": "error"}, {"stop_reason": None}]
)
def test_empty_or_unknown_stop_cannot_bypass_technical_validity(tmp_path, changes):
    prompt, record = cache_fixture(tmp_path)
    rewrite_rollout(tmp_path, 0, **changes)
    with pytest.raises(ValueError, match="Empty or unknown"):
        mod.cache_entry(tmp_path, 0, record, prompt, 20, include_invalid=True)
    args = SimpleNamespace(cache_dir=tmp_path, include_invalid_rollouts=True)
    with pytest.raises(ValueError, match="Empty or unknown"):
        mod.collect_one(
            0,
            {},
            record,
            prompt,
            None,
            None,
            SimpleNamespace(sample=lambda **kw: pytest.fail("No paid call")),
            args,
            None,
        )


def source_fixture(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "full"
    source.mkdir()
    destination.mkdir()
    (source / ".lock").touch()
    prompt, accepted = cache_fixture(source)
    accepted = dict(accepted)
    rejected = {**accepted, "id": "old-rejected", "image_sha256": "second-image"}
    new = {**accepted, "id": "new-candidate", "image_sha256": "new-image"}
    reject_fixture(source, index=1, record=rejected, stop="length")
    raw = response(prompt.length).topk_prompt_logprobs_np
    np.savez_compressed(
        source / "0000.raw-topk.npz", token_ids=raw.token_ids, logprobs=raw.logprobs
    )
    old_identity = {
        "teacher_model": "teacher",
        "student_model": "student",
        "seed": 42,
        "top_k": mod.TOP_K,
        "max_new_tokens": 8192,
        "records": [accepted, rejected],
    }
    mod.atomic_json(source / "manifest.json", old_identity)
    mod.atomic_json(
        source / "usage.json",
        {"pending": None, "estimated_compute_usd": 0.1, "calls": 3},
    )
    identity = {
        **old_identity,
        "include_invalid_rollouts": True,
        "records": [rejected, new, accepted],
    }
    return source, destination, identity, [prompt] * 3


def test_reuse_reorders_by_record_id_preserves_source_and_does_not_copy_rejection(
    tmp_path,
):
    source, destination, identity, prompts = source_fixture(tmp_path)
    before = snapshot(source)
    reuse_cache(source, destination, identity, prompts, 20)
    assert snapshot(source) == before
    assert mod.digest(destination / "0002.npz") == mod.digest(source / "0000.npz")
    assert mod.digest(destination / "0002.rollout.json") == mod.digest(
        source / "0000.rollout.json"
    )
    assert mod.digest(destination / "0000.rollout.json") == mod.digest(
        source / "0001.rollout.json"
    )
    assert not (destination / "0000.rejected.json").exists()
    assert not (destination / "0000.npz").exists()
    assert not (destination / "0001.rollout.json").exists()
    datum, _ = mod.cache_entry(
        destination, 2, identity["records"][2], prompts[2], 20, include_invalid=True
    )
    assert datum.model_input.chunks[-1].tokens == [4]
    provenance = json.loads((destination / "reuse.json").read_text())
    assert [
        (r["index"], r["source_index"], r["complete"]) for r in provenance["imports"]
    ] == [(0, 1, False), (2, 0, True)]
    assert provenance["source_estimated_compute_usd"] == 0.1
    assert provenance["imports"][0]["source_rejection_sha256"] == mod.digest(
        source / "0001.rejected.json"
    )
    imported = snapshot(destination)
    reuse_cache(source, destination, identity, prompts, 20)
    assert snapshot(source) == before and snapshot(destination) == imported


@pytest.mark.parametrize(
    "damage",
    ["pending", "protocol", "arrays_hash", "rollout_hash", "record", "ambiguous"],
)
def test_reuse_rejects_unverified_source_without_modifying_it(tmp_path, damage):
    source, destination, identity, prompts = source_fixture(tmp_path)
    if damage == "pending":
        mod.atomic_json(
            source / "usage.json",
            {"pending": {"key": "uncertain"}, "estimated_compute_usd": 0.1},
        )
    elif damage == "protocol":
        identity["seed"] += 1
    elif damage == "arrays_hash":
        with (source / "0000.npz").open("ab") as stream:
            stream.write(b"edited")
    elif damage == "rollout_hash":
        with (source / "0000.rollout.json").open("a") as stream:
            stream.write(" ")
    elif damage == "record":
        identity["records"][2] = {
            **identity["records"][2],
            "image_sha256": "changed-image",
        }
    else:
        mod.atomic_json(source / "0000.rejected.json", {"ambiguous": True})
    before = snapshot(source)
    with pytest.raises(ValueError):
        reuse_cache(source, destination, identity, prompts, 20)
    assert snapshot(source) == before


def test_reuse_refuses_changed_destination_content(tmp_path):
    source, destination, identity, prompts = source_fixture(tmp_path)
    reuse_cache(source, destination, identity, prompts, 20)
    (destination / "0002.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing imported"):
        reuse_cache(source, destination, identity, prompts, 20)


def include_loader_fixture(tmp_path, monkeypatch):
    args, tokenizer, renderer, records = loader_fixture(tmp_path, monkeypatch)
    args.include_invalid_rollouts = True
    args.skip_rejected = False
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["include_invalid_rollouts"] = True
    mod.atomic_json(path, manifest)
    rewrite_rollout(tmp_path, 0, html="unclosed output", stop_reason="length")
    mod.atomic_json(
        tmp_path / "0001.rollout.json",
        {
            "record": records[1],
            "tokens": [4],
            "html": "not HTML",
            "stop_reason": "stop",
        },
    )
    np.savez_compressed(
        tmp_path / "0001.npz",
        token_ids=np.arange(mod.TOP_K)[None, :],
        logprobs=np.full((1, mod.TOP_K), np.log(0.099)),
    )
    mod.atomic_json(
        tmp_path / "0001.json",
        {
            "record": records[1],
            "rollout_sha256": mod.digest(tmp_path / "0001.rollout.json"),
            "arrays_sha256": mod.digest(tmp_path / "0001.npz"),
        },
    )
    return args, tokenizer, renderer


def test_full_loader_retains_every_candidate_and_reports_overlapping_quality_counts(
    tmp_path, monkeypatch
):
    args, tokenizer, renderer = include_loader_fixture(tmp_path, monkeypatch)
    datums, _, metadata = kd.load_train(args, tokenizer, renderer)
    assert len(datums) == args.train_examples == metadata["candidate_examples"] == 2
    assert metadata["rejected_examples"] == 0
    assert metadata["teacher_format_invalid_examples"] == 2
    assert metadata["teacher_truncated_examples"] == 1
    assert metadata["target_filter_policy"] == "include_all_token_valid_rollouts"
    # Quality categories overlap; they are not exclusions and need not sum to N.
    (tmp_path / "0001.json").unlink()
    with pytest.raises(FileNotFoundError):
        kd.load_train(args, tokenizer, renderer)


def test_loader_requires_explicit_matching_include_policy(tmp_path, monkeypatch):
    args, tokenizer, renderer = include_loader_fixture(tmp_path, monkeypatch)
    args.include_invalid_rollouts = False
    with pytest.raises(ValueError, match="policies differ"):
        kd.load_train(args, tokenizer, renderer)
