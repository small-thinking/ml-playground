"""Local regression checks for explicit teacher-target exclusion without replacement."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import kd, kd_collect as mod

tinker = pytest.importorskip("tinker")
VALID = "<table><tr><td>fixture</td></tr></table>"


def reject_fixture(directory, index=0, record=None, stop="stop", html="invalid"):
    record = record or {
        "id": "fixture",
        "prompt_sha256": "prompt",
        "image_sha256": "image",
    }
    rollout = {"record": record, "tokens": [3], "html": html, "stop_reason": stop}
    path = directory / f"{index:04d}.rollout.json"
    mod.atomic_json(path, rollout)
    try:
        mod.validate_rollout(rollout)
    except ValueError as exc:
        reason = str(exc)
    else:
        reason = "invented rejection"
    rejection = {
        "record": record,
        "rollout_sha256": mod.digest(path),
        "reason": reason,
        "output_tokens": 1,
    }
    mod.atomic_json(directory / f"{index:04d}.rejected.json", rejection)
    return record


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("html,stop", [("invalid", "stop"), (VALID, "length")])
def test_invalid_rollout_preserved_without_topk_or_replacement(
    tmp_path, skip, html, stop
):
    prompt = tinker.ModelInput.from_ints([1, 2])
    record = {
        "id": "fixture",
        "prompt_sha256": mod.json_hash(prompt.model_dump(mode="json")),
    }
    args = SimpleNamespace(
        cache_dir=tmp_path,
        max_new_tokens=10,
        max_sequence_tokens=100,
        seed=42,
        skip_rejected=skip,
    )
    budget = mod.CollectionBudget(tmp_path / "usage.json", maximum=1)
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        assert not kwargs.get("topk_prompt_logprobs")
        return SimpleNamespace(
            result=lambda **kw: SimpleNamespace(
                sequences=[SimpleNamespace(tokens=[3, 4], stop_reason=stop)]
            )
        )

    tokenizer = SimpleNamespace(decode=lambda *a, **kw: html)
    renderer = SimpleNamespace(get_stop_sequences=lambda: [99])

    def invoke():
        return mod.collect_one(
            0,
            {},
            record,
            prompt,
            tokenizer,
            renderer,
            SimpleNamespace(sample=sample),
            args,
            budget,
        )

    if skip:
        assert invoke() is False
    else:
        with pytest.raises(ValueError):
            invoke()
    assert len(calls) == 1
    rollout_path = tmp_path / "0000.rollout.json"
    original = rollout_path.read_bytes()
    saved = json.loads(original)
    assert saved["tokens"] == [3, 4] and saved["html"] == html
    rejection = mod.rejected_entry(tmp_path, 0, record)
    assert rejection["rollout_sha256"] == mod.digest(rollout_path)
    assert budget.state["calls"] == 1 and budget.state["pending"] is None
    assert not (tmp_path / "0000.npz").exists()
    assert not (tmp_path / "0000.json").exists()
    # Explicit resume reuses the original rejected output; no fresh rollout.
    args.skip_rejected = True
    assert invoke() is False
    assert len(calls) == 1 and rollout_path.read_bytes() == original


def test_missing_skip_flag_defaults_to_fail_closed(tmp_path):
    prompt = tinker.ModelInput.from_ints([1])
    record = reject_fixture(tmp_path)
    args = SimpleNamespace(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        mod.collect_one(
            0,
            {},
            record,
            prompt,
            None,
            None,
            SimpleNamespace(sample=lambda **kw: pytest.fail("No request permitted")),
            args,
            None,
        )


@pytest.mark.parametrize(
    "change", ["hash", "record", "rollout_record", "reason", "valid", "ambiguous"]
)
def test_tampered_valid_or_ambiguous_rejection_is_rejected(tmp_path, change):
    record = reject_fixture(tmp_path)
    path = tmp_path / "0000.rejected.json"
    rejection = json.loads(path.read_text())
    rollout_path = tmp_path / "0000.rollout.json"
    rollout = json.loads(rollout_path.read_text())
    if change == "hash":
        rollout_path.write_text(rollout_path.read_text() + " ")
    elif change == "record":
        rejection["record"] = {**record, "id": "changed"}
    elif change == "rollout_record":
        rollout["record"] = {**record, "id": "changed"}
        mod.atomic_json(rollout_path, rollout)
        rejection["rollout_sha256"] = mod.digest(rollout_path)
    elif change == "reason":
        rejection["reason"] = "different reason"
    elif change == "valid":
        rollout["html"] = VALID
        mod.atomic_json(rollout_path, rollout)
        rejection["rollout_sha256"] = mod.digest(rollout_path)
    else:
        mod.atomic_json(tmp_path / "0000.json", {"accepted": True})
    mod.atomic_json(path, rejection)
    with pytest.raises(ValueError):
        mod.rejected_entry(tmp_path, 0, record)


def loader_fixture(tmp_path, monkeypatch):
    prompt = tinker.ModelInput.from_ints([1, 2])
    tokenizer_spec = {"model": {"vocab": {"fixture": 0}}}

    class Tokenizer:
        backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(tokenizer_spec))

        def __len__(self):
            return 20

    renderer = SimpleNamespace(build_generation_prompt=lambda _: prompt)
    monkeypatch.setattr(kd, "image_message", lambda *a: object())
    rows, records = [], []
    for i in range(3):
        image = tmp_path / f"image-{i}.fixture"
        image.write_bytes(f"software fixture {i}".encode())
        rows.append({"id": f"fixture-{i}", "image": image.name})
        records.append(
            {
                "id": rows[-1]["id"],
                "image_sha256": mod.digest(image),
                "prompt_sha256": mod.json_hash(prompt.model_dump(mode="json")),
            }
        )
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dev.write_text("{}\n")
    args = SimpleNamespace(
        cache_dir=tmp_path,
        train_manifest=train,
        dev_manifest=dev,
        data_root=tmp_path,
        train_examples=2,
        max_pixels=100,
        max_sequence_tokens=100,
        skip_rejected=True,
    )
    manifest = {
        "teacher_model": kd.TEACHER_MODEL,
        "teacher_processor_revision": kd.TEACHER_REVISION,
        "student_model": kd.MODEL,
        "student_processor_revision": kd.PROCESSOR_REVISION,
        "cookbook_revision": kd.COOKBOOK_REVISION,
        "top_k": mod.TOP_K,
        "loss_temperature": 1,
        "rollout_temperature": 0,
        "train_manifest_sha256": mod.digest(train),
        "dev_manifest_sha256": mod.digest(dev),
        "tokenizer_sha256": mod.json_hash(tokenizer_spec),
        "max_pixels": 100,
        "max_sequence_tokens": 100,
        "prompt_sha256": hashlib.sha256(kd.PROMPT.encode()).hexdigest(),
        "records": records,
    }
    mod.atomic_json(tmp_path / "manifest.json", manifest)
    mod.atomic_json(
        tmp_path / "0000.rollout.json",
        {"record": records[0], "tokens": [3], "html": VALID, "stop_reason": "stop"},
    )
    np.savez_compressed(
        tmp_path / "0000.npz",
        token_ids=np.arange(mod.TOP_K)[None, :],
        logprobs=np.full((1, mod.TOP_K), np.log(0.099)),
    )
    mod.atomic_json(
        tmp_path / "0000.json",
        {
            "record": records[0],
            "arrays_sha256": mod.digest(tmp_path / "0000.npz"),
            "rollout_sha256": mod.digest(tmp_path / "0000.rollout.json"),
        },
    )
    return args, Tokenizer(), renderer, records


def test_loader_excludes_verified_candidate_without_replacement_and_counts_it(
    tmp_path, monkeypatch
):
    args, tokenizer, renderer, records = loader_fixture(tmp_path, monkeypatch)
    reject_fixture(tmp_path, index=1, record=records[1])
    datums, _, metadata = kd.load_train(args, tokenizer, renderer)
    assert len(datums) == 1
    assert metadata["candidate_examples"] == 2
    assert metadata["rejected_examples"] == 1
    assert metadata["target_filter_policy"] == "exclude_invalid_without_replacement"
    # Third candidate has no files: successful loading proves no replacement.
    assert not (tmp_path / "0002.json").exists()
    args.skip_rejected = False
    with pytest.raises(FileNotFoundError):
        kd.load_train(args, tokenizer, renderer)


def test_loader_never_treats_missing_entry_as_rejected(tmp_path, monkeypatch):
    args, tokenizer, renderer, _ = loader_fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        kd.load_train(args, tokenizer, renderer)


def test_loader_refuses_ambiguous_acceptance_and_rejection(tmp_path, monkeypatch):
    args, tokenizer, renderer, records = loader_fixture(tmp_path, monkeypatch)
    reject_fixture(tmp_path, index=1, record=records[1])
    mod.atomic_json(tmp_path / "0001.json", {"accepted": True})
    with pytest.raises(ValueError, match="ambiguous"):
        kd.load_train(args, tokenizer, renderer)


def test_loader_refuses_an_all_rejected_candidate_subset(tmp_path, monkeypatch):
    args, tokenizer, renderer, records = loader_fixture(tmp_path, monkeypatch)
    (tmp_path / "0000.json").unlink()
    (tmp_path / "0000.npz").unlink()
    reject_fixture(tmp_path, index=0, record=records[0])
    reject_fixture(tmp_path, index=1, record=records[1])
    with pytest.raises(ValueError, match="No valid"):
        kd.load_train(args, tokenizer, renderer)
