import json

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROFILE_PATH,
    ManifestError,
    build_manifest,
    content_id,
    dataset_profile,
    write_manifest,
)


def _rows(prefix, count):
    return [
        {"question": f"{prefix} question {index}", "answer": f"reasoning #### {index}"}
        for index in range(count)
    ]


def test_content_id_is_stable_and_uses_question_and_answer():
    row = {"question": "How many?", "answer": "Work #### 2"}

    assert content_id(row) == content_id(dict(row))
    assert content_id(row) != content_id(
        {"question": "How many?", "answer": "Work #### 3"}
    )


def test_manifest_is_deterministic_and_disjoint():
    train_rows = _rows("train", 8)
    test_rows = _rows("test", 4)

    first = build_manifest(
        train_rows, test_rows, "revision", seed=7, sft_count=2, rl_count=3, eval_count=2
    )
    second = build_manifest(
        train_rows, test_rows, "revision", seed=7, sft_count=2, rl_count=3, eval_count=2
    )

    assert first == second
    assert len(set(first.sft_ids + first.rl_ids + first.eval_ids)) == 7
    assert first.manifest_hash


def test_manifest_rejects_insufficient_rows():
    with pytest.raises(ManifestError, match="need 5 train rows"):
        build_manifest(
            _rows("train", 4),
            _rows("test", 2),
            "revision",
            sft_count=2,
            rl_count=3,
            eval_count=2,
        )


def test_manifest_writes_ids_and_metadata_without_raw_examples(tmp_path):
    manifest = build_manifest(
        _rows("train", 5),
        _rows("test", 2),
        "revision",
        sft_count=2,
        rl_count=2,
        eval_count=1,
    )
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)

    saved = json.loads(path.read_text())
    assert saved["manifest_hash"] == manifest.manifest_hash
    assert "question" not in path.read_text()
    assert "answer" not in path.read_text()


def test_dataset_profile_reports_only_aggregate_data_facts():
    profile = dataset_profile(_rows("train", 5))

    assert profile["examples"] == 5
    assert profile["answer_marker_fraction"] == 1.0
    assert profile["question_chars_p90"] >= profile["question_chars_p50"]


def test_committed_manifest_has_the_frozen_disjoint_protocol():
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    all_ids = manifest["sft_ids"] + manifest["rl_ids"] + manifest["eval_ids"]

    assert len(manifest["sft_ids"]) == 512
    assert len(manifest["rl_ids"]) == 1500
    assert len(manifest["eval_ids"]) == 256
    assert len(set(all_ids)) == len(all_ids)


def test_committed_profile_contains_only_aggregate_dataset_facts():
    profile = json.loads(DEFAULT_PROFILE_PATH.read_text())

    assert profile["train"]["examples"] == 7473
    assert profile["test"]["examples"] == 1319
    assert profile["train"]["answer_marker_fraction"] == 1.0
