import json

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROFILE_PATH,
    ManifestError,
    build_manifest,
    content_id,
    dataset_profile,
    read_manifest,
    select_rows,
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
        train_rows,
        test_rows,
        "revision",
        seed=7,
        sft_train_count=2,
        sft_validation_count=1,
        rl_train_count=3,
        rl_monitor_count=1,
        calibration_test_count=1,
    )
    second = build_manifest(
        train_rows,
        test_rows,
        "revision",
        seed=7,
        sft_train_count=2,
        sft_validation_count=1,
        rl_train_count=3,
        rl_monitor_count=1,
        calibration_test_count=1,
    )

    assert first == second
    all_ids = (
        first.sft_train_ids
        + first.sft_validation_ids
        + first.rl_train_ids
        + first.rl_monitor_ids
        + first.calibration_test_ids
        + first.formal_test_ids
    )
    assert len(set(all_ids)) == 11
    assert first.manifest_hash


def test_manifest_rejects_insufficient_rows():
    with pytest.raises(ManifestError, match="need 6 train rows"):
        build_manifest(
            _rows("train", 4),
            _rows("test", 3),
            "revision",
            sft_train_count=2,
            sft_validation_count=1,
            rl_train_count=2,
            rl_monitor_count=1,
            calibration_test_count=1,
        )


def test_manifest_writes_ids_and_metadata_without_raw_examples(tmp_path):
    manifest = build_manifest(
        _rows("train", 6),
        _rows("test", 3),
        "revision",
        sft_train_count=2,
        sft_validation_count=1,
        rl_train_count=2,
        rl_monitor_count=1,
        calibration_test_count=1,
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
    manifest = read_manifest(DEFAULT_MANIFEST_PATH)
    all_ids = (
        manifest.sft_train_ids
        + manifest.sft_validation_ids
        + manifest.rl_train_ids
        + manifest.rl_monitor_ids
        + manifest.calibration_test_ids
        + manifest.formal_test_ids
    )

    assert len(manifest.sft_train_ids) == 5000
    assert len(manifest.sft_validation_ids) == 500
    assert len(manifest.rl_train_ids) == 1800
    assert len(manifest.rl_monitor_ids) == 173
    assert len(manifest.calibration_test_ids) == 32
    assert len(manifest.formal_test_ids) == 1287
    assert manifest.calibration_test_ids[0] == (
        "gsm8k-22ddac99e9cde0cfcbc6e426ee9b9e1e5faeb4112501fb365b9d7e277ae8ebab"
    )
    assert len(set(all_ids)) == len(all_ids)


def test_select_rows_restores_manifest_order_and_rejects_missing_ids():
    rows = _rows("test", 3)
    selected = select_rows(rows, (content_id(rows[2]), content_id(rows[0])))

    assert selected == (rows[2], rows[0])
    with pytest.raises(ManifestError, match="missing manifest ID"):
        select_rows(rows, ("gsm8k-missing",))


def test_committed_profile_contains_only_aggregate_dataset_facts():
    profile = json.loads(DEFAULT_PROFILE_PATH.read_text())

    assert profile["train"]["examples"] == 7473
    assert profile["test"]["examples"] == 1319
    assert profile["train"]["answer_marker_fraction"] == 1.0
