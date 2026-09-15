import json

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.log_checkpoint_eval import (
    make_payload,
)


@pytest.fixture
def inputs():
    likelihood = {"assistant_nll": 0.1, "assistant_perplexity": 1.105}
    report = {
        "config": {
            "split": "test",
            "max_new_tokens": 8192,
            "max_pixels": 1048576,
            "source_run": "/PRIVATE_MARKER/run.json",
        },
        "manifest_sha256": "manifest-hash",
        "examples": 100,
        "processor_revision": "processor-hash",
        "seed": 42,
        "source_run_sha256": "source-hash",
        "sampler_paths": {"before": "PRIVATE_MARKER"},
        "before": likelihood,
        "after": likelihood,
    }
    metrics = {
        "eval/numeric_f1_count": 98,
        "eval/official_rd_similarity_raw_count": 100,
        "eval/cell_f1": 0.4,
        "eval/official_rd_similarity": 0.7,
        "eval/private": "PRIVATE_MARKER",
    }
    return report, metrics


def test_payload_allowlist_excludes_private_source_fields(inputs):
    payload = make_payload(*inputs, "before", "prediction-hash", "project", None)
    assert "PRIVATE_MARKER" not in json.dumps(payload)
    assert payload["metrics"]["quality/cell_f1"] == 0.4
    assert payload["metrics"]["likelihood/nll"] == 0.1
    assert "runtime/wall_seconds" not in payload["metrics"]
    assert payload["config"]["model_role"] == "base"


def test_identity_stable_but_distinct_predictions_and_protocols_split_runs(inputs):
    first = make_payload(*inputs, "before", "one", "project", None)
    assert first == make_payload(*inputs, "before", "one", "project", None)
    second = make_payload(*inputs, "after", "two", "project", first["run_id"])
    assert first["group"] == second["group"]
    assert first["run_id"] != second["run_id"]
    assert second["config"]["baseline_run_id"] == first["run_id"]
    inputs[0]["config"]["max_pixels"] = 123
    assert (
        make_payload(*inputs, "before", "one", "project", None)["group"]
        != first["group"]
    )


def test_nonfinite_metrics_rejected(inputs):
    inputs[1]["eval/cell_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        make_payload(*inputs, "before", "one", "project", None)


def test_public_display_label_preserves_identity_and_rejects_paths(inputs):
    plain = make_payload(*inputs, "before", "one", "project", None)
    named = make_payload(
        *inputs, "before", "one", "project", None, run_label="control160"
    )
    assert "-control160-" in named["name"]
    assert {k: v for k, v in named.items() if k != "name"} == {
        k: v for k, v in plain.items() if k != "name"
    }
    with pytest.raises(ValueError, match="without paths"):
        make_payload(
            *inputs, "before", "one", "project", None, run_label="/private/data"
        )


def test_opd_keeps_shared_comparison_group_and_distinct_identity(inputs):
    baseline = make_payload(*inputs, "before", "one", "project", None)
    trained = [
        make_payload(
            *inputs, "after", "two", "project", baseline["run_id"], trained_role=role
        )
        for role in ("sft", "kd", "opd")
    ]
    assert all(payload["group"] == baseline["group"] for payload in trained)
    assert len({payload["run_id"] for payload in [baseline, *trained]}) == 4
    opd = trained[-1]
    assert opd["config"]["model_role"] == "opd"
    assert opd["config"]["baseline_run_id"] == baseline["run_id"]
    assert "opd" in opd["tags"] and "-opd-" in opd["name"]
    assert "PRIVATE_MARKER" not in json.dumps(opd)
