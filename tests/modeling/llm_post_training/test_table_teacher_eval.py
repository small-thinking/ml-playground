"""Teacher evaluation identity, durable spend, and mocked complete-run contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import teacher_eval as mod


def test_reserved_failure_blocks_automatic_paid_retry(tmp_path):
    budget = mod.ConcurrentBudget(tmp_path / "usage.json", 1)

    def fail():
        raise TimeoutError("uncertain provider response")

    with pytest.raises(TimeoutError):
        mod.paid_request(
            budget.for_example(),
            "call",
            0.1,
            tmp_path / "row.json",
            fail,
            lambda r: 0.01,
        )
    assert budget.state["pending"] == {"call": {"estimated_usd_bound": 0.1}}
    with pytest.raises(ValueError, match="Uncertain paid request"):
        mod.ConcurrentBudget(tmp_path / "usage.json", 1)


def test_budget_blocks_before_call_and_completion_cannot_replay(tmp_path):
    budget = mod.ConcurrentBudget(tmp_path / "usage.json", 0.1)
    calls = []
    with pytest.raises(ValueError, match="budget reached"):
        mod.paid_request(
            budget.for_example(),
            "large",
            0.1,
            tmp_path / "large.json",
            lambda: calls.append(1),
            lambda r: 0,
        )
    assert calls == []
    path = tmp_path / "ok.json"
    mod.paid_request(
        budget.for_example(), "small", 0.05, path, lambda: {"a": 1}, lambda r: 0.02
    )
    assert json.loads(path.read_text()) == {"a": 1}
    assert path.stat().st_mode & 0o777 == 0o600
    assert budget.state["estimated_compute_usd"] == 0.02
    assert budget.state["pending"] is None
    with pytest.raises(ValueError, match="do not replay"):
        mod.paid_request(
            budget.for_example(),
            "small",
            0.05,
            path,
            lambda: pytest.fail("Paid replay"),
            lambda r: 0.02,
        )


def test_cost_uses_selected_teacher_and_includes_ignored_nll_token():
    ex = (
        {},
        SimpleNamespace(length=100),
        SimpleNamespace(model_input=SimpleNamespace(length=199)),
    )
    spec = mod.teacher_spec(mod.LARGE_TEACHER_MODEL)
    result = mod.costs([ex] * 100, spec, 8192)
    assert result["estimated_compute_usd_bound"] == pytest.approx(
        (30000 * 3 + 819300 * 7.5) / 1e6
    )


@pytest.mark.parametrize("problem", ["count", "split", "hash", "escape", "duplicate"])
def test_manifest_rejects_incomplete_or_changed_data(tmp_path, problem):
    image, label = tmp_path / "img", tmp_path / "label"
    image.write_bytes(b"image")
    label.write_text("table")
    row = {
        "id": "one",
        "split": "dev",
        "image": "img",
        "label": "label",
        "image_sha256": mod.digest(image),
        "label_sha256": mod.digest(label),
    }
    expected = 1
    rows = [row]
    if problem == "count":
        expected = 100
    elif problem == "split":
        row["split"] = "test"
    elif problem == "hash":
        row["image_sha256"] = "stale"
    elif problem == "escape":
        row["image"] = "../outside"
    else:
        rows *= 2
        expected = 2
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        mod.validate_manifest(manifest, tmp_path, expected)


def test_mock_complete_run_preserves_raw_local_outputs_and_accounting(
    tmp_path, monkeypatch
):
    tinker = pytest.importorskip("tinker")
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text("local manifest")
    records = [{"id": "a"}, {"id": "b"}]
    examples = [
        (
            row,
            SimpleNamespace(length=10),
            SimpleNamespace(model_input=SimpleNamespace(length=19)),
        )
        for row in records
    ]
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"))
    renderer = SimpleNamespace(get_stop_sequences=lambda: [1])
    monkeypatch.setattr(mod, "validate_manifest", lambda *a: records)
    monkeypatch.setattr(mod, "OfficialScorer", lambda *a: None)
    monkeypatch.setattr(mod, "load_renderer", lambda *a: (tokenizer, renderer))
    monkeypatch.setattr(mod, "prepare_examples", lambda *a: examples)
    monkeypatch.setattr(
        mod,
        "evaluate",
        lambda *a: ([{"id": r["id"]} for r in records], {"eval/cell_f1": 0.5}),
    )
    monkeypatch.setattr(
        mod,
        "likelihood_one",
        lambda sampler, ex: {
            "id": ex[0]["id"],
            "token_logprobs": [-1.0, -2.0],
            "target_tokens": [1, 2],
        },
    )
    monkeypatch.setattr(
        mod,
        "sample_one",
        lambda sampler, ex, *a: {
            "id": ex[0]["id"],
            "tokens": [1, 2],
            "token_logprobs": [-1.0, -2.0],
            "html": "private",
            "input_tokens": 10,
            "output_tokens": 2,
        },
    )
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return object()

    monkeypatch.setattr(
        tinker, "ServiceClient", lambda: SimpleNamespace(create_sampling_client=create)
    )
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        model=mod.LARGE_TEACHER_MODEL,
        manifest=manifest,
        data_root=tmp_path,
        expected_examples=2,
        official_repo=tmp_path,
        tinker_cookbook_dir=tmp_path,
        max_pixels=1048576,
        max_new_tokens=8192,
        seed=20260913,
        env_file=None,
        execute=True,
        max_estimated_usd=1,
        concurrency=2,
    )
    mod.run(args)
    report = json.loads((args.output_dir / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["pending_request"] is None
    assert report["metrics"]["assistant_nll"] == 1.5
    assert report["metrics"]["supervised_tokens"] == 4
    assert report["estimated_compute_usd"] == pytest.approx(
        2 * ((20 * 3 + 7.5) + (10 * 3 + 2 * 7.5)) / 1e6
    )
    assert len(report["output_hashes"]) == 5
    assert all(
        mod.digest(args.output_dir / name) == value
        for name, value in report["output_hashes"].items()
    )
    assert seen[0]["base_model"] == args.model
    assert seen[0]["retry_config"].enable_retry_logic is False
    assert (args.output_dir / "0000_prediction.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="fresh output"):
        mod.run(args)
