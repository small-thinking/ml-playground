"""Saved-checkpoint evaluation accounting and likelihood alignment, without APIs."""

from types import SimpleNamespace

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import checkpoint_eval as mod


def test_before_after_budget_includes_full_nll_and_ignored_sample_token():
    ex = (
        {},
        SimpleNamespace(length=100),
        SimpleNamespace(model_input=SimpleNamespace(length=199)),
    )
    cost = mod.estimate_cost([ex] * 100, 8192)
    assert cost["nll_input_tokens"] == 40000
    assert cost["generation_input_tokens"] == 20000
    assert cost["generation_output_token_bound"] == 1638400
    assert cost["estimated_compute_usd_bound"] == pytest.approx(
        (60000 * 0.33 + 1638600 * 1.005) / 1e6
    )


def test_sampler_alignment_masks_prompt_but_includes_first_answer_and_eos():
    tinker = pytest.importorskip("tinker")
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 20, 30]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[20, 30, 99], dtype="int64"),
            "weights": tinker.TensorData(data=[0, 1, 1], dtype="float32"),
        },
    )
    seen = []

    def logprobs(full):
        seen.extend(full.chunks[-1].tokens)
        return SimpleNamespace(result=lambda **kw: [None, -100.0, -2.0, -3.0])

    row = mod.likelihood_one(
        SimpleNamespace(compute_logprobs=logprobs), ({"id": "example"}, None, datum)
    )
    assert row == {
        "id": "example",
        "token_logprobs": [-2.0, -3.0],
        "target_tokens": [30, 99],
    }
    assert seen == [99]
    summary = mod.likelihood_summary([row])
    assert summary["assistant_nll"] == 2.5
    assert summary["supervised_tokens"] == 2
    assert summary["assistant_perplexity"] == pytest.approx(12.182493960703473)


def test_collect_keeps_all_rows_and_unique_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    results = mod.collect(range(100), lambda i: {"id": str(i)}, path, 4)
    assert len(results) == 100
    assert len({r["id"] for r in results}) == 100
    assert len(path.read_text().splitlines()) == 100


def test_failure_cancels_queued_requests(tmp_path, monkeypatch):
    from concurrent.futures import Future

    futures = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            future = Future()
            if not futures:
                future.set_exception(RuntimeError("request failed"))
            futures.append(future)
            return future

    monkeypatch.setattr(mod, "ThreadPoolExecutor", Pool)
    with pytest.raises(RuntimeError, match="request failed"):
        mod.collect(range(100), lambda i: {"id": str(i)}, tmp_path / "partial.jsonl", 4)
    assert all(f.cancelled() for f in futures[1:])


def test_incomplete_dev_is_rejected_before_model_loading(tmp_path, monkeypatch):
    import json
    import sys

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "status": "completed",
                "model": mod.MODEL,
                "processor_revision": mod.PROCESSOR_REVISION,
                "cookbook_revision": mod.COOKBOOK_REVISION,
            }
        )
    )
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(json.dumps({"id": "only-one", "split": "dev"}) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "checkpoint_eval",
            "--source-run",
            str(source),
            "--manifest",
            str(manifest),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--tinker-cookbook-dir",
            str(tmp_path),
            "--official-repo",
            str(tmp_path),
            "--execute",
        ],
    )
    monkeypatch.setattr(
        mod, "load_renderer", lambda *a: pytest.fail("Do not load model")
    )
    with pytest.raises(ValueError, match="entire"):
        mod.main()
