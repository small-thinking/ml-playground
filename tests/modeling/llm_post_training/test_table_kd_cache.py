"""Offline cache, request-accounting and publication regressions for KD."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import kd_collect as mod
from modeling.llm_post_training.vlm_table_extraction_lab import tinker_inference
from modeling.llm_post_training.vlm_table_extraction_lab.kd_targets import soft_targets
from modeling.llm_post_training.vlm_table_extraction_lab.log_checkpoint_eval import (
    make_payload,
)
from modeling.llm_post_training.vlm_table_extraction_lab.training_logging import (
    training_config,
)

tinker = pytest.importorskip("tinker")
HTML = "<table><tr><td>software fixture</td></tr></table>"


def fixture_prompt():
    return tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[12]),
            tinker.types.ImageChunk(data=b"fixture", format="png", expected_tokens=2),
            tinker.EncodedTextChunk(tokens=[13]),
        ]
    )


def response(prompt_length, completion_length=2):
    ids = np.zeros((prompt_length + completion_length, mod.TOP_K), dtype=np.int32)
    lps = np.full(ids.shape, np.nan, dtype=np.float32)
    ids[prompt_length:] = np.arange(mod.TOP_K)
    lps[prompt_length:] = np.log(0.09)
    return SimpleNamespace(
        topk_prompt_logprobs_np=SimpleNamespace(token_ids=ids, logprobs=lps)
    )


def test_completion_alignment_excludes_entire_masked_image_prefix():
    prompt = fixture_prompt()
    native = response(prompt.length)
    ids, lps = mod.completion_topk(native, prompt.length, 2)
    assert ids.shape == lps.shape == (2, mod.TOP_K)
    assert np.isfinite(lps).all()
    datum = soft_targets(prompt, [4, 5], ids, lps, 20)
    weights = datum.loss_fn_inputs["weights"].to_numpy()
    np.testing.assert_array_equal(weights[: prompt.length - 1], 0)
    np.testing.assert_allclose(weights[prompt.length - 1 :].sum(axis=1), 1)
    assert datum.model_input.chunks[1] is prompt.chunks[1]
    assert datum.model_input.chunks[-1].tokens == [4]


@pytest.mark.parametrize("kind", ["missing", "short", "wrong_k", "different_shapes"])
def test_completion_topk_shape_or_missing_response_fails(kind):
    native = response(4)
    if kind == "missing":
        native.topk_prompt_logprobs_np = None
    elif kind == "short":
        native.topk_prompt_logprobs_np.token_ids = np.zeros((5, mod.TOP_K))
        native.topk_prompt_logprobs_np.logprobs = np.zeros((5, mod.TOP_K))
    elif kind == "wrong_k":
        native.topk_prompt_logprobs_np.token_ids = np.zeros((6, mod.TOP_K - 1))
        native.topk_prompt_logprobs_np.logprobs = np.zeros((6, mod.TOP_K - 1))
    else:
        native.topk_prompt_logprobs_np.logprobs = np.zeros((5, mod.TOP_K))
    with pytest.raises(ValueError):
        mod.completion_topk(native, 4, 2)


def test_reservation_persists_before_request_and_blocks_uncertain_resume(tmp_path):
    path = tmp_path / "usage.json"
    budget = mod.CollectionBudget(path, maximum=0.111)
    budget.reserve("fixture:rollout", 0.1)
    disk = json.loads(path.read_text())
    assert disk["pending"] == {"key": "fixture:rollout", "estimated_usd_bound": 0.1}
    with pytest.raises(ValueError, match="Uncertain"):
        mod.CollectionBudget(path, maximum=100)
    budget.settle(0.04)
    resumed = mod.CollectionBudget(path, maximum=0.111)
    assert resumed.state["estimated_compute_usd"] == 0.04
    assert resumed.state["calls"] == 1
    with pytest.raises(ValueError, match="budget"):
        resumed.reserve("fixture:topk", 0.07)
    assert json.loads(path.read_text())["pending"] is None


def test_settlement_above_reserved_bound_preserves_uncertain_request(tmp_path):
    path = tmp_path / "usage.json"
    budget = mod.CollectionBudget(path, maximum=1)
    budget.reserve("fixture:topk", 0.1)
    with pytest.raises(ValueError, match="exceeds"):
        budget.settle(0.2)
    with pytest.raises(ValueError, match="Uncertain"):
        mod.CollectionBudget(path, maximum=1)


def cache_fixture(directory):
    prompt = fixture_prompt()
    record = {
        "id": "software-fixture",
        "image_sha256": "image-hash",
        "prompt_sha256": mod.json_hash(prompt.model_dump(mode="json")),
    }
    rollout = {"record": record, "tokens": [4, 5], "html": HTML, "stop_reason": "stop"}
    mod.atomic_json(directory / "0000.rollout.json", rollout)
    arrays = directory / "0000.npz"
    np.savez_compressed(
        arrays,
        token_ids=np.tile(np.arange(mod.TOP_K), (2, 1)),
        logprobs=np.full((2, mod.TOP_K), np.log(0.09)),
    )
    metadata = {
        "record": record,
        "arrays_sha256": mod.digest(arrays),
        "rollout_sha256": mod.digest(directory / "0000.rollout.json"),
    }
    mod.atomic_json(directory / "0000.json", metadata)
    return prompt, record


def test_cache_readback_reconstructs_targets_without_mutating_artifacts(tmp_path):
    prompt, record = cache_fixture(tmp_path)
    before = {p.name: mod.digest(p) for p in tmp_path.iterdir()}
    datum, lp = mod.cache_entry(tmp_path, 0, record, prompt, 20)
    assert lp.shape == (2, mod.TOP_K)
    assert datum.model_input.length == prompt.length + 1
    assert before == {p.name: mod.digest(p) for p in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "changed", ["record", "prompt", "arrays", "rollout", "missing_arrays"]
)
def test_cache_rejects_changed_identity_or_content(tmp_path, changed):
    prompt, record = cache_fixture(tmp_path)
    if changed == "record":
        record = {**record, "image_sha256": "another-image"}
    elif changed == "prompt":
        prompt = prompt.append_int(10)
    elif changed == "arrays":
        with (tmp_path / "0000.npz").open("ab") as stream:
            stream.write(b"changed")
    elif changed == "rollout":
        with (tmp_path / "0000.rollout.json").open("a") as stream:
            stream.write(" ")
    else:
        (tmp_path / "0000.npz").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        mod.cache_entry(tmp_path, 0, record, prompt, 20)


@pytest.mark.parametrize("stop,html", [("length", HTML), ("stop", "not a table")])
def test_cache_rejects_semantically_invalid_rollout_even_with_matching_hash(
    tmp_path, stop, html
):
    prompt, record = cache_fixture(tmp_path)
    path = tmp_path / "0000.rollout.json"
    rollout = json.loads(path.read_text())
    rollout.update(stop_reason=stop, html=html)
    mod.atomic_json(path, rollout)
    metadata = json.loads((tmp_path / "0000.json").read_text())
    metadata["rollout_sha256"] = mod.digest(path)
    mod.atomic_json(tmp_path / "0000.json", metadata)
    with pytest.raises(ValueError):
        mod.cache_entry(tmp_path, 0, record, prompt, 20)


def test_topk_persistence_failure_does_not_repeat_a_paid_request(tmp_path, monkeypatch):
    prompt, record = cache_fixture(tmp_path)
    (tmp_path / "0000.json").unlink()
    (tmp_path / "0000.npz").unlink()
    args = SimpleNamespace(
        cache_dir=tmp_path, max_new_tokens=100, max_sequence_tokens=100, seed=42
    )

    class Tokenizer:
        def __len__(self):
            return 20

    budget = mod.CollectionBudget(tmp_path / "usage.json", maximum=1)
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        assert json.loads((tmp_path / "usage.json").read_text())["pending"] is not None
        return SimpleNamespace(result=lambda **kw: response(prompt.length))

    original_save = mod.np.savez_compressed

    def fail_save(*a, **kw):
        raise OSError("simulated disk interruption")

    monkeypatch.setattr(mod.np, "savez_compressed", fail_save)
    with pytest.raises(OSError, match="simulated"):
        mod.collect_one(
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
    assert len(calls) == 1
    monkeypatch.setattr(mod.np, "savez_compressed", original_save)
    # Safe implementations either retain pending uncertainty or replay a saved
    # raw response locally. Neither may submit the same paid Top-K request again.
    try:
        resumed = mod.CollectionBudget(tmp_path / "usage.json", maximum=1)
    except ValueError as error:
        assert "Uncertain" in str(error)
        return

    def forbidden_sample(**kwargs):
        pytest.fail("Paid Top-K request repeated after response persistence failed")

    mod.collect_one(
        0,
        {},
        record,
        prompt,
        Tokenizer(),
        None,
        SimpleNamespace(sample=forbidden_sample),
        args,
        resumed,
    )


def fake_tokenizer(spec, special=None):
    return SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(spec)),
        special_tokens_map=special or {"eos_token": "<end>"},
    )


@pytest.mark.parametrize("change", ["ids", "merges", "normalizer", "special"])
def test_tokenizer_compatibility_checks_more_than_vocabulary_size(change):
    spec = {
        "model": {"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]},
        "normalizer": {"type": "NFC"},
    }
    altered = copy.deepcopy(spec)
    special = {"eos_token": "<end>"}
    if change == "ids":
        altered["model"]["vocab"] = {"a": 1, "b": 0}
    elif change == "merges":
        altered["model"]["merges"] = [["b", "a"]]
    elif change == "normalizer":
        altered["normalizer"]["type"] = "NFKC"
    else:
        special = {"eos_token": "<different>"}
    with pytest.raises(ValueError, match="tokenization"):
        mod.tokenizer_identity(fake_tokenizer(spec), fake_tokenizer(altered, special))
    assert mod.tokenizer_identity(
        fake_tokenizer(spec), fake_tokenizer(spec)
    ) == mod.json_hash(spec)


def test_wrong_model_revision_rejected_before_cookbook_import(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tinker_inference,
        "verify_cookbook",
        lambda *a: tmp_path,
    )
    with pytest.raises(ValueError, match="pinned"):
        tinker_inference.load_renderer(
            tinker_inference.TEACHER_MODEL,
            tinker_inference.PROCESSOR_REVISION,
            tmp_path,
        )


def public_training_fixture():
    report = {
        key: "public"
        for key in (
            "model",
            "initialization",
            "hosted_weight_revision",
            "processor_revision",
            "cookbook_revision",
            "official_revision",
            "metrics_version",
            "tinker_version",
            "train_manifest_sha256",
            "dev_manifest_sha256",
            "train_examples",
            "dev_examples",
            "optimizer_steps",
            "lora",
            "optimizer",
            "loss_reduction",
            "learning_rate_schedule",
            "warmup_steps",
            "rates_usd_per_million",
            "implementation_sha256",
        )
    }
    report.update(
        algorithm="off_policy_topk_kd",
        teacher_model=mod.TEACHER_MODEL,
        teacher_processor_revision=mod.TEACHER_REVISION,
        cache_sha256="cache-hash",
        top_k=10,
        loss_temperature=1,
        rollout_temperature=0,
        tokenizer_sha256="tokens-hash",
        retained_mass={"retained_mass_mean": 0.98},
        teacher_hosted_weight_revision=None,
        prompt="fixed public instruction",
        cost={"training_tokens": 100, "estimated_usd_bound": 0.01},
        config={
            k: 1
            for k in (
                "epochs",
                "batch_size",
                "seed",
                "warmup_ratio",
                "eval_every",
                "generate_every",
                "generate_train",
                "generate_dev",
                "train_nll_endpoints_only",
                "max_sequence_tokens",
                "max_new_tokens",
                "max_pixels",
                "dataset_label",
                "inference_concurrency",
            )
        },
    )
    report["config"].update(
        cache_dir="/PRIVATE_MARKER/cache", env_file="/PRIVATE_MARKER/env"
    )
    report["raw_topk"] = "PRIVATE_MARKER"
    report["sampler_paths"] = {"after": "PRIVATE_MARKER"}
    return report


def test_kd_configuration_allowlist_and_publisher_role_preserve_comparison_group():
    public = training_config(public_training_fixture())
    assert "PRIVATE_MARKER" not in json.dumps(public)
    assert public["algorithm"] == "off_policy_topk_kd"
    assert public["generate_dev"] == 1
    assert public["top_k"] == 10 and public["teacher_hosted_weight_revision"] is None
    report = {
        "config": {
            "split": "test",
            "max_new_tokens": 8192,
            "max_pixels": 1048576,
            "source_run": "/PRIVATE_MARKER/run.json",
        },
        "manifest_sha256": "manifest",
        "examples": 100,
        "processor_revision": "processor",
        "seed": 42,
        "source_run_sha256": "source",
        "after": {"assistant_nll": 0.1, "assistant_perplexity": 1.105},
        "before": {"assistant_nll": 0.1, "assistant_perplexity": 1.105},
    }
    metrics = {
        "eval/numeric_f1_count": 98,
        "eval/official_rd_similarity_raw_count": 100,
        "eval/cell_f1": 0.4,
        "eval/official_rd_similarity": 0.7,
    }
    kd = make_payload(
        report,
        metrics,
        "after",
        "predictions",
        "project",
        "base-id",
        training=public,
        training_run_id="training-id",
        trained_role="kd",
    )
    sft = make_payload(report, metrics, "after", "predictions", "project", "base-id")
    base = make_payload(
        report, metrics, "before", "predictions", "project", None, trained_role="kd"
    )
    assert kd["group"] == sft["group"] == base["group"]
    assert kd["run_id"] != sft["run_id"]
    assert kd["config"]["model_role"] == "kd" and "-kd-" in kd["name"]
    assert "kd" in kd["tags"] and base["config"]["model_role"] == "base"
    assert kd["config"]["training"] == public
    assert "PRIVATE_MARKER" not in json.dumps(kd)
