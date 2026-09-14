"""Training math and preflight checks: no hosted training calls in this suite."""

import os
from types import SimpleNamespace

import pytest
import torch

from modeling.llm_post_training.vlm_table_extraction_lab import sft
from modeling.llm_post_training.vlm_table_extraction_lab.synthetic_data import generate


def test_shift_preserves_image_and_predicts_first_answer_and_eos():
    tinker = pytest.importorskip("tinker")
    image = tinker.types.ImageChunk(data=b"fixture", format="png", expected_tokens=2)
    full = tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[10]),
            image,
            tinker.EncodedTextChunk(tokens=[11, 20, 21, 99]),
        ]
    )
    datum = sft.next_token_datum(
        full, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    )
    assert datum.model_input.chunks[1] is image
    assert datum.model_input.length == 6
    assert datum.loss_fn_inputs["target_tokens"].data == [0, 0, 11, 20, 21, 99]
    assert datum.loss_fn_inputs["weights"].data == [0, 0, 0, 1, 1, 1]
    assert datum.model_input.chunks[-1].tokens == [11, 20, 21]
    assert full.chunks[-1].tokens == [11, 20, 21, 99]


def test_global_token_mean_and_nll_masking():
    tinker = pytest.importorskip("tinker")

    def datum(mask):
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints([0] * len(mask)),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=[0] * len(mask), dtype="int64"),
                "weights": tinker.TensorData(data=mask, dtype="float32"),
            },
        )

    datums = [datum([0, 1]), datum([0, 1, 1])]
    normalized = sft.token_mean_batch(datums)
    assert sum(
        sum(d.loss_fn_inputs["weights"].data) for d in normalized
    ) == pytest.approx(1)
    assert normalized[1].loss_fn_inputs["weights"].data == pytest.approx(
        [0, 1 / 3, 1 / 3]
    )
    assert datums[1].loss_fn_inputs["weights"].data == [0, 1, 1]
    output = SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": SimpleNamespace(data=[float("nan"), -1.0])},
            {"logprobs": SimpleNamespace(data=[-999.0, -2.0, -3.0])},
        ]
    )
    assert sft.mean_nll(output, datums) == 2
    output.loss_fn_outputs[1]["logprobs"].data[-1] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        sft.mean_nll(output, datums)
    with pytest.raises(ValueError, match="no supervised"):
        sft.token_mean_batch([datum([0])])


def test_data_rejects_overlap_bad_hash_and_test_split(tmp_path):
    generate(tmp_path)
    train, dev = [
        sft.read_jsonl(tmp_path / f"{split}.jsonl") for split in ("train", "dev")
    ]
    sft.validate_records(train, dev, tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        sft.validate_records(
            train, [{**train[0], "id": "dev", "split": "dev"}], tmp_path
        )
    with pytest.raises(ValueError, match="hash"):
        sft.validate_records([{**train[0], "image_sha256": "bad"}], dev, tmp_path)
    with pytest.raises(ValueError, match="train"):
        sft.validate_records([{**train[0], "split": "test"}], dev, tmp_path)


def test_cost_includes_masked_tokens_and_both_before_after():
    example = (
        {},
        SimpleNamespace(length=100),
        SimpleNamespace(model_input=SimpleNamespace(length=150)),
    )
    cost = sft.estimate_cost([example] * 8, [example] * 4, 4, 2048)
    assert cost["training_tokens"] == 4800
    assert cost["nll_forward_tokens"] == 3600
    assert cost["generation_prefill_tokens"] == 2400
    assert cost["generation_output_token_bound"] == 49152
    assert cost["estimated_usd_bound"] == pytest.approx(
        (4800 * 0.737 + 6000 * 0.33 + 49152 * 1.005) / 1e6
    )


def test_real_pinned_renderer_mask_and_generation_prefix(tmp_path):
    """Opt-in local integration; no service client or network is required with cache."""
    cookbook = os.environ.get("TINKER_COOKBOOK_DIR")
    if not cookbook:
        pytest.skip("Set TINKER_COOKBOOK_DIR for real pinned-renderer integration")
    generate(tmp_path, train_count=1, dev_count=1)
    rows = sft.read_jsonl(tmp_path / "train.jsonl")
    tokenizer, renderer = sft.load_renderer(sft.MODEL, sft.PROCESSOR_REVISION, cookbook)
    _, prompt, datum = sft.prepare_examples(rows, tmp_path, renderer, 1048576, 8192)[0]
    mask = datum.loss_fn_inputs["weights"].data
    targets = datum.loss_fn_inputs["target_tokens"].data
    first_answer = next(i for i, w in enumerate(mask) if w)
    assert first_answer == prompt.length - 1
    text = tokenizer.decode(
        [t for t, w in zip(targets, mask) if w], skip_special_tokens=True
    )
    assert text.strip() == (tmp_path / rows[0]["label"]).read_text().strip()
    assert targets[-1] in tokenizer.all_special_ids
    # Prompt's text+image chunks exactly match the teacher-forced prefix.
    expected = []
    for chunk in prompt.chunks:
        expected.extend(getattr(chunk, "tokens", [0] * chunk.length))
    assert targets[: prompt.length - 1] == expected[1:]
    with pytest.raises(ValueError, match="exceeds"):
        sft.prepare_examples(rows, tmp_path, renderer, 1048576, 20)


def test_budget_rejects_before_hosted_execution(tmp_path, monkeypatch):
    import sys

    data = tmp_path / "data"
    generate(data)
    output = tmp_path / "run"
    args = [
        "sft",
        "--train-manifest",
        str(data / "train.jsonl"),
        "--dev-manifest",
        str(data / "dev.jsonl"),
        "--data-root",
        str(data),
        "--output-dir",
        str(output),
        "--tinker-cookbook-dir",
        str(tmp_path),
        "--official-repo",
        str(tmp_path),
        "--execute",
        "--max-estimated-usd",
        "0.000001",
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(sft, "OfficialScorer", lambda _: None)
    monkeypatch.setattr(sft, "load_renderer", lambda *a: (None, None))
    example = (
        {},
        SimpleNamespace(length=100),
        SimpleNamespace(model_input=SimpleNamespace(length=150)),
    )
    monkeypatch.setattr(sft, "prepare_examples", lambda rows, *a: [example] * len(rows))
    monkeypatch.setattr(sft, "run", lambda *a: pytest.fail("Must not start hosted run"))
    with pytest.raises(ValueError, match="budget"):
        sft.main()
    assert not output.exists()
