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
    assert cost["nll_forward_tokens"] == 9000
    assert cost["nll_evaluations"] == 5
    assert cost["generation_prefill_tokens"] == 2800
    assert cost["generation_output_token_bound"] == 57344
    assert cost["estimated_usd_bound"] == pytest.approx(
        (4800 * 0.737 + 11800 * 0.33 + 57344 * 1.005) / 1e6
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


def test_monitor_boundaries_and_warmup():
    assert sft.evaluation_steps(16, 4) == [0, 4, 8, 12, 16]
    assert sft.evaluation_steps(10, 4) == [0, 4, 8, 10]
    assert sft.evaluation_steps(1, 4) == [0, 1]
    assert [sft.scheduled_lr(i, 5e-5, 2) for i in (1, 2, 3, 16)] == [
        2.5e-5,
        5e-5,
        5e-5,
        5e-5,
    ]
    assert sft.scheduled_lr(1, 1e-4, 0) == 1e-4


def test_periodic_run_evaluates_after_updates_without_training_on_dev(
    tmp_path, monkeypatch
):
    import json
    import math

    tinker = pytest.importorskip("tinker")
    root, output_dir = tmp_path / "data", tmp_path / "out"
    generate(root)
    output_dir.mkdir()
    train_rows, dev_rows = [
        sft.read_jsonl(root / f"{split}.jsonl") for split in ("train", "dev")
    ]

    def examples(rows, offset):
        return [
            (
                row,
                SimpleNamespace(length=2),
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints([offset + i, 10]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=[10, 11], dtype="int64"
                        ),
                        "weights": tinker.TensorData(data=[0.0, 1.0], dtype="float32"),
                    },
                ),
            )
            for i, row in enumerate(rows)
        ]

    train, dev = examples(train_rows, 0), examples(dev_rows, 100)
    rates, forward_steps, sampler_stages = [], [], []

    def future(value):
        return SimpleNamespace(result=lambda **kw: value)

    class Client:
        updates = 0

        def get_info(self):
            return SimpleNamespace(model_dump=lambda **kw: {})

        def result(self, data):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-1.0, -1 / (self.updates + 1)])}
                    for _ in data
                ]
            )

        def forward(self, data, **kw):
            forward_steps.append(self.updates)
            return future(self.result(data))

        def forward_backward(self, data, **kw):
            assert all(d.model_input.chunks[0].tokens[0] < 100 for d in data)
            assert sum(
                sum(d.loss_fn_inputs["weights"].data) for d in data
            ) == pytest.approx(1)
            return future(self.result(data))

        def optim_step(self, params):
            rates.append(params.learning_rate)
            self.updates += 1
            return future(SimpleNamespace(metrics={"unclipped_grad_l2:mean": 0.1}))

        def save_weights_for_sampler(self, stage, **kw):
            sampler_stages.append(stage)
            return future(SimpleNamespace(path="sampler-" + stage))

        def save_state(self, stage, **kw):
            return future(SimpleNamespace(path="state-" + stage))

    client = Client()
    monkeypatch.setattr(
        tinker,
        "ServiceClient",
        lambda: SimpleNamespace(
            create_lora_training_client=lambda **kw: client,
            create_sampling_client=lambda **kw: object(),
        ),
    )
    generated_counts = []

    def generate_predictions(client, examples, tokenizer, stop, args, path):
        generated_counts.append(len(examples))
        return {
            row["id"]: {
                "html": (root / row["label"]).read_text(),
                "input_tokens": 2,
                "output_tokens": 1,
                "stop_reason": "stop",
                "latency_seconds": 0,
            }
            for row, _, _ in examples
        }

    monkeypatch.setattr(sft, "generate", generate_predictions)
    args = SimpleNamespace(
        output_dir=output_dir,
        data_root=root,
        rank=8,
        seed=20260914,
        epochs=4,
        batch_size=2,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        eval_every=4,
        generate_every=8,
    )
    report = {"cost": sft.estimate_cost(train, dev, 4, 2048)}
    sft.run(
        args,
        train,
        dev,
        None,
        SimpleNamespace(get_stop_sequences=lambda: []),
        lambda a, b: 1.0,
        report,
    )
    assert report["status"] == "completed"
    assert forward_steps == [0, 0, 4, 4, 8, 8, 12, 12, 16, 16]
    assert rates[:3] == [2.5e-5, 5e-5, 5e-5]
    assert sampler_stages == ["before", "step_0008", "after"]
    assert generated_counts == [8, 4, 4, 8, 4]
    curve = [
        json.loads(line)
        for line in (output_dir / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [entry["step"] for entry in curve] == [0, 4, 8, 12, 16]
    assert curve[-1]["dev/assistant_perplexity"] == pytest.approx(math.exp(1 / 17))
    assert len(list(output_dir.glob("*_likelihoods.json"))) == 10
