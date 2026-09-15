"""No-network OPD service contracts, journaling, budgets, and update ordering."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import opd

tinker = pytest.importorskip("tinker")


class Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class Tokenizer:
    def __len__(self):
        return 10

    def decode(self, tokens, skip_special_tokens):
        return "<table><tr><td>PRIVATE_MARKER</td></tr></table>"


def lp_output(datums, value=-2):
    return SimpleNamespace(
        loss_fn_outputs=[
            {
                "logprobs": tinker.TensorData.from_numpy(
                    np.full(d.model_input.length, value, dtype=np.float32)
                )
            }
            for d in datums
        ]
    )


@pytest.fixture
def rig(tmp_path, monkeypatch):
    args = SimpleNamespace(
        output_dir=tmp_path,
        max_estimated_usd=1,
        rank=8,
        seed=42,
        epochs=1,
        batch_size=2,
        max_new_tokens=3,
        expected_output_tokens=2,
        inference_concurrency=1,
        learning_rate=1e-4,
        eval_every=1,
        generate_dev=False,
        data_root=tmp_path,
    )
    (tmp_path / "rollouts").mkdir()
    events, logged = [], []
    prompt = tinker.ModelInput.from_ints([8, 9])
    train = [
        ({"id": f"PRIVATE_MARKER_{i}", "image_sha256": "digest"}, prompt)
        for i in range(3)
    ]
    gold = tinker.Datum(
        model_input=prompt,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array([0, 1])),
            "weights": tinker.TensorData.from_numpy(np.array([0, 1], dtype=np.float32)),
        },
    )
    dev = [({"id": f"dev{i}"}, prompt, gold) for i in range(2)]

    class Teacher:
        def compute_logprobs(self, full):
            events.append(("teacher", client.step))
            # At least this rollout's raw sampler response is durable already.
            records = [
                json.loads(p.read_text())
                for p in (tmp_path / "rollouts").glob("*.json")
                if "learner" not in p.name
            ]
            assert any("teacher_full_logprobs" not in r for r in records)
            assert full.length == 4
            assert full.chunks[-1].tokens == [1, 2]
            # Prompt LPs may be missing; score from position P, not P-1.
            return Future([None, None, -1.0, -3.0])

    class Sampler:
        def __init__(self, step):
            self.step = step

        def sample(self, *, prompt, num_samples, sampling_params):
            assert self.step == client.step
            assert num_samples == 1
            assert sampling_params.temperature == sampling_params.top_p == 1
            assert sampling_params.top_k == -1
            assert sampling_params.max_tokens == args.max_new_tokens
            assert sampling_params.stop == [2]
            journal = json.loads((tmp_path / "usage.json").read_text())
            assert journal["pending"]["key"].startswith("step_")
            events.append(("sample", self.step))
            return Future(
                SimpleNamespace(
                    sequences=[
                        SimpleNamespace(
                            tokens=[1, 2], logprobs=[-2.0, -2.0], stop_reason="stop"
                        )
                    ]
                )
            )

    class Client:
        step = 0

        def get_info(self):
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"private": "PRIVATE_MARKER"}
            )

        def save_weights_and_get_sampling_client(self, *, retry_config):
            assert retry_config.enable_retry_logic is False
            events.append(("snapshot", self.step))
            return Sampler(self.step)

        def forward(self, datums, *, loss_fn):
            assert loss_fn == "cross_entropy"
            assert len(datums) == len(dev)
            events.append(("dev", self.step))
            return Future(lp_output(datums))

        def forward_backward(self, datums, *, loss_fn):
            assert loss_fn == "importance_sampling"
            for datum in datums:
                if set(datum.loss_fn_inputs) != {
                    "target_tokens",
                    "logprobs",
                    "advantages",
                }:
                    raise ValueError(
                        "importance_sampling received unsupported input keys"
                    )
            # This fixture always has two completion tokens per sampled datum.
            count = 2 * len(datums)
            for datum in datums:
                np.testing.assert_allclose(
                    datum.loss_fn_inputs["advantages"].to_numpy(),
                    [0, 1 / count, -1 / count],
                )
            events.append(("backward", self.step))
            return Future(lp_output(datums))

        def optim_step(self, params):
            assert (
                tmp_path / "rollouts" / f"{self.step + 1:04d}_learner.json"
            ).exists()
            assert params.learning_rate == args.learning_rate
            events.append(("optim", self.step))
            self.step += 1
            return Future(None)

        def save_weights_for_sampler(self, name, *, ttl_seconds):
            assert ttl_seconds == 7 * 86400
            events.append(("save", self.step))
            return Future(SimpleNamespace(path=f"PRIVATE_MARKER/{name}"))

        def save_state(self, name, *, ttl_seconds):
            events.append(("state", self.step))
            return Future(SimpleNamespace(path=f"PRIVATE_MARKER/{name}"))

    client = Client()

    class Service:
        def create_sampling_client(
            self, *, base_model=None, model_path=None, retry_config
        ):
            if model_path is not None:
                assert model_path.endswith("/after")
                return object()
            assert base_model == opd.TEACHER_MODEL
            assert retry_config.enable_retry_logic is False
            return Teacher()

        def create_lora_training_client(
            self, *, base_model, rank, seed, train_attn, train_mlp, train_unembed
        ):
            assert base_model == opd.MODEL
            assert train_attn and train_mlp and not train_unembed
            return client

    monkeypatch.setattr(tinker, "ServiceClient", Service)
    report = {"optimizer_steps": 2, "warmup_steps": 0}
    logger = SimpleNamespace(log=logged.append)
    renderer = SimpleNamespace(get_stop_sequences=lambda: [2])
    return SimpleNamespace(
        args=args,
        train=train,
        dev=dev,
        events=events,
        logged=logged,
        client=client,
        report=report,
        logger=logger,
        renderer=renderer,
    )


def run_rig(rig):
    opd.run(
        rig.args,
        rig.train,
        rig.dev,
        Tokenizer(),
        rig.renderer,
        None,
        rig.report,
        rig.logger,
    )


def test_fresh_rollout_one_update_full_dev_and_private_artifact_contract(rig):
    run_rig(rig)
    assert rig.events == [
        ("dev", 0),
        ("snapshot", 0),
        ("sample", 0),
        ("teacher", 0),
        ("sample", 0),
        ("teacher", 0),
        ("backward", 0),
        ("optim", 0),
        ("save", 1),
        ("dev", 1),
        ("snapshot", 1),
        ("sample", 1),
        ("teacher", 1),
        ("backward", 1),
        ("optim", 1),
        ("state", 2),
        ("save", 2),
        ("dev", 2),
    ]
    report = json.loads((rig.args.output_dir / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["checks"] == {
        "completed_optimizer_steps": 2,
        "full_dev_examples": 2,
        "rollouts": 3,
    }
    assert report["usage"] == {
        "rollout_tokens": 6,
        "prefill_tokens": 6,
        "teacher_forward_tokens": 12,
        "training_tokens": 9,
        "format_invalid": 0,
        "truncated": 0,
    }
    assert report["evaluation_stages"] == ["before", "step_0001", "after"]
    records = [
        json.loads(p.read_text())
        for p in (rig.args.output_dir / "rollouts").glob("*.json")
        if "learner" not in p.name
    ]
    assert len(records) == 3
    for record in records:
        assert record["tokens"] == [1, 2]
        assert record["sampling_logprobs"] == [-2, -2]
        assert record["teacher_full_logprobs"] == [None, None, -1, -3]
        assert "PRIVATE_MARKER" in record["html"]
    assert sorted(r["optimizer_step_before_rollout"] for r in records) == [0, 0, 1]
    # Only scalar aggregate metrics cross the logging boundary.
    assert "PRIVATE_MARKER" not in json.dumps(rig.logged)
    assert all(
        isinstance(value, (float, int))
        for event in rig.logged
        for value in event.values()
    )
    journal = json.loads((rig.args.output_dir / "usage.json").read_text())
    expected = (
        6 * opd.FORWARD_RATE
        + 6 * opd.SAMPLE_RATE
        + 12 * opd.TEACHER_FORWARD_RATE
        + 9 * opd.TRAIN_RATE
        + 12 * opd.FORWARD_RATE
        + 3 * opd.TEACHER_SAMPLE_RATE
    ) / 1e6
    assert report["estimated_compute_usd"] == pytest.approx(expected)
    assert journal["estimated_compute_usd"] == pytest.approx(expected)
    assert journal["pending"] is None


def test_initial_dev_budget_failure_happens_before_any_compute(rig):
    rig.args.max_estimated_usd = 1e-12
    with pytest.raises(ValueError, match="budget reached"):
        run_rig(rig)
    assert rig.events == []
    assert rig.client.step == 0


def test_rollout_budget_failure_never_samples_scores_or_updates(rig):
    # Enough for initial Dev, insufficient for the first reserved rollout batch.
    rig.args.max_estimated_usd = (
        2 * sum(d.model_input.length for _, _, d in rig.dev) * opd.FORWARD_RATE / 1e6
    )
    with pytest.raises(ValueError, match="budget reached"):
        run_rig(rig)
    assert rig.events == [("dev", 0)]
    assert rig.client.step == 0
    assert not list((rig.args.output_dir / "rollouts").iterdir())


def test_invalid_sampler_response_is_saved_and_never_sent_to_teacher(rig):
    class InvalidSampler:
        def sample(self, **kwargs):
            return Future(
                SimpleNamespace(
                    sequences=[
                        SimpleNamespace(
                            tokens=[1], logprobs=[-99999.0], stop_reason="stop"
                        )
                    ]
                )
            )

    class NoTeacher:
        def compute_logprobs(self, full):
            pytest.fail("Invalid sampler LP must fail before teacher request")

    row, prompt = rig.train[0]
    with pytest.raises(ValueError):
        opd.collect_one(
            InvalidSampler(), NoTeacher(), row, prompt, Tokenizer(), [2], rig.args, 1, 0
        )
    record = json.loads((rig.args.output_dir / "rollouts" / "0001_00.json").read_text())
    assert record["sampling_logprobs"] == [-99999]
    assert "teacher_full_logprobs" not in record


def test_cost_bound_includes_every_dev_pass_and_caps_unequal_batches(rig):
    cost = opd.estimate_cost(rig.train, rig.dev, rig.args)
    assert cost["training_tokens"] == 12
    assert cost["nll_forward_tokens"] == 12
    expected = (
        6 * opd.FORWARD_RATE
        + 9 * opd.SAMPLE_RATE
        + 15 * opd.TEACHER_FORWARD_RATE
        + 12 * opd.TRAIN_RATE
        + 12 * opd.FORWARD_RATE
        + 3 * opd.TEACHER_SAMPLE_RATE
    ) / 1e6
    assert cost["estimated_usd_bound"] == pytest.approx(expected)
    assert cost["estimated_usd_length_scenario"] < cost["estimated_usd_bound"]
    rig.args.generate_dev = True
    with_generation = opd.estimate_cost(rig.train, rig.dev, rig.args)
    assert with_generation["estimated_usd_bound"] - cost[
        "estimated_usd_bound"
    ] == pytest.approx((4 * opd.FORWARD_RATE + 6 * opd.SAMPLE_RATE) / 1e6)


def test_final_generation_follows_complete_final_dev_nll_and_is_budgeted(
    rig, monkeypatch
):
    rig.args.generate_dev = True

    def generate(sampler, examples, tokenizer, stop, args, path):
        assert rig.events[-1] == ("dev", 2)
        assert len(examples) == len(rig.dev)
        journal = json.loads((args.output_dir / "usage.json").read_text())
        assert journal["pending"] is None
        assert isinstance(sampler, opd.BudgetedSampler)
        # The real wrapper has separate raw-response and concurrency tests.
        for index, (_, prompt, _) in enumerate(examples):
            request = sampler.budget.for_example()
            request.reserve(
                str(index),
                (
                    prompt.length * opd.FORWARD_RATE
                    + args.max_new_tokens * opd.SAMPLE_RATE
                )
                / 1e6,
            )
            request.settle(
                (prompt.length * opd.FORWARD_RATE + 2 * opd.SAMPLE_RATE) / 1e6
            )
        rig.events.append(("generation", 2))
        return {row["id"]: {"output_tokens": 2} for row, _, _ in examples}

    def evaluate(rows, root, predictions, scorer):
        assert len(rows) == len(predictions) == len(rig.dev)
        return {"private": "PRIVATE_MARKER"}, {"eval/cell_f1": 0.5}

    monkeypatch.setattr(opd, "generate", generate)
    monkeypatch.setattr(opd, "evaluate", evaluate)
    run_rig(rig)
    assert rig.events[-1] == ("generation", 2)
    assert rig.report["after"]["dev"]["eval/cell_f1"] == 0.5
    assert "PRIVATE_MARKER" not in json.dumps(rig.logged)
    expected = (
        6 * opd.FORWARD_RATE
        + 6 * opd.SAMPLE_RATE
        + 12 * opd.TEACHER_FORWARD_RATE
        + 9 * opd.TRAIN_RATE
        + 12 * opd.FORWARD_RATE
        + 3 * opd.TEACHER_SAMPLE_RATE
        + 4 * opd.FORWARD_RATE
        + 4 * opd.SAMPLE_RATE
    ) / 1e6
    assert rig.report["estimated_compute_usd"] == pytest.approx(expected)


def test_bad_teacher_vector_is_persisted_before_alignment_rejection(rig):
    sampler = SimpleNamespace(
        sample=lambda **kwargs: Future(
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        tokens=[1, 2], logprobs=[-2, -2], stop_reason="stop"
                    )
                ]
            )
        )
    )
    teacher = SimpleNamespace(compute_logprobs=lambda full: Future([None, -1]))
    row, prompt = rig.train[0]
    with pytest.raises(ValueError, match="length differs"):
        opd.collect_one(sampler, teacher, row, prompt, Tokenizer(), [2], rig.args, 1, 0)
    record = json.loads((rig.args.output_dir / "rollouts" / "0001_00.json").read_text())
    assert record["teacher_full_logprobs"] == [None, -1]
    assert record["tokens"] == [1, 2]


def test_native_input_rejection_never_optimizes_and_preserves_paid_evidence(
    rig, monkeypatch
):
    from modeling.llm_post_training.vlm_table_extraction_lab.opd_targets import (
        normalize_batch,
    )

    # Reproduce the native API rejection if diagnostic-only mask leaks again.
    monkeypatch.setattr(opd, "native_batch", normalize_batch)
    with pytest.raises(ValueError, match="unsupported input keys"):
        run_rig(rig)
    assert rig.client.step == 0
    assert not any(event[0] == "optim" for event in rig.events)
    records = list((rig.args.output_dir / "rollouts").glob("0001_*.json"))
    assert len(records) == 2
    assert all("teacher_full_logprobs" in json.loads(p.read_text()) for p in records)
    assert not (rig.args.output_dir / "rollouts" / "0001_learner.json").exists()
    journal = json.loads((rig.args.output_dir / "usage.json").read_text())
    assert journal["pending"]["key"] == "step_1:rollout_teacher_backward"
