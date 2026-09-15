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
        max_pixels=1024,
        max_sequence_tokens=32,
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
        def sample(
            self,
            *,
            prompt,
            num_samples,
            include_prompt_logprobs,
            topk_prompt_logprobs,
            sampling_params,
        ):
            assert prompt.length == 4 and prompt.chunks[-1].tokens == [1, 2]
            assert num_samples == 1 and include_prompt_logprobs
            assert topk_prompt_logprobs == 2
            assert sampling_params.max_tokens == 1 and sampling_params.temperature == 0
            events.append(("teacher_topk", client.step))
            return Future(
                SimpleNamespace(
                    prompt_logprobs=[None, None, -1, -3],
                    topk_prompt_logprobs_np=SimpleNamespace(
                        token_ids=np.tile([3, 4], (4, 1)),
                        logprobs=np.log([[0.6, 0.3]] * 4),
                    ),
                )
            )

        def compute_logprobs(self, full):
            events.append(("teacher", client.step))
            # At least this rollout's raw sampler response is durable already.
            records = [
                json.loads(p.read_text())
                for p in (tmp_path / "rollouts").glob("*.json")
                if "learner" not in p.name and "policy" not in p.name
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
        optim_metrics = None

        def get_info(self):
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"private": "PRIVATE_MARKER"}
            )

        def save_weights_and_get_sampling_client(self, *, retry_config):
            assert retry_config.enable_retry_logic is False
            events.append(("snapshot", self.step))
            return Sampler(self.step)

        def forward(self, datums, *, loss_fn):
            if loss_fn == "importance_sampling":
                assert all(
                    set(d.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
                    for d in datums
                )
                events.append(("policy_forward", self.step))
                assert (
                    json.loads((tmp_path / "usage.json").read_text())["pending"]
                    is not None
                )
                return Future(lp_output(datums, -2 + 0.05 * self.step))
            assert loss_fn == "cross_entropy"
            assert all(
                set(d.loss_fn_inputs) == {"target_tokens", "weights"} for d in datums
            )
            if datums[0].model_input.length == 4:
                assert len(datums) == args.train_probe_examples
                events.append(("probe", self.step))
            else:
                assert len(datums) == len(dev)
                events.append(("dev", self.step))
            return Future(lp_output(datums))

        def forward_backward(self, datums, *, loss_fn):
            if loss_fn == "cross_entropy":
                assert all(
                    set(d.loss_fn_inputs) == {"target_tokens", "weights"}
                    for d in datums
                )
                soft = datums[0].loss_fn_inputs["weights"].to_numpy().ndim == 2
                total = sum(
                    d.loss_fn_inputs["weights"].to_numpy().sum() for d in datums
                )
                assert total == pytest.approx(1 if soft else args.gold_weight)
                events.append(("soft_backward" if soft else "gold_backward", self.step))
                if soft:
                    return Future(
                        SimpleNamespace(
                            loss_fn_outputs=[
                                {
                                    "logprobs": tinker.TensorData.from_numpy(
                                        np.full(
                                            d.loss_fn_inputs["weights"].shape,
                                            -2,
                                            dtype=np.float32,
                                        )
                                    )
                                }
                                for d in datums
                            ]
                        )
                    )
                return Future(lp_output(datums))
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
            mix = (
                1 - args.gold_weight
                if getattr(args, "objective", None) == "hybrid_gold"
                else 1
            )
            for datum in datums:
                np.testing.assert_allclose(
                    datum.loss_fn_inputs["advantages"].to_numpy(),
                    [0, mix / count, -mix / count],
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
            return Future(SimpleNamespace(metrics=self.optim_metrics))

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
                assert model_path.startswith("PRIVATE_MARKER/")
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


def configure_ablation(rig, monkeypatch, objective, *, diagnostic_every=1, probes=1):
    from modeling.llm_post_training.vlm_table_extraction_lab import kd_collect

    rig.args.objective = objective
    rig.args.gold_weight = 0.25
    rig.args.diagnostic_every = diagnostic_every
    rig.args.train_probe_examples = probes
    rig.args.generate_dev_every = 0
    (rig.args.data_root / "label.html").write_text(
        "<table><tr><td>PRIVATE_MARKER</td></tr></table>"
    )
    for row, _ in rig.train:
        row["label"] = "label.html"
    monkeypatch.setattr(opd, "TOP_K", 2)
    monkeypatch.setattr(kd_collect, "TOP_K", 2)

    def prepare(rows, root, renderer, max_pixels, max_sequence_tokens):
        return [
            (
                row,
                prompt,
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints([0, 0, 1, 2]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_numpy(
                            np.array([0, 0, 1, 2])
                        ),
                        "weights": tinker.TensorData.from_numpy(
                            np.array([0, 0, 1, 1], dtype=np.float32)
                        ),
                    },
                ),
            )
            for row in rows
            for _, prompt in rig.train[:1]
        ]

    monkeypatch.setattr(opd, "prepare_examples", prepare)
    return prepare([r for r, _ in rig.train], None, None, None, None)


@pytest.mark.parametrize(
    "objective", ["sampled_reverse_kl", "topk_forward_kl", "hybrid_gold"]
)
def test_all_objectives_strict_native_schemas_diagnostics_probes_and_exact_usage(
    rig, monkeypatch, objective
):
    gold = configure_ablation(rig, monkeypatch, objective)
    rig.client.optim_metrics = {
        "unclipped_grad_l2:mean": 2.5,
        "private_unexpected": "PRIVATE_MARKER",
    }
    run_rig(rig)
    assert [event for event in rig.events if event[0] in {"dev", "probe"}] == [
        ("dev", 0),
        ("probe", 0),
        ("dev", 1),
        ("probe", 1),
        ("dev", 2),
        ("probe", 2),
    ]
    assert [event for event in rig.events if event[0] == "optim"] == [
        ("optim", 0),
        ("optim", 1),
    ]
    diag_passes = 2 if objective == "topk_forward_kl" else 1
    forwards = [event for event in rig.events if event[0] == "policy_forward"]
    assert forwards == (
        [
            ("policy_forward", 0),
            ("policy_forward", 1),
            ("policy_forward", 1),
            ("policy_forward", 2),
        ]
        if diag_passes == 2
        else [("policy_forward", 1), ("policy_forward", 2)]
    )
    for step in range(2):
        if objective == "hybrid_gold":
            assert (
                rig.events.index(("backward", step))
                < rig.events.index(("gold_backward", step))
                < rig.events.index(("optim", step))
            )
        elif objective == "topk_forward_kl":
            assert rig.events.index(("soft_backward", step)) < rig.events.index(
                ("optim", step)
            )
        assert rig.events.index(("optim", step)) < rig.events.index(
            ("policy_forward", step + 1)
        )
    extra = rig.report["extra_usage"]
    assert extra == {
        "gold_training_tokens": 12 if objective == "hybrid_gold" else 0,
        "diagnostic_forward_tokens": 9 * diag_passes,
        "train_probe_forward_tokens": 12,
    }
    expected = (
        6 * opd.FORWARD_RATE
        + 6 * opd.SAMPLE_RATE
        + 12 * opd.TEACHER_FORWARD_RATE
        + 9 * opd.TRAIN_RATE
        + 12 * opd.FORWARD_RATE
        + 3 * opd.TEACHER_SAMPLE_RATE
        + extra["gold_training_tokens"] * opd.TRAIN_RATE
        + (extra["diagnostic_forward_tokens"] + 12) * opd.FORWARD_RATE
    ) / 1e6
    assert rig.report["estimated_compute_usd"] == pytest.approx(expected)
    cost = opd.estimate_cost(rig.train, rig.dev, rig.args, gold)
    assert cost["gold_training_tokens"] == extra["gold_training_tokens"]
    assert cost["train_probe_forward_tokens"] == 12
    assert cost["diagnostic_forward_tokens_bound"] == 16 * diag_passes
    assert cost["estimated_usd_bound"] >= expected
    steps = [
        json.loads(line)
        for line in (rig.args.output_dir / "steps.jsonl").read_text().splitlines()
    ]
    assert all(
        s["optimizer/gradient_norm_available"] == 1
        and s["optimizer/unclipped_gradient_norm"] == 2.5
        and s["optimizer/gradient_norm_exceeds_clip_threshold"] == 1
        for s in steps
    )
    assert all("opd_diagnostics/update_log_ratio_mean" in s for s in steps)
    assert "PRIVATE_MARKER" not in json.dumps(rig.logged)
    for step in (1, 2):
        for suffix in ("pre_policy", "post_policy"):
            assert (
                rig.args.output_dir / "rollouts" / f"{step:04d}_{suffix}.json"
            ).exists()
    if objective == "topk_forward_kl":
        assert len(list((rig.args.output_dir / "rollouts").glob("*.raw-topk.npz"))) == 3
        assert all(
            s["training/retained_mass_mean"] == pytest.approx(0.9) for s in steps
        )


@pytest.mark.parametrize("grad", [None, float("nan"), -1, True])
def test_missing_or_invalid_gradient_metric_stays_unavailable(rig, grad):
    rig.client.optim_metrics = (
        None if grad is None else {"unclipped_grad_l2:mean": grad}
    )
    run_rig(rig)
    events = [e for e in rig.logged if "optimizer/gradient_norm_available" in e]
    assert len(events) == 2
    assert all(e["optimizer/gradient_norm_available"] == 0 for e in events)
    assert all("optimizer/unclipped_gradient_norm" not in e for e in events)
    assert all(
        "optimizer/gradient_norm_exceeds_clip_threshold" not in e for e in events
    )


@pytest.mark.parametrize("objective", ["topk_forward_kl", "hybrid_gold"])
def test_extra_gold_and_diagnostic_cost_is_reserved_before_rollout(
    rig, monkeypatch, objective
):
    configure_ablation(rig, monkeypatch, objective)
    initial = 8 * opd.FORWARD_RATE / 1e6
    baseline_batch = (
        4 * opd.FORWARD_RATE
        + 6 * opd.SAMPLE_RATE
        + 10 * opd.TEACHER_FORWARD_RATE
        + 2 * opd.TEACHER_SAMPLE_RATE
        + 8 * opd.TRAIN_RATE
    ) / 1e6
    rig.args.max_estimated_usd = 1.1 * (initial + baseline_batch) + 1e-12
    with pytest.raises(ValueError, match="budget reached"):
        run_rig(rig)
    assert rig.events == [("dev", 0), ("probe", 0)]
    assert rig.client.step == 0


def test_periodic_full_dev_generation_settles_budget_then_training_continues(
    rig, monkeypatch
):
    configure_ablation(rig, monkeypatch, "sampled_reverse_kl", diagnostic_every=0)
    rig.args.generate_dev = True
    rig.args.generate_dev_every = 1

    def generate(sampler, examples, tokenizer, stop, args, path):
        step = rig.client.step
        assert rig.events[-2:] == [("dev", step), ("probe", step)]
        assert len(examples) == len(rig.dev)
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
        rig.events.append(("generation", step))
        return {row["id"]: {"output_tokens": 2} for row, _, _ in examples}

    monkeypatch.setattr(opd, "generate", generate)
    monkeypatch.setattr(opd, "evaluate", lambda *args: ({}, {"eval/cell_f1": 0.5}))
    run_rig(rig)
    assert [event for event in rig.events if event[0] == "generation"] == [
        ("generation", 1),
        ("generation", 2),
    ]
    assert rig.events.index(("generation", 1)) < rig.events.index(("snapshot", 1))
    cost = opd.estimate_cost(rig.train, rig.dev, rig.args)
    assert cost["dev_generation_stages"] == 2
    expected = (
        6 * opd.FORWARD_RATE
        + 6 * opd.SAMPLE_RATE
        + 12 * opd.TEACHER_FORWARD_RATE
        + 9 * opd.TRAIN_RATE
        + 12 * opd.FORWARD_RATE
        + 3 * opd.TEACHER_SAMPLE_RATE
        + 12 * opd.FORWARD_RATE
        + 8 * opd.FORWARD_RATE
        + 8 * opd.SAMPLE_RATE
    ) / 1e6
    assert rig.report["estimated_compute_usd"] == pytest.approx(expected)


@pytest.mark.parametrize("objective", ["sampled_reverse_kl", "topk_forward_kl"])
def test_sparse_diagnostics_cover_first_and_final_steps_without_extra_middle_forward(
    rig, monkeypatch, objective
):
    configure_ablation(rig, monkeypatch, objective, diagnostic_every=10, probes=0)
    rig.args.batch_size = 1
    rig.args.eval_every = 2
    rig.report["optimizer_steps"] = 3
    run_rig(rig)
    forwards = [event for event in rig.events if event[0] == "policy_forward"]
    expected = [("policy_forward", 1), ("policy_forward", 3)]
    if objective == "topk_forward_kl":
        expected = [
            ("policy_forward", 0),
            ("policy_forward", 1),
            ("policy_forward", 2),
            ("policy_forward", 3),
        ]
    assert forwards == expected
    assert [event for event in rig.events if event[0] == "dev"] == [
        ("dev", 0),
        ("dev", 2),
        ("dev", 3),
    ]
    rows = [
        json.loads(line)
        for line in (rig.args.output_dir / "steps.jsonl").read_text().splitlines()
    ]
    assert "opd_diagnostics/update_log_ratio_mean" in rows[0]
    assert "opd_diagnostics/update_log_ratio_mean" not in rows[1]
    assert "opd_diagnostics/update_log_ratio_mean" in rows[2]
    assert all("training/rollout_cell_f1" in row for row in rows)
    assert rig.report["extra_usage"]["diagnostic_forward_tokens"] == (
        12 if objective == "topk_forward_kl" else 6
    )
