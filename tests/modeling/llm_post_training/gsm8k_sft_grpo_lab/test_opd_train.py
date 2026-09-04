import asyncio
import json

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import build_manifest, content_id
from modeling.llm_post_training.gsm8k_sft_grpo_lab.opd_train import (
    E10Config,
    _normalized_topk,
    _student_topk_datums,
    build_doctor_report,
    run_e10_training,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.opd_train import TopKPrefixTarget


class _FakeModelInput:
    @staticmethod
    def from_ints(tokens):
        return tuple(tokens)


class _FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class _FakeAdamParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTinker:
    ModelInput = _FakeModelInput
    SamplingParams = _FakeSamplingParams

    class types:
        Datum = _FakeDatum
        AdamParams = _FakeAdamParams


class _FakeFuture:
    def __init__(self, result):
        self.result = result

    async def result_async(self):
        return self.result


class _FakeTokenizer:
    eos_token_id = None

    def encode(self, text, **kwargs):
        return [1 + index % 5 for index, _ in enumerate(text.split() or ["x"])]

    def decode(self, tokens):
        return "work \\boxed{1}"

    def get_vocab(self):
        return {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


class _FakeSamplingClient:
    def get_tokenizer(self):
        return _FakeTokenizer()

    async def sample_async(self, prompt, num_samples, sampling_params, **kwargs):
        sequence = type("Sequence", (), {"tokens": [1, 2, 3]})()
        topk = [None] + [[(1, -0.2), (2, -2.0)] for _ in range(len(prompt) - 1)]
        return type(
            "Result",
            (),
            {"sequences": [sequence] * num_samples, "topk_prompt_logprobs": topk},
        )()


class _FakeTrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
        self.saved = []

    def get_tokenizer(self):
        return _FakeTokenizer()

    async def save_weights_and_get_sampling_client_async(self):
        return _FakeSamplingClient()

    async def forward_backward_async(self, data, loss_fn):
        self.forward_backward_calls.append((data, loss_fn))
        loss_sum = sum(
            sum(datum.loss_fn_inputs["weights"]) * 1.2 for datum in data
        )
        return _FakeFuture(type("Result", (), {"metrics": {"loss:sum": loss_sum}})())

    async def optim_step_async(self, params):
        return _FakeFuture(object())

    async def save_state_async(self, name, ttl_seconds):
        self.saved.append(("state", name, ttl_seconds))
        return _FakeFuture(type("Result", (), {"path": f"tinker://state/{name}"})())

    async def save_weights_for_sampler_async(self, name, ttl_seconds):
        self.saved.append(("sampler", name, ttl_seconds))
        return _FakeFuture(type("Result", (), {"path": f"tinker://sampler/{name}"})())


class _FakeServiceClient:
    def __init__(self):
        self.training_client = _FakeTrainingClient()
        self.from_state = None

    async def create_training_client_from_state_async(self, path, **kwargs):
        self.from_state = (path, kwargs)
        return self.training_client

    async def create_sampling_client_async(self, **kwargs):
        return _FakeSamplingClient()


class _FakeRun:
    id = "unit-test-run"
    url = "https://wandb.example/e10"

    def __init__(self):
        self.logs = []
        self.summary = {}
        self.defined_metrics = []
        self.finished = False

    def define_metric(self, name, **kwargs):
        self.defined_metrics.append((name, kwargs))

    def log(self, payload, step):
        self.logs.append((payload, step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    class Table:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    def __init__(self):
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def _rows(prefix, count):
    return [
        {"question": f"{prefix} question {index}", "answer": "reasoning #### 1"}
        for index in range(count)
    ]


def _manifest_and_rows():
    rows = _rows("train", 193)
    manifest = build_manifest(
        rows,
        _rows("test", 2),
        "revision",
        sft_train_count=64,
        sft_validation_count=64,
        rl_train_count=64,
        rl_monitor_count=1,
        calibration_test_count=1,
    )
    by_id = {content_id(row): row for row in rows}
    return manifest, tuple(by_id[item] for item in manifest.rl_train_ids), tuple(
        by_id[item] for item in manifest.sft_validation_ids
    )


def _trace_file(tmp_path, rows):
    path = tmp_path / "e9_traces.jsonl"
    path.write_text(
        "".join(
            json.dumps({"example_id": content_id(row), "response": "teacher \\boxed{1}"})
            + "\n"
            for row in rows
        )
    )
    return path


def _config(tmp_path, rows):
    return E10Config(reference_trace_path=str(_trace_file(tmp_path, rows)), trace_output_dir=str(tmp_path))


def test_topk_target_is_normalized_and_materialized_as_weighted_ce():
    token_ids, probabilities, mass, entropy = _normalized_topk([(4, -0.3), (5, -1.5)])
    assert token_ids == (4, 5)
    assert sum(probabilities) == pytest.approx(1.0)
    assert mass == pytest.approx(0.740818221 + 0.223130160, rel=1e-6)
    assert entropy > 0.0
    target = TopKPrefixTarget(
        example_id="example",
        rollout_id=0,
        prefix_position=1,
        student_input_tokens=(10, 11),
        teacher_token_ids=token_ids,
        teacher_probs=probabilities,
        teacher_topk_mass=mass,
        teacher_entropy=entropy,
    )
    data, input_tokens, weighted_positions = _student_topk_datums((target,), _FakeTinker)
    assert len(data) == 2
    assert input_tokens == 4
    assert weighted_positions == pytest.approx(1.0)
    assert data[0].loss_fn_inputs["weights"][:-1] == [0.0]
    assert data[0].loss_fn_inputs["target_tokens"][-1] == 4


def test_e10_preflight_and_run_preserve_parent_and_split_provenance(tmp_path):
    manifest, rl_rows, development_rows = _manifest_and_rows()
    config = _config(tmp_path, rl_rows)
    doctor = build_doctor_report(
        config,
        manifest,
        environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
    )
    assert doctor["network_called"] is False
    assert doctor["formal_overlap_count"] == 0
    assert doctor["development_overlap_count"] == 0
    assert doctor["estimated_cost_breakdown_usd"]["total_max_usd"] < config.hard_cap_usd

    service = _FakeServiceClient()
    wandb = _FakeWandb()
    report = asyncio.run(
        run_e10_training(
            config,
            allow_paid=True,
            manifest=manifest,
            environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            tinker_module=_FakeTinker,
            wandb_module=wandb,
            service_client=service,
            rl_rows=rl_rows,
            development_rows=development_rows,
        )
    )
    assert service.from_state[0] == config.parent_state_path
    assert len(service.training_client.forward_backward_calls) == config.training_steps
    assert all(loss_fn == "cross_entropy" for _, loss_fn in service.training_client.forward_backward_calls)
    assert report["formal_overlap_count"] == 0
    assert report["selected_checkpoint"]["step"] == 0
    assert report["actual_token_priced_total_usd"] <= config.hard_cap_usd
    assert (tmp_path / "e10_opd_report_unit-test-run.json").exists()
    assert (tmp_path / "e10_metric_dictionary_unit-test-run.md").exists()
    assert any("train/topk_teacher_to_student_kl" in payload for payload, _ in wandb.run.logs)
    assert any("data/on_policy_group_unique_response_frac" in payload for payload, _ in wandb.run.logs)
    assert wandb.run.summary["selection/selected_is_initialization"] == 1.0
    assert wandb.run.finished is True
