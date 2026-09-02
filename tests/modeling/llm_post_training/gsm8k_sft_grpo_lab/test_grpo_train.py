import asyncio

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import build_manifest
from modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train import (
    GRPOConfig,
    GRPOTrainingError,
    build_doctor_report,
    estimate_max_token_cost_usd,
    run_grpo_training,
)


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
    def encode(self, text, **kwargs):
        return [10, 11]

    def decode(self, tokens):
        return f"\\boxed{{{tokens[0]}}}"


class _FakeSamplingClient:
    def get_tokenizer(self):
        return _FakeTokenizer()

    async def sample_async(self, prompt, num_samples, sampling_params):
        sequences = [
            type("Sequence", (), {"tokens": [1], "logprobs": [-0.2]})(),
            type("Sequence", (), {"tokens": [0], "logprobs": [-0.3]})(),
            type("Sequence", (), {"tokens": [1], "logprobs": [-0.4]})(),
            type("Sequence", (), {"tokens": [0], "logprobs": [-0.5]})(),
        ]
        return type("Result", (), {"sequences": sequences[:num_samples]})()


class _FakeTrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
        self.optim_calls = []
        self.saved = []

    def get_tokenizer(self):
        return _FakeTokenizer()

    async def save_weights_and_get_sampling_client_async(self):
        return _FakeSamplingClient()

    async def forward_backward_async(self, data, loss_fn):
        self.forward_backward_calls.append((data, loss_fn))
        return _FakeFuture(type("Result", (), {"metrics": {"loss:sum": 0.25}})())

    async def optim_step_async(self, params):
        self.optim_calls.append(params)
        return _FakeFuture(object())

    async def save_state_async(self, name, ttl_seconds):
        self.saved.append(("state", name, ttl_seconds))
        return _FakeFuture(
            type("Result", (), {"path": f"tinker://run/weights/{name}"})()
        )

    async def save_weights_for_sampler_async(self, name, ttl_seconds):
        self.saved.append(("sampler", name, ttl_seconds))
        return _FakeFuture(
            type("Result", (), {"path": f"tinker://run/sampler_weights/{name}"})()
        )


class _FakeServiceClient:
    def __init__(self):
        self.training_client = _FakeTrainingClient()
        self.loaded = None

    async def create_training_client_from_state_async(self, path, **kwargs):
        self.loaded = (path, kwargs)
        return self.training_client

    async def create_sampling_client_async(self, **kwargs):
        return _FakeSamplingClient()


class _FakeRun:
    id = "unit-test-run"
    url = "https://wandb.example/e4"

    def __init__(self):
        self.logs = []
        self.summary = {}
        self.finished = False

    def log(self, payload, step):
        self.logs.append((payload, step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.run = _FakeRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def _rows(prefix, count):
    return [
        {"question": f"{prefix} question {index}", "answer": "reasoning #### 1"}
        for index in range(count)
    ]


def _manifest():
    return build_manifest(
        _rows("train", 8),
        _rows("test", 2),
        "revision",
        sft_train_count=4,
        sft_validation_count=1,
        rl_train_count=2,
        rl_monitor_count=1,
        calibration_test_count=1,
    )


def _config(**kwargs):
    config = {
        "steps": 2,
        "batch_size": 2,
        "group_size": 4,
        "monitor_examples": 0,
        "checkpoint_every": 1,
        "progress_every": 1,
        "hard_cap_usd": 1.0,
        "train_usd_per_million": 1.0,
        "prefill_usd_per_million": 1.0,
        "sample_usd_per_million": 1.0,
        "max_prompt_tokens": 8,
        "max_output_tokens": 8,
    }
    config.update(kwargs)
    return GRPOConfig(**config)


def test_preflight_has_a_bound_and_never_calls_the_network():
    report = build_doctor_report(
        _config(),
        _manifest(),
        environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
        tinker_version="0.27.0",
        wandb_version="0.21.1",
    )

    assert report["network_called"] is False
    assert report["training_rollouts"] == 16
    assert report["checkpoint_steps"] == [1, 2]
    assert estimate_max_token_cost_usd(_config(), _manifest()) == pytest.approx(
        0.000512
    )
    assert report["ready_for_paid_run"] is True


def test_group_size_below_four_rejects_comparison_breaking_monitoring():
    with pytest.raises(GRPOTrainingError, match="at least four"):
        _config(group_size=3).validate(_manifest())


def test_grpo_training_restores_parent_state_and_masks_prompt_tokens():
    manifest = _manifest()
    service = _FakeServiceClient()
    wandb = _FakeWandb()
    progress = []
    clock_values = iter(range(100))

    report = asyncio.run(
        run_grpo_training(
            _config(),
            allow_paid=True,
            manifest=manifest,
            environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            tinker_module=_FakeTinker,
            wandb_module=wandb,
            service_client=service,
            train_rows=_rows("rl", 2),
            monitor_rows=_rows("monitor", 1),
            clock=lambda: float(next(clock_values)),
            progress=progress.append,
        )
    )

    assert service.loaded[0] == _config().parent_state_path
    assert len(service.training_client.forward_backward_calls) == 2
    assert all(
        loss_fn == "importance_sampling"
        for _, loss_fn in service.training_client.forward_backward_calls
    )
    data = service.training_client.forward_backward_calls[0][0]
    assert data[0].loss_fn_inputs["advantages"][0] == 0.0
    assert any(value != 0.0 for value in data[0].loss_fn_inputs["advantages"][1:])
    assert report["selected_checkpoint"]["step"] == 2
    assert wandb.init_kwargs["group"] == "gsm8k-sft-grpo-v1"
    assert wandb.run.summary["checkpoint/selected_step"] == 2
    assert wandb.run.finished is True
    assert any("step=1/2" in message and "mixed=" in message for message in progress)


def test_grpo_monitor_scores_the_disjoint_holdout_and_can_keep_the_parent():
    manifest = _manifest()
    service = _FakeServiceClient()
    wandb = _FakeWandb()

    report = asyncio.run(
        run_grpo_training(
            _config(monitor_examples=1),
            allow_paid=True,
            manifest=manifest,
            environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            tinker_module=_FakeTinker,
            wandb_module=wandb,
            service_client=service,
            train_rows=_rows("rl", 2),
            monitor_rows=_rows("monitor", 1),
            progress=lambda _: None,
        )
    )

    assert [record["step"] for record in report["checkpoints"]] == [0, 1, 2]
    assert report["selected_checkpoint"]["step"] == 0
    logged = [payload for payload, _ in wandb.run.logs]
    assert any("rl_monitor/pass_at_4" in payload for payload in logged)
