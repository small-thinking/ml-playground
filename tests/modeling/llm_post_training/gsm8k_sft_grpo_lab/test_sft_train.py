import asyncio

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import build_manifest
from modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train import (
    SFTConfig,
    SFTTrainingError,
    build_doctor_report,
    build_sft_completion,
    estimate_max_token_cost_usd,
    run_sft_training,
    tokenize_sft_example,
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


class _FakeTinker:
    ModelInput = _FakeModelInput

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
        return list(range(max(1, len(text.split()))))


def _loss_for(data, nll):
    supervised = sum(sum(item.loss_fn_inputs["weights"]) for item in data)
    return type("Result", (), {"metrics": {"loss:sum": supervised * nll}})()


class _FakeTrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
        self.forward_calls = []
        self.optim_calls = []
        self.saved = []

    def get_tokenizer(self):
        return _FakeTokenizer()

    async def forward_backward_async(self, data, loss_fn):
        self.forward_backward_calls.append((data, loss_fn))
        return _FakeFuture(_loss_for(data, 1.2))

    async def optim_step_async(self, params):
        self.optim_calls.append(params)
        return _FakeFuture(object())

    async def forward_async(self, data, loss_fn):
        self.forward_calls.append((data, loss_fn))
        nll = 2.0 - 0.5 * ((len(self.forward_calls) - 1) // 1)
        return _FakeFuture(_loss_for(data, nll))

    async def save_state_async(self, name, ttl_seconds):
        self.saved.append(("state", name, ttl_seconds))
        return _FakeFuture(type("Result", (), {"path": f"tinker://state/{name}"})())

    async def save_weights_for_sampler_async(self, name, ttl_seconds):
        self.saved.append(("sampler", name, ttl_seconds))
        return _FakeFuture(type("Result", (), {"path": f"tinker://sampler/{name}"})())


class _FakeServiceClient:
    def __init__(self):
        self.training_client = _FakeTrainingClient()
        self.kwargs = None

    async def create_lora_training_client_async(self, **kwargs):
        self.kwargs = kwargs
        return self.training_client


class _FakeRun:
    id = "unit-test-run"
    url = "https://wandb.example/e1"

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
        {"question": f"{prefix} question {index}", "answer": f"reasoning #### {index}"}
        for index in range(count)
    ]


def _manifest():
    return build_manifest(
        _rows("train", 8),
        _rows("test", 2),
        "revision",
        sft_train_count=4,
        sft_validation_count=2,
        rl_train_count=1,
        rl_monitor_count=1,
        calibration_test_count=1,
    )


def _config():
    return SFTConfig(
        batch_size=2,
        validation_every=1,
        progress_every=1,
        max_sequence_tokens=64,
        hard_cap_usd=1.0,
        train_usd_per_million=1.0,
    )


def test_sft_completion_converts_the_gsm8k_marker_to_boxed_format():
    assert build_sft_completion("work it out #### 1,200") == "work it out\n\\boxed{1,200}"
    with pytest.raises(SFTTrainingError, match="missing"):
        build_sft_completion("work it out")


def test_tokenization_masks_the_prompt_and_supervises_the_completion():
    example = tokenize_sft_example(
        {"question": "How many?", "answer": "Two plus two is four. #### 4"},
        _FakeTokenizer(),
        max_sequence_tokens=64,
    )

    assert example is not None
    assert any(weight == 0.0 for weight in example.weights)
    assert example.supervised_tokens == sum(example.weights)


def test_preflight_has_a_bounded_cost_and_explicit_validation_schedule():
    manifest = _manifest()
    report = build_doctor_report(
        _config(),
        manifest,
        environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
        tinker_version="0.27.0",
        wandb_version="0.21.1",
    )

    assert report["network_called"] is False
    assert report["training_steps"] == 2
    assert report["validation_steps"] == [1, 2]
    assert estimate_max_token_cost_usd(_config(), manifest) == pytest.approx(0.00064)
    assert report["ready_for_paid_run"] is True


def test_sft_training_logs_train_and_validation_metrics_and_selects_checkpoint():
    manifest = _manifest()
    train_rows = _rows("train", 4)
    validation_rows = _rows("validation", 2)
    wandb = _FakeWandb()
    service = _FakeServiceClient()
    progress = []
    clock_values = iter(range(100))

    report = asyncio.run(
        run_sft_training(
            _config(),
            allow_paid=True,
            manifest=manifest,
            environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            tinker_module=_FakeTinker,
            wandb_module=wandb,
            service_client=service,
            train_rows=train_rows,
            validation_rows=validation_rows,
            clock=lambda: float(next(clock_values)),
            progress=progress.append,
        )
    )

    assert report["training_steps"] == 2
    assert report["selected_checkpoint"]["step"] == 2
    assert len(report["validation_checkpoints"]) == 2
    assert service.kwargs["base_model"] == "Qwen/Qwen3.5-9B-Base"
    assert len(service.training_client.forward_backward_calls) == 2
    assert len(service.training_client.optim_calls) == 2
    assert wandb.init_kwargs["group"] == "gsm8k-sft-grpo-v1"
    logged = [payload for payload, _ in wandb.run.logs]
    assert any("train/nll" in payload and "train/perplexity" in payload for payload in logged)
    assert any("sft_validation/nll" in payload for payload in logged)
    assert wandb.run.summary["checkpoint/selected_step"] == 2
    assert wandb.run.finished is True
    assert any("step=1/2" in message and "eta=" in message for message in progress)
    assert any("validation step=2/2" in message for message in progress)
