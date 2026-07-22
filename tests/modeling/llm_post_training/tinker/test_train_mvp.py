import asyncio
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.tinker.train_mvp import (
    SFT_SMOKE_EXAMPLES,
    TrainingConfig,
    TrainingMVPError,
    build_doctor_report,
    estimate_actual_token_cost_usd,
    estimate_max_token_cost_usd,
    prepare_training_batch,
    run_training_mvp,
)


class FakeTensorData:
    def __init__(self, data):
        self.data = list(data)


class FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = {
            key: FakeTensorData(value) for key, value in loss_fn_inputs.items()
        }


class FakeAdamParams:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class FakeModelInput:
    @staticmethod
    def from_ints(tokens):
        return tuple(tokens)


class FakeSamplingParams:
    def __init__(self, max_tokens, temperature, seed):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed


FAKE_TINKER = SimpleNamespace(
    ModelInput=FakeModelInput,
    SamplingParams=FakeSamplingParams,
    types=SimpleNamespace(Datum=FakeDatum, AdamParams=FakeAdamParams),
)


class FakeTokenizer:
    def encode(self, text):
        return list(range(1, len(text.split()) + 2))

    def decode(self, tokens):
        return f"answer-{tokens[0]}"


class FakeFuture:
    def __init__(self, result):
        self.result = result

    async def result_async(self):
        return self.result


class FakeSamplingClient:
    def __init__(self, output_token):
        self.output_token = output_token
        self.calls = 0

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls += 1
        assert prompt
        assert num_samples == 1
        assert sampling_params.max_tokens == 32
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[self.output_token])])


class FakeTrainingClient:
    def __init__(self, tokenizer, trained_sampling_client):
        self.tokenizer = tokenizer
        self.trained_sampling_client = trained_sampling_client
        self.forward_calls = []
        self.optim_calls = []
        self.saved_name = None

    def get_tokenizer(self):
        return self.tokenizer

    async def forward_backward_async(self, data, loss_fn):
        self.forward_calls.append((data, loss_fn))
        result = SimpleNamespace(
            metrics={"loss:sum": float(4 - len(self.forward_calls))},
            loss_fn_outputs=[],
        )
        return FakeFuture(result)

    async def optim_step_async(self, adam_params):
        self.optim_calls.append(adam_params)
        return FakeFuture(SimpleNamespace())

    def save_weights_and_get_sampling_client(self):
        self.saved_name = "ephemeral"
        return self.trained_sampling_client


class FakeServiceClient:
    def __init__(self):
        self.base_sampling_client = FakeSamplingClient(10)
        self.trained_sampling_client = FakeSamplingClient(20)
        self.training_client = FakeTrainingClient(
            FakeTokenizer(), self.trained_sampling_client
        )
        self.base_model = None
        self.training_kwargs = None

    async def create_sampling_client_async(self, base_model):
        self.base_model = base_model
        return self.base_sampling_client

    async def create_lora_training_client_async(self, **kwargs):
        self.training_kwargs = kwargs
        return self.training_client


class FakeWandbRun:
    def __init__(self):
        self.url = "https://wandb.example/run/mvp"
        self.logs = []
        self.summary = {}
        self.finished = False

    def log(self, metrics, step):
        self.logs.append((metrics, step))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.init_kwargs = None
        self.run = FakeWandbRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def configured_environ():
    return {
        "TINKER_API_KEY": "tinker-secret",
        "WANDB_API_KEY": "wandb-secret",
        "WANDB_PROJECT": "test-project",
    }


def test_max_cost_is_below_hard_cap():
    config = TrainingConfig()

    assert estimate_max_token_cost_usd(config) == pytest.approx(0.000714816)
    assert estimate_max_token_cost_usd(config) < config.hard_cap_usd


def test_actual_cost_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        estimate_actual_token_cost_usd(TrainingConfig(), -1, 1, 1)


def test_doctor_is_local_only_and_does_not_expose_keys():
    report = build_doctor_report(
        TrainingConfig(),
        environ=configured_environ(),
        tinker_version="0.23.2",
        wandb_version="0.21.1",
    )

    assert report.network_called is False
    assert report.ready_for_paid_run is True
    assert "tinker-secret" not in str(report)
    assert "wandb-secret" not in str(report)


def test_training_is_blocked_before_clients_without_paid_flag():
    service_client = FakeServiceClient()
    fake_wandb = FakeWandb()

    with pytest.raises(TrainingMVPError, match="--allow-paid"):
        asyncio.run(
            run_training_mvp(
                TrainingConfig(),
                allow_paid=False,
                environ=configured_environ(),
                tinker_module=FAKE_TINKER,
                wandb_module=fake_wandb,
                service_client=service_client,
            )
        )

    assert service_client.base_model is None
    assert fake_wandb.init_kwargs is None


@pytest.mark.parametrize("missing_key", ["TINKER_API_KEY", "WANDB_API_KEY"])
def test_training_is_blocked_when_a_required_key_is_missing(missing_key):
    environ = configured_environ()
    del environ[missing_key]
    service_client = FakeServiceClient()

    with pytest.raises(TrainingMVPError, match=missing_key):
        asyncio.run(
            run_training_mvp(
                TrainingConfig(),
                allow_paid=True,
                environ=environ,
                tinker_module=FAKE_TINKER,
                wandb_module=FakeWandb(),
                service_client=service_client,
            )
        )

    assert service_client.base_model is None


def test_training_is_blocked_when_cap_is_too_low():
    service_client = FakeServiceClient()

    with pytest.raises(TrainingMVPError, match="hard cap"):
        asyncio.run(
            run_training_mvp(
                TrainingConfig(hard_cap_usd=0.0001),
                allow_paid=True,
                environ=configured_environ(),
                tinker_module=FAKE_TINKER,
                wandb_module=FakeWandb(),
                service_client=service_client,
            )
        )

    assert service_client.base_model is None


def test_batch_masks_prompts_and_trains_on_completions():
    batch = prepare_training_batch(FakeTokenizer(), FAKE_TINKER, TrainingConfig())

    assert len(batch.data) == len(SFT_SMOKE_EXAMPLES)
    assert batch.input_tokens > batch.supervised_tokens > 0
    for datum in batch.data:
        weights = datum.loss_fn_inputs["weights"].data
        targets = datum.loss_fn_inputs["target_tokens"].data
        assert len(datum.model_input) == len(weights) == len(targets)
        assert 0.0 in weights
        assert weights[-1] == 1.0


def test_batch_matches_installed_tinker_datum_contract():
    tinker = pytest.importorskip("tinker")

    batch = prepare_training_batch(FakeTokenizer(), tinker, TrainingConfig())

    assert len(batch.data) == 2
    assert batch.data[0].loss_fn_inputs["weights"].dtype == "float32"
    assert batch.data[0].loss_fn_inputs["target_tokens"].dtype == "int64"


def test_remote_mvp_runs_three_updates_and_logs_basic_metrics():
    service_client = FakeServiceClient()
    fake_wandb = FakeWandb()
    times = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])

    report = asyncio.run(
        run_training_mvp(
            TrainingConfig(),
            allow_paid=True,
            environ=configured_environ(),
            tinker_module=FAKE_TINKER,
            wandb_module=fake_wandb,
            service_client=service_client,
            clock=lambda: next(times),
        )
    )

    training_client = service_client.training_client
    assert len(training_client.forward_calls) == 3
    assert len(training_client.optim_calls) == 3
    assert all(call[1] == "cross_entropy" for call in training_client.forward_calls)
    assert all(param.learning_rate == 1e-4 for param in training_client.optim_calls)
    assert training_client.saved_name == "ephemeral"
    assert service_client.base_sampling_client.calls == 1
    assert service_client.trained_sampling_client.calls == 1

    assert len(fake_wandb.run.logs) == 3
    for expected_step, (metrics, logged_step) in enumerate(
        fake_wandb.run.logs, start=1
    ):
        assert logged_step == expected_step
        assert {
            "train/loss",
            "tokens/cumulative_train",
            "cost/estimated_cumulative_train_usd",
            "timing/step_seconds",
        } <= metrics.keys()
    assert fake_wandb.run.summary["sample/before_text"] == "answer-10"
    assert fake_wandb.run.summary["sample/after_text"] == "answer-20"
    assert fake_wandb.run.finished is True

    assert report.steps_completed == 3
    assert report.wandb_project == "test-project"
    assert report.before_response_text == "answer-10"
    assert report.after_response_text == "answer-20"
    assert report.estimated_token_cost_usd < report.hard_cap_usd
    assert "tinker-secret" not in str(report)
    assert "wandb-secret" not in str(report)


def test_default_service_client_uses_standard_httpx_transport(monkeypatch):
    import httpx

    service_client = FakeServiceClient()
    captured = {}
    fake_http_client = object()

    def build_http_client(**kwargs):
        captured["httpx_kwargs"] = kwargs
        return fake_http_client

    def build_service_client(**kwargs):
        captured["service_kwargs"] = kwargs
        return service_client

    fake_tinker = SimpleNamespace(
        ModelInput=FakeModelInput,
        SamplingParams=FakeSamplingParams,
        types=FAKE_TINKER.types,
        ServiceClient=build_service_client,
    )
    monkeypatch.setattr(httpx, "AsyncClient", build_http_client)
    times = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])

    asyncio.run(
        run_training_mvp(
            TrainingConfig(),
            allow_paid=True,
            environ=configured_environ(),
            tinker_module=fake_tinker,
            wandb_module=FakeWandb(),
            clock=lambda: next(times),
        )
    )

    assert captured["httpx_kwargs"] == {"follow_redirects": True}
    assert captured["service_kwargs"]["http_client"] is fake_http_client
