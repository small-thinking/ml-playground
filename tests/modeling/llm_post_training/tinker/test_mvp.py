import asyncio
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.tinker.mvp import (
    SMOKE_PROMPT,
    SmokeConfig,
    SmokeTestError,
    build_doctor_report,
    estimate_token_cost_usd,
    run_remote_sample,
)


class FakeTokenizer:
    def encode(self, text):
        assert text == SMOKE_PROMPT
        return [10, 11, 12]

    def decode(self, tokens):
        assert tokens == [391]
        return "391"


class FakeSamplingClient:
    def __init__(self):
        self.calls = 0

    def get_tokenizer(self):
        return FakeTokenizer()

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls += 1
        assert prompt == (10, 11, 12)
        assert num_samples == 1
        assert sampling_params.max_tokens == 64
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[391])])


class FakeServiceClient:
    def __init__(self, sampling_client):
        self.sampling_client = sampling_client
        self.base_model = None

    def create_sampling_client(self, base_model):
        self.base_model = base_model
        return self.sampling_client


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
)


def test_cost_estimate_uses_prefill_and_sample_rates():
    config = SmokeConfig()

    cost = estimate_token_cost_usd(512, 64, config)

    assert cost == pytest.approx(0.00023328)
    assert cost < config.hard_cap_usd


def test_doctor_is_local_only_and_does_not_expose_key():
    config = SmokeConfig()

    report = build_doctor_report(
        config,
        environ={"TINKER_API_KEY": "secret-value"},
        sdk_version="0.23.2",
    )

    assert report.network_called is False
    assert report.api_key_configured is True
    assert "secret-value" not in str(report)
    assert report.ready_for_remote_sample is True


def test_remote_sample_is_blocked_without_explicit_paid_flag():
    sampling_client = FakeSamplingClient()

    with pytest.raises(SmokeTestError, match="--allow-paid"):
        asyncio.run(
            run_remote_sample(
                SmokeConfig(),
                allow_paid=False,
                environ={"TINKER_API_KEY": "secret-value"},
                tinker_module=FAKE_TINKER,
                service_client=FakeServiceClient(sampling_client),
            )
        )

    assert sampling_client.calls == 0


def test_remote_sample_is_blocked_without_api_key():
    sampling_client = FakeSamplingClient()

    with pytest.raises(SmokeTestError, match="TINKER_API_KEY"):
        asyncio.run(
            run_remote_sample(
                SmokeConfig(),
                allow_paid=True,
                environ={},
                tinker_module=FAKE_TINKER,
                service_client=FakeServiceClient(sampling_client),
            )
        )

    assert sampling_client.calls == 0


def test_remote_sample_is_blocked_when_cap_is_too_low():
    sampling_client = FakeSamplingClient()
    config = SmokeConfig(hard_cap_usd=0.0001)

    with pytest.raises(SmokeTestError, match="hard cap"):
        asyncio.run(
            run_remote_sample(
                config,
                allow_paid=True,
                environ={"TINKER_API_KEY": "secret-value"},
                tinker_module=FAKE_TINKER,
                service_client=FakeServiceClient(sampling_client),
            )
        )

    assert sampling_client.calls == 0


def test_remote_sample_uses_exactly_one_bounded_request():
    sampling_client = FakeSamplingClient()
    service_client = FakeServiceClient(sampling_client)

    report = asyncio.run(
        run_remote_sample(
            SmokeConfig(),
            allow_paid=True,
            environ={"TINKER_API_KEY": "secret-value"},
            tinker_module=FAKE_TINKER,
            service_client=service_client,
        )
    )

    assert sampling_client.calls == 1
    assert service_client.base_model == "Qwen/Qwen3.5-4B"
    assert report.prompt_tokens == 3
    assert report.output_tokens == 1
    assert report.response_text == "391"
    assert report.estimated_token_cost_usd < report.hard_cap_usd
