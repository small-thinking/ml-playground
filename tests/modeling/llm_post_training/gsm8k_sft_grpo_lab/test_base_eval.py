import asyncio

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval import (
    BaseEvalConfig,
    BaseEvalError,
    build_doctor_report,
    evaluation_protocol_id,
    estimate_max_token_cost_usd,
    parse_args,
    run_remote_evaluation,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import build_manifest


class _FakeModelInput:
    @staticmethod
    def from_ints(tokens):
        return tuple(tokens)


class _FakeTinker:
    ModelInput = _FakeModelInput

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class _FakeTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))

    def decode(self, tokens):
        return tokens[0]


class _FakeSamplingClient:
    def get_tokenizer(self):
        return _FakeTokenizer()

    async def sample_async(self, prompt, num_samples, sampling_params):
        assert len(prompt) > 0
        assert num_samples == 4
        assert sampling_params.kwargs["max_tokens"] == 512
        responses = (r"\boxed{4}", r"\boxed{5}", r"\boxed{5}", r"\boxed{4}")
        return type(
            "Result",
            (),
            {
                "sequences": tuple(
                    type("Sequence", (), {"tokens": (text,)}) for text in responses
                )
            },
        )()


class _FakeServiceClient:
    async def create_sampling_client_async(self, base_model):
        assert base_model == "Qwen/Qwen3.5-9B-Base"
        return _FakeSamplingClient()


class _FakeRun:
    def __init__(self):
        self.url = "https://wandb.ai/example/mini-posttraining-lab/runs/e0a"
        self.logs = []
        self.summary = {}
        self.finished = False

    def log(self, payload):
        self.logs.append(payload)

    def finish(self):
        self.finished = True


class _FakeWandb:
    class Table:
        def __init__(self, columns, data):
            assert isinstance(columns, list)
            assert isinstance(data, list)
            self.columns = columns
            self.data = data

    def __init__(self):
        self.init_kwargs = None
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def _manifest_and_rows():
    train_rows = [
        {"question": "train one", "answer": "work #### 1"},
        {"question": "train two", "answer": "work #### 2"},
    ]
    rows = [{"question": "What is 2 + 2?", "answer": "work #### 4"}]
    return (
        build_manifest(
            train_rows, rows, "revision", sft_count=1, rl_count=1, eval_count=1
        ),
        rows,
    )


def test_preflight_reports_a_bounded_e0a_cost_without_network():
    config = BaseEvalConfig()
    report = build_doctor_report(
        config,
        environ={"HF_TOKEN": "set", "TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
        tinker_version="0.23.2",
        wandb_version="0.21.1",
    )

    assert estimate_max_token_cost_usd(config) == pytest.approx(0.17399808)
    assert report["network_called"] is False
    assert report["hf_token_configured"] is True
    assert report["ready_for_paid_run"] is True


def test_protocol_id_changes_when_a_comparability_condition_changes():
    manifest, _ = _manifest_and_rows()

    assert evaluation_protocol_id(
        BaseEvalConfig(eval_examples=1), manifest
    ) != evaluation_protocol_id(
        BaseEvalConfig(eval_examples=1, temperature=0.7), manifest
    )
    assert evaluation_protocol_id(
        BaseEvalConfig(eval_examples=1), manifest
    ) != evaluation_protocol_id(
        BaseEvalConfig(eval_examples=1, max_output_tokens=256), manifest
    )


def test_run_name_records_an_explicit_retry():
    assert BaseEvalConfig(attempt=2).run_name.endswith("-g4-a02")


def test_cli_accepts_an_explicit_cost_cap():
    args = parse_args(
        ["--run", "--allow-paid", "--attempt", "3", "--hard-cap-usd", "1"]
    )

    assert args.attempt == 3
    assert args.hard_cap_usd == pytest.approx(1.0)


def test_remote_evaluation_logs_metrics_and_raw_rollout_table():
    manifest, rows = _manifest_and_rows()
    wandb = _FakeWandb()
    config = BaseEvalConfig(eval_examples=1)

    report = asyncio.run(
        run_remote_evaluation(
            config,
            manifest,
            rows,
            allow_paid=True,
            environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            tinker_module=_FakeTinker,
            wandb_module=wandb,
            service_client=_FakeServiceClient(),
        )
    )

    assert report["generated_rollouts"] == 4
    assert report["metrics"]["eval/exact_match"] == pytest.approx(0.5)
    assert wandb.init_kwargs["project"] == "mini-posttraining-lab"
    assert wandb.init_kwargs["group"] == "gsm8k-sft-grpo-v1"
    table = wandb.run.logs[1]["eval/rollouts_table"]
    assert "advantage" in table.columns
    assert len(table.data) == 4
    assert wandb.run.finished is True


def test_remote_evaluation_requires_explicit_paid_authorization():
    manifest, rows = _manifest_and_rows()

    with pytest.raises(BaseEvalError, match="blocked"):
        asyncio.run(
            run_remote_evaluation(
                BaseEvalConfig(eval_examples=1),
                manifest,
                rows,
                allow_paid=False,
                environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            )
        )
