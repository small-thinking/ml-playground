import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.tinker.train_sft import (
    DEFAULT_CONFIG_PATH,
    EvaluationSummary,
    MathExample,
    SFTExperimentError,
    _materialize_batch,
    build_doctor_report,
    estimate_max_token_cost_usd,
    evaluate_sampling_client,
    extract_final_answer,
    load_config,
    load_dataset_candidates,
    normalize_answer,
    override_steps,
    parse_args,
    prepare_data_locally,
    prepare_dataset,
    quality_comparison_is_valid,
    run_sft_experiment,
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
    def __init__(self):
        self.eos_token_id = 2
        self.decoded = {
            201: "Reasoning\nFinal answer: 1",
            202: r"Reasoning\nFinal answer: \boxed{2}",
            299: "Final answer: 0",
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(50, 50 + len(text.split())))

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is True
        assert enable_thinking is False
        tokens = [1]
        for message in messages:
            if message["role"] == "assistant":
                tokens.append(9)
            else:
                tokens.append(3 if message["role"] == "system" else 4)
            tokens.extend(range(20, 20 + len(message["content"].split())))
        if add_generation_prompt:
            tokens.append(9)
        else:
            tokens.append(2)
        return tokens

    def decode(self, tokens):
        return self.decoded[tokens[0]]


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        tokens = super().apply_chat_template(*args, **kwargs)
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shuffle_kwargs = None

    def shuffle(self, **kwargs):
        self.shuffle_kwargs = kwargs
        return self

    def __iter__(self):
        return iter(self.rows)


class FakeFuture:
    def __init__(self, result):
        self.result = result

    async def result_async(self):
        return self.result


class FakeSamplingClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append((prompt, num_samples, sampling_params))
        token = self.outputs[len(self.calls) - 1]
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[token])])


class FakeTrainingClient:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.forward_calls = []
        self.optim_calls = []
        self.saved_state = None
        self.saved_sampler = None

    def get_tokenizer(self):
        return self.tokenizer

    async def forward_backward_async(self, data, loss_fn):
        self.forward_calls.append((data, loss_fn))
        return FakeFuture(SimpleNamespace(metrics={"loss:sum": 4.0}))

    async def optim_step_async(self, params):
        self.optim_calls.append(params)
        return FakeFuture(SimpleNamespace())

    async def save_state_async(self, name, ttl_seconds):
        self.saved_state = (name, ttl_seconds)
        return FakeFuture(SimpleNamespace(path=f"tinker://state/{name}"))

    async def save_weights_for_sampler_async(self, name, ttl_seconds):
        self.saved_sampler = (name, ttl_seconds)
        return FakeFuture(SimpleNamespace(path=f"tinker://sampler/{name}"))


class FakeServiceClient:
    def __init__(self, tokenizer):
        self.base_sampling_client = FakeSamplingClient([201, 299])
        self.final_sampling_client = FakeSamplingClient([201, 202])
        self.training_client = FakeTrainingClient(tokenizer)
        self.base_model = None
        self.model_path = None
        self.training_kwargs = None

    async def create_sampling_client_async(self, **kwargs):
        if "base_model" in kwargs:
            self.base_model = kwargs["base_model"]
            return self.base_sampling_client
        self.model_path = kwargs["model_path"]
        return self.final_sampling_client

    async def create_lora_training_client_async(self, **kwargs):
        self.training_kwargs = kwargs
        return self.training_client


class FakeWandbRun:
    def __init__(self):
        self.id = "unit-test-run"
        self.url = "https://wandb.example/unit-test-run"
        self.logs = []
        self.summary = {}
        self.finished = False

    def log(self, payload, step):
        self.logs.append((payload, step))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = FakeWandbRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def _config(tmp_path: Path, **overrides):
    config = replace(
        load_config(DEFAULT_CONFIG_PATH),
        output_dir=str(tmp_path / "outputs"),
        candidate_examples=6,
        train_examples=3,
        eval_examples=2,
        steps=2,
        batch_size=2,
        max_sequence_tokens=256,
        max_eval_prompt_tokens=128,
        max_eval_output_tokens=8,
    )
    return replace(config, **overrides)


def _rows(count=6):
    return [
        {
            "question": f"Question {index}",
            "final_answer": str(index + 1),
            "difficulty": index + 0.5,
            "topic": "Algebra",
            "r1_solution_1": f"Work for question {index}. Final answer: {index + 1}",
        }
        for index in range(count)
    ]


def _examples(count=6):
    rows = _rows(count)
    return [
        MathExample(
            example_id=f"id-{index}",
            question=row["question"],
            solution=row["r1_solution_1"],
            final_answer=row["final_answer"],
            topic=row["topic"],
            difficulty=row["difficulty"],
        )
        for index, row in enumerate(rows)
    ]


def _configured_environ():
    return {
        "TINKER_API_KEY": "tinker-secret",
        "WANDB_API_KEY": "wandb-secret",
        "HF_TOKEN": "hf-secret",
        "WANDB_PROJECT": "test-project",
    }


def test_default_config_is_pinned_and_steps_are_overridable():
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config.dataset_id == "zwhe99/DeepMath-103K"
    assert len(config.dataset_revision) == 40
    assert config.dataset_license == "MIT"
    assert config.streaming is True
    assert config.steps == 100
    assert config.max_eval_output_tokens == 2048
    assert config.min_eval_completion_rate == pytest.approx(0.8)
    assert override_steps(config, 2).steps == 2
    assert estimate_max_token_cost_usd(config) < config.hard_cap_usd


def test_invalid_iteration_override_is_rejected():
    with pytest.raises(SFTExperimentError, match="steps must be positive"):
        override_steps(load_config(DEFAULT_CONFIG_PATH), 0)


def test_dataset_loader_pins_revision_and_builds_content_ids(tmp_path):
    config = _config(tmp_path)
    fake_dataset = FakeDataset(_rows())
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_dataset

    candidates = load_dataset_candidates(config, load_dataset_fn=fake_load_dataset)

    assert len(candidates) == 6
    assert len({example.example_id for example in candidates}) == 6
    assert captured["args"] == (config.dataset_id,)
    assert captured["kwargs"]["revision"] == config.dataset_revision
    assert captured["kwargs"]["streaming"] is True
    assert fake_dataset.shuffle_kwargs == {
        "seed": config.seed,
        "buffer_size": config.shuffle_buffer,
    }


def test_prepared_data_is_disjoint_and_masks_prompt_tokens(tmp_path):
    config = _config(tmp_path)
    prepared = prepare_dataset(_examples(), FakeTokenizer(), config)

    assert len(prepared.evaluation) == 2
    assert len(prepared.train) == 3
    assert not (
        {item.example_id for item in prepared.evaluation}
        & {item.source.example_id for item in prepared.train}
    )
    for item in prepared.train:
        assert len(item.input_tokens) == len(item.target_tokens) == len(item.weights)
        assert 0.0 in item.weights
        assert item.weights[-1] == 1.0
        assert item.supervised_tokens > 0


def test_chat_template_mapping_result_uses_input_ids(tmp_path):
    prepared = prepare_dataset(_examples(), MappingTokenizer(), _config(tmp_path))

    assert len(prepared.train) == 3


def test_prepare_data_writes_ids_not_raw_examples(tmp_path):
    config = _config(tmp_path)
    dataset = FakeDataset(_rows())

    report = prepare_data_locally(
        config,
        load_dataset_fn=lambda *args, **kwargs: dataset,
        tokenizer=FakeTokenizer(),
    )

    manifest = Path(report.manifest_path).read_text(encoding="utf-8")
    assert report.train_examples == 3
    assert report.eval_examples == 2
    assert "Question 0" not in manifest
    assert "Work for question" not in manifest
    assert "example_id" in manifest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"Reasoning... \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("Reasoning\nFinal answer: 42", "42"),
        ("Reasoning\nAnswer = -3", "-3"),
    ],
)
def test_extract_final_answer(text, expected):
    assert extract_final_answer(text) == expected


def test_normalize_answer_handles_common_latex_variants():
    assert normalize_answer(r"$\dfrac{1}{2}$.") == normalize_answer(
        r"\boxed{\frac{1}{2}}"
    )


def test_doctor_is_local_only_and_never_exposes_keys(tmp_path):
    report = build_doctor_report(
        _config(tmp_path),
        environ=_configured_environ(),
        tinker_version="0.23.2",
        wandb_version="0.21.1",
    )

    assert report.network_called is False
    assert report.ready_for_paid_run is True
    assert "tinker-secret" not in str(report)
    assert "wandb-secret" not in str(report)
    assert "hf-secret" not in str(report)


def test_paid_gate_blocks_before_dataset_or_clients(tmp_path):
    service_client = FakeServiceClient(FakeTokenizer())
    load_called = False

    def fake_load_dataset(*args, **kwargs):
        nonlocal load_called
        load_called = True
        return FakeDataset(_rows())

    with pytest.raises(SFTExperimentError, match="--allow-paid"):
        asyncio.run(
            run_sft_experiment(
                _config(tmp_path),
                allow_paid=False,
                environ=_configured_environ(),
                tinker_module=FAKE_TINKER,
                wandb_module=FakeWandb(),
                service_client=service_client,
                load_dataset_fn=fake_load_dataset,
            )
        )

    assert load_called is False
    assert service_client.base_model is None


def test_paid_gate_blocks_step_override_above_cost_cap(tmp_path):
    service_client = FakeServiceClient(FakeTokenizer())
    config = _config(tmp_path, steps=10_000, hard_cap_usd=0.01)

    with pytest.raises(SFTExperimentError, match="hard cap"):
        asyncio.run(
            run_sft_experiment(
                config,
                allow_paid=True,
                environ=_configured_environ(),
                tinker_module=FAKE_TINKER,
                wandb_module=FakeWandb(),
                service_client=service_client,
                candidates=_examples(),
            )
        )

    assert service_client.base_model is None


def test_materialized_batch_matches_installed_tinker_contract(tmp_path):
    tinker = pytest.importorskip("tinker")
    prepared = prepare_dataset(_examples(), FakeTokenizer(), _config(tmp_path))

    data, input_tokens, supervised_tokens = _materialize_batch(
        prepared.train[:2], tinker
    )

    assert len(data) == 2
    assert input_tokens > supervised_tokens > 0
    assert data[0].loss_fn_inputs["weights"].dtype == "float32"
    assert data[0].loss_fn_inputs["target_tokens"].dtype == "int64"


def test_full_fake_run_trains_requested_steps_saves_and_evaluates(tmp_path):
    config = _config(tmp_path)
    service_client = FakeServiceClient(FakeTokenizer())
    fake_wandb = FakeWandb()
    times = iter([0.0, 0.1, 1.0, 1.2])
    progress = []

    report = asyncio.run(
        run_sft_experiment(
            config,
            allow_paid=True,
            environ=_configured_environ(),
            tinker_module=FAKE_TINKER,
            wandb_module=fake_wandb,
            service_client=service_client,
            candidates=_examples(),
            clock=lambda: next(times),
            progress=progress.append,
        )
    )

    training_client = service_client.training_client
    assert len(training_client.forward_calls) == 2
    assert len(training_client.optim_calls) == 2
    assert all(call[1] == "cross_entropy" for call in training_client.forward_calls)
    assert all(
        param.learning_rate == config.learning_rate
        for param in training_client.optim_calls
    )
    assert training_client.saved_state[1] == config.checkpoint_ttl_seconds
    assert training_client.saved_sampler[1] == config.checkpoint_ttl_seconds
    assert service_client.model_path == report.sampler_path

    assert report.steps_completed == 2
    assert report.baseline_accuracy == pytest.approx(0.5)
    assert report.final_accuracy == pytest.approx(1.0)
    assert report.accuracy_gain == pytest.approx(0.5)
    assert report.quality_comparison_valid is True
    assert Path(report.manifest_path).exists()
    assert Path(report.report_path).exists()
    payload = json.loads(Path(report.report_path).read_text(encoding="utf-8"))
    assert payload["summary"]["checkpoint_path"] == report.checkpoint_path
    assert fake_wandb.run.summary["eval/accuracy_gain"] == pytest.approx(0.5)
    assert fake_wandb.run.summary["eval/quality_comparison_valid"] is True
    assert fake_wandb.run.finished is True
    assert any("step=1/2" in message for message in progress)
    assert progress[-1].startswith("complete baseline=")


def test_iterations_alias_is_accepted():
    args = parse_args(["--iterations", "7"])

    assert args.steps == 7


def test_truncated_evaluation_invalidates_quality_comparison(tmp_path):
    config = replace(_config(tmp_path), max_eval_output_tokens=1)
    summary = asyncio.run(
        evaluate_sampling_client(
            FakeSamplingClient([201]),
            FakeTokenizer(),
            FAKE_TINKER,
            _examples(1),
            config,
        )
    )

    assert summary.accuracy == pytest.approx(1.0)
    assert summary.completion_rate == pytest.approx(0.0)
    assert summary.truncation_rate == pytest.approx(1.0)
    assert summary.score_completed == pytest.approx(0.0)
    assert quality_comparison_is_valid(summary, summary, 0.8) is False


def test_quality_comparison_requires_both_completion_rates():
    complete = EvaluationSummary(
        accuracy=0.5,
        score_completed=0.5,
        parse_rate=1.0,
        completion_rate=1.0,
        truncation_rate=0.0,
        prompt_tokens=10,
        output_tokens=10,
        observations=(),
    )
    incomplete = replace(complete, completion_rate=0.75, truncation_rate=0.25)

    assert quality_comparison_is_valid(complete, complete, 0.8) is True
    assert quality_comparison_is_valid(complete, incomplete, 0.8) is False
