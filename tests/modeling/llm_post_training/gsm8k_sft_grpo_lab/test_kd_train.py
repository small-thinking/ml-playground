import asyncio

import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    build_manifest,
    content_id,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.kd_train import (
    KDConfig,
    KDTrainingError,
    TEACHER_SCORE,
    _student_training_cost_ledger,
    _teacher_cost_ledger,
    build_doctor_report,
    estimate_max_token_cost_usd,
    run_kd_training,
    tokenize_teacher_response,
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
    eos_token_id = None

    def encode(self, text, **kwargs):
        return list(range(max(1, len(text.split()))))

    def decode(self, tokens):
        return f"\\boxed{{{tokens[0]}}}"


class _FakeSamplingClient:
    def get_tokenizer(self):
        return _FakeTokenizer()

    async def sample_async(self, prompt, num_samples, sampling_params):
        sequence = type("Sequence", (), {"tokens": [1]})()
        return type("Result", (), {"sequences": [sequence] * num_samples})()


def _loss_for(data, nll):
    supervised = sum(sum(item.loss_fn_inputs["weights"]) for item in data)
    return type("Result", (), {"metrics": {"loss:sum": supervised * nll}})()


class _FakeTrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
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
        self.created_lora = None
        self.base_model_requests = []
        self.monitor_paths = []

    async def create_lora_training_client_async(self, base_model, rank, seed, **kwargs):
        self.created_lora = (base_model, rank, seed, kwargs)
        return self.training_client

    async def create_sampling_client_async(self, **kwargs):
        if "base_model" in kwargs:
            self.base_model_requests.append(kwargs["base_model"])
        if "model_path" in kwargs:
            self.monitor_paths.append(kwargs["model_path"])
        return _FakeSamplingClient()


class _FakeRun:
    id = "unit-test-run"
    url = "https://wandb.example/e9"

    def __init__(self):
        self.logs = []
        self.defined_metrics = []
        self.summary = {}
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


def _full_candidate_rows(manifest):
    rows_by_id = {
        row_id: row for row in _rows("train", 8) for row_id in [content_id(row)]
    }
    return [
        rows_by_id[row_id] for row_id in manifest.sft_train_ids + manifest.rl_train_ids
    ]


def _development_rows(manifest):
    rows_by_id = {
        row_id: row for row in _rows("train", 8) for row_id in [content_id(row)]
    }
    return [rows_by_id[row_id] for row_id in manifest.sft_validation_ids]


def _config(tmp_path, **kwargs):
    payload = {
        "teacher_batch_size": 1,
        "batch_size": 2,
        "max_student_steps": 3,
        "development_examples": 1,
        "development_input_token_interval": 999,
        "progress_every": 1,
        "max_sequence_tokens": 64,
        "hard_cap_usd": 1.0,
        "teacher_prefill_usd_per_million": 1.0,
        "teacher_sample_usd_per_million": 1.0,
        "train_usd_per_million": 1.0,
        "trace_output_dir": str(tmp_path),
    }
    payload.update(kwargs)
    return KDConfig(**payload)


def test_teacher_response_masks_the_prompt_and_supervises_only_the_trace(tmp_path):
    example = tokenize_teacher_response(
        {"question": "How many?", "answer": "reasoning #### 1"},
        "Two minus one is one. \\boxed{1}",
        _FakeTokenizer(),
        _config(tmp_path),
    )

    assert example is not None
    assert any(weight == 0.0 for weight in example.weights)
    assert example.supervised_tokens == sum(example.weights)


def test_actual_token_cost_ledgers_apply_the_configured_rates(tmp_path):
    config = _config(
        tmp_path,
        teacher_prefill_usd_per_million=1.0,
        teacher_sample_usd_per_million=2.0,
        train_usd_per_million=3.0,
    )

    teacher = _teacher_cost_ledger(1_000_000, 2_000_000, config)
    student = _student_training_cost_ledger(1_000_000, 17, config)

    assert teacher.input_usd == 1.0
    assert teacher.output_usd == 4.0
    assert teacher.total_usd == 5.0
    assert student.supervised_target_tokens == 17
    assert student.total_usd == 3.0


def test_scalar_teacher_score_is_not_silently_routed_through_hard_kd(tmp_path):
    with pytest.raises(KDTrainingError, match="RLAIF"):
        _config(tmp_path, signal_kind=TEACHER_SCORE).validate(_manifest())


def test_preflight_records_base_provenance_and_separate_teacher_cost(tmp_path):
    config = _config(tmp_path)
    report = build_doctor_report(
        config,
        _manifest(),
        environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
        tinker_version="0.27.0",
        wandb_version="0.21.1",
    )

    assert report["network_called"] is False
    assert report["initialization_source"] == "base_fresh_lora"
    assert report["parent_checkpoint"] is None
    assert report["reference_e4_checkpoint"].endswith("step75")
    assert report["teacher_candidate_count"] == 6
    assert report["student_training_data_policy"] == "all_accepted_teacher_traces_once"
    assert report["estimated_cost_breakdown_usd"]["teacher_generation_max_usd"] > 0
    assert estimate_max_token_cost_usd(config, _manifest()) < config.hard_cap_usd
    assert report["ready_for_paid_run"] is True


def test_kd_starts_a_fresh_base_lora_and_records_token_provenance(tmp_path):
    manifest = _manifest()
    config = _config(tmp_path)
    service = _FakeServiceClient()
    wandb = _FakeWandb()
    progress_messages = []

    report = asyncio.run(
        run_kd_training(
            config,
            allow_paid=True,
            manifest=manifest,
            environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
            tinker_module=_FakeTinker,
            wandb_module=wandb,
            service_client=service,
            train_rows=_full_candidate_rows(manifest),
            development_rows=_development_rows(manifest),
            progress=progress_messages.append,
        )
    )

    assert service.created_lora[:3] == (
        config.model_id,
        config.lora_rank,
        config.seed,
    )
    assert "Qwen/Qwen3.5-397B-A17B" in service.base_model_requests
    assert config.model_id in service.base_model_requests
    assert len(service.training_client.forward_backward_calls) == 3
    data = service.training_client.forward_backward_calls[0][0]
    assert all(
        loss_fn == "cross_entropy"
        for _, loss_fn in service.training_client.forward_backward_calls
    )
    assert any(weight == 0.0 for weight in data[0].loss_fn_inputs["weights"])
    assert report["teacher_outcomes"]["teacher_correct"] == 6
    assert (
        report["student_optimized_input_tokens"]
        == report["student_selected_input_tokens"]
    )
    assert report["selected_checkpoint"]["step"] == 0
    assert report["selected_checkpoint_is_initialization"] is True
    assert report["distillation_schema_version"] == "gsm8k-distillation-schema-v3"
    actual_cost = report["actual_token_priced_cost_ledger"]
    assert actual_cost["teacher_generation"]["input_tokens"] > 0
    assert actual_cost["teacher_generation"]["output_tokens"] > 0
    assert (
        actual_cost["student_training"]["optimized_input_tokens"]
        == report["student_optimized_input_tokens"]
    )
    assert actual_cost["development_inference"]["input_tokens"] > 0
    assert actual_cost["development_inference"]["output_tokens"] > 0
    assert actual_cost["total_usd"] == pytest.approx(
        report["actual_token_priced_total_usd"]
    )
    assert (tmp_path / "e9_teacher_traces_unit-test-run.jsonl").exists()
    assert (tmp_path / "e9_kd_report_unit-test-run.json").exists()
    assert (tmp_path / "e9_metric_dictionary_unit-test-run.md").exists()
    assert (
        "dev/*",
        {"step_metric": "dev/optimized_input_tokens"},
    ) in wandb.run.defined_metrics
    assert wandb.run.summary["selection/selected_is_initialization"] == 1.0
    assert wandb.run.summary["dev/pass_at_1"] == 1.0
    assert wandb.run.summary["dev/pass_at_4"] == 1.0
    assert wandb.run.summary["selection/selected_dev_pass_at_1"] == 1.0
    assert wandb.run.summary["selection/selected_dev_pass_at_4"] == 1.0
    assert wandb.run.summary["cost/teacher_input_usd"] > 0
    assert wandb.run.summary["cost/teacher_output_usd"] > 0
    assert wandb.run.summary["cost/dev_input_usd"] > 0
    assert wandb.run.summary["cost/dev_output_usd"] > 0
    development_tables = [
        payload["tables/development_rollouts"]
        for payload, _ in wandb.run.logs
        if "tables/development_rollouts" in payload
    ]
    assert len(development_tables) == 2
    assert len(development_tables[0].data) == 4
    assert wandb.init_kwargs["config"]["initialization_source"] == "base_fresh_lora"
    assert wandb.run.finished is True
    assert any(
        message.startswith("cost[teacher actual]") for message in progress_messages
    )
    assert any(
        message.startswith("cost[development step=0 initialization=Base actual]")
        for message in progress_messages
    )
    assert any(
        message.startswith("cost[student KD final actual]")
        for message in progress_messages
    )


def test_e9_rejects_checkpoint_initialization(tmp_path):
    with pytest.raises(KDTrainingError, match="Base-to-KD"):
        _config(tmp_path, initialization_source="e4_grpo_step75").validate(_manifest())
    with pytest.raises(KDTrainingError, match="Base-to-KD"):
        _config(tmp_path, initialization_label="e4-grpo-step75").validate(_manifest())


def test_e9_requires_the_full_candidate_union_and_one_pass_step_capacity(tmp_path):
    with pytest.raises(KDTrainingError, match="full-corpus"):
        _config(tmp_path, teacher_candidate_partitions=("rl_train",)).validate(
            _manifest()
        )
    with pytest.raises(KDTrainingError, match="every accepted"):
        _config(tmp_path, max_student_steps=2).validate(_manifest())


def test_kd_requires_g4_development_evaluation_and_token_cadence(tmp_path):
    with pytest.raises(KDTrainingError, match="development_examples"):
        _config(tmp_path, development_examples=0).validate(_manifest())
    with pytest.raises(KDTrainingError, match="G=4"):
        _config(tmp_path, development_group_size=1).validate(_manifest())

    config = _config(tmp_path, development_input_token_interval=10)
    report = build_doctor_report(
        config,
        _manifest(),
        environ={"TINKER_API_KEY": "set", "WANDB_API_KEY": "set"},
        tinker_version="0.27.0",
        wandb_version="0.21.1",
    )

    assert config.development_evaluations_upper_bound(_manifest()) == 39
    assert report["max_development_evaluations"] == 40
    assert report["development_partition"] == "sft_validation"
