"""Cost-gated, verifier-filtered teacher-response distillation for GSM8K.

E9 is deliberately the hard-target baseline in the distillation ladder:

* a frozen teacher writes a solution for every allowed training prompt;
* the GSM8K answer verifier keeps only correct, boxed traces; and
* a fresh LoRA on the untouched Base student is trained with ordinary token
  cross-entropy.

The ``signal_kind`` routing is intentionally explicit.  A teacher response is
hard-target KD; a teacher score over a student rollout is a future RLAIF branch,
not a different setting of this cross-entropy experiment.  Top-K teacher
distributions are also a future branch, with a different target tensor shape.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv

from modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval import (
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    PREFILL_USD_PER_MILLION,
    PROMPT_VERSION,
    SAMPLE_USD_PER_MILLION,
    SEED,
    SUITE_ID,
    WANDB_PROJECT,
    build_prompt,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    SplitManifest,
    content_id,
    load_official_train_rows,
    load_official_train_rows_for_partitions,
    read_manifest,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.distillation_schema import (
    DISTILLATION_SCHEMA_VERSION,
    DISTILLATION_SIGNAL_KINDS,
    HARD_RESPONSE,
    TEACHER_JUDGE,
    TOPK_RESPONSE,
    configure_wandb_metrics,
    metric_schema_dict,
    method_spec,
    validate_logged_metric_keys,
    write_metric_dictionary,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import (
    Completion,
    evaluate_groups,
    score_completion,
)


EXPERIMENT_ID = "e9"
TEACHER_RESPONSE = HARD_RESPONSE
TEACHER_TOPK = TOPK_RESPONSE
TEACHER_SCORE = TEACHER_JUDGE

TEACHER_MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
TEACHER_TEMPERATURE = 0.7
TEACHER_PREFILL_USD_PER_MILLION = 3.0
TEACHER_SAMPLE_USD_PER_MILLION = 7.5

# E4 is the frozen formal comparison reference, not the E9 initialization.
E4_COMPARISON_CHECKPOINT = (
    "e4-grpo-qwen-qwen3-5-9b-base-r32-b8-g4-lr2e-5-s100-m64-a01-step75"
)
E4_FORMAL_PASS_AT_1 = 0.71969697
E4_FORMAL_PASS_AT_4 = 0.75058275
INITIALIZATION_SOURCE = "base_fresh_lora"
INITIALIZATION_LABEL = "base-fresh-lora"
FULL_TRAINING_PARTITIONS = ("sft_train", "rl_train")
FULL_TRAINING_DATA_LABEL = "full-allowed-train-once"

LORA_RANK = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MAX_STUDENT_STEPS = 900
DEFAULT_TEACHER_BATCH_SIZE = 16
DEFAULT_DEVELOPMENT_PARTITION = "sft_validation"
DEFAULT_DEVELOPMENT_EXAMPLES = 64
DEFAULT_DEVELOPMENT_GROUP_SIZE = 4
DEFAULT_DEVELOPMENT_INPUT_TOKEN_INTERVAL = 500_000
DEFAULT_PROGRESS_EVERY = 8
CHECKPOINT_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_SEQUENCE_TOKENS = 1024
DEFAULT_HARD_CAP_USD = 60.0
TRAIN_USD_PER_MILLION = 1.463
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
OUTPUT_DIR = Path(__file__).parent / "outputs"


class KDTrainingError(RuntimeError):
    """Raised when a knowledge-distillation request is unsafe or invalid."""


@dataclass(frozen=True)
class KDConfig:
    """E9's executable recipe within the shared distillation method schema.

    The fields here intentionally remain narrow: this module currently owns only
    hard-response KD.  Later methods get their own signal-to-Datum adapters but
    share the method registry, metric contract, and report/selection provenance.
    """

    model_id: str = MODEL_ID
    project: str = WANDB_PROJECT
    suite_id: str = SUITE_ID
    experiment_id: str = EXPERIMENT_ID
    signal_kind: str = TEACHER_RESPONSE
    attempt: int = 1
    teacher_model_id: str = TEACHER_MODEL_ID
    teacher_temperature: float = TEACHER_TEMPERATURE
    teacher_candidate_partitions: Tuple[str, ...] = FULL_TRAINING_PARTITIONS
    teacher_batch_size: int = DEFAULT_TEACHER_BATCH_SIZE
    teacher_max_prompt_tokens: int = MAX_PROMPT_TOKENS
    teacher_max_output_tokens: int = MAX_OUTPUT_TOKENS
    require_teacher_boxed_format: bool = True
    teacher_prefill_usd_per_million: float = TEACHER_PREFILL_USD_PER_MILLION
    teacher_sample_usd_per_million: float = TEACHER_SAMPLE_USD_PER_MILLION
    initialization_source: str = INITIALIZATION_SOURCE
    initialization_label: str = INITIALIZATION_LABEL
    lora_rank: int = LORA_RANK
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS
    max_student_steps: int = DEFAULT_MAX_STUDENT_STEPS
    development_partition: str = DEFAULT_DEVELOPMENT_PARTITION
    development_examples: int = DEFAULT_DEVELOPMENT_EXAMPLES
    development_group_size: int = DEFAULT_DEVELOPMENT_GROUP_SIZE
    development_input_token_interval: int = DEFAULT_DEVELOPMENT_INPUT_TOKEN_INTERVAL
    progress_every: int = DEFAULT_PROGRESS_EVERY
    checkpoint_ttl_seconds: int = CHECKPOINT_TTL_SECONDS
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD
    train_usd_per_million: float = TRAIN_USD_PER_MILLION
    seed: int = SEED
    trace_output_dir: str = str(OUTPUT_DIR)

    def validate(self, manifest: Optional[SplitManifest] = None) -> None:
        if self.experiment_id != EXPERIMENT_ID:
            raise KDTrainingError(
                "the first KD recipe is reserved for experiment_id=e9"
            )
        try:
            method = method_spec(self.signal_kind)
        except ValueError as exc:
            raise KDTrainingError(str(exc)) from exc
        if self.signal_kind == TEACHER_TOPK:
            raise KDTrainingError(
                "teacher-topk requires a top-K target tensor and is not implemented "
                "in the hard-response baseline"
            )
        if self.signal_kind == TEACHER_SCORE:
            raise KDTrainingError(
                "teacher-score is a future RLAIF branch: it must score on-policy "
                "student rollouts rather than use cross-entropy teacher targets"
            )
        if method.implementation_status != "implemented":
            raise KDTrainingError(
                f"{self.signal_kind} is registered in {DISTILLATION_SCHEMA_VERSION} "
                "but does not yet have an executable signal-to-Datum adapter"
            )
        if not all(
            (
                self.model_id,
                self.project,
                self.suite_id,
                self.teacher_model_id,
                self.initialization_label,
            )
        ):
            raise KDTrainingError(
                "model, teacher, suite, and initialization IDs are required"
            )
        if (
            self.initialization_source != INITIALIZATION_SOURCE
            or self.initialization_label != INITIALIZATION_LABEL
        ):
            raise KDTrainingError(
                "E9 is the Base-to-KD baseline and must use the declared fresh Base LoRA"
            )
        positive_ints = {
            "attempt": self.attempt,
            "teacher_batch_size": self.teacher_batch_size,
            "teacher_max_prompt_tokens": self.teacher_max_prompt_tokens,
            "teacher_max_output_tokens": self.teacher_max_output_tokens,
            "lora_rank": self.lora_rank,
            "batch_size": self.batch_size,
            "max_sequence_tokens": self.max_sequence_tokens,
            "max_student_steps": self.max_student_steps,
            "development_group_size": self.development_group_size,
            "development_input_token_interval": self.development_input_token_interval,
            "progress_every": self.progress_every,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise KDTrainingError(f"{name} must be positive")
        if self.development_partition != DEFAULT_DEVELOPMENT_PARTITION:
            raise KDTrainingError(
                "hard-response KD development must use the frozen sft_validation split"
            )
        if self.teacher_candidate_partitions != FULL_TRAINING_PARTITIONS:
            raise KDTrainingError(
                "E9 is the full-corpus Base-to-KD recipe and must use exactly "
                "sft_train plus rl_train teacher candidates"
            )
        if self.development_examples <= 0:
            raise KDTrainingError("development_examples must be positive")
        if self.development_group_size != 4:
            raise KDTrainingError("development evaluation must retain G=4 for pass@4")
        if (
            min(
                self.teacher_temperature,
                self.learning_rate,
                self.hard_cap_usd,
                self.teacher_prefill_usd_per_million,
                self.teacher_sample_usd_per_million,
                self.train_usd_per_million,
            )
            <= 0
        ):
            raise KDTrainingError("rates, prices, and hard cap must be positive")
        if self.seed < 0:
            raise KDTrainingError("seed cannot be negative")
        if manifest is not None:
            manifest.validate()
            if self.development_examples > len(manifest.sft_validation_ids):
                raise KDTrainingError(
                    "development_examples exceeds frozen sft_validation"
                )
            maximum_batches = math.ceil(
                _teacher_candidate_count(self, manifest) / self.batch_size
            )
            if self.max_student_steps < maximum_batches:
                raise KDTrainingError(
                    "max_student_steps cannot accommodate one pass over every "
                    "accepted full-corpus E9 trace"
                )

    @property
    def run_name(self) -> str:
        model_slug = self.model_id.lower().replace("/", "-").replace(".", "-")
        teacher_slug = self.teacher_model_id.rsplit("/", 1)[-1].lower()
        teacher_slug = teacher_slug.replace(".", "-")
        source_slug = re.sub(r"[^a-zA-Z0-9]+", "-", self.initialization_label)
        lr_slug = f"{self.learning_rate:.0e}".replace("-0", "-")
        return (
            f"{self.experiment_id}-kd-{self.signal_kind}-{model_slug}"
            f"-teacher-{teacher_slug}-from-{source_slug.lower().strip('-')}"
            f"-r{self.lora_rank}-b{self.batch_size}-lr{lr_slug}"
            f"-{FULL_TRAINING_DATA_LABEL}"
            f"-dev{self.development_partition.replace('_', '-')}{self.development_examples}"
            f"-di{self.development_input_token_interval}"
            f"-a{self.attempt:02d}-seed{self.seed}"
        )

    def development_evaluations_upper_bound(self, manifest: SplitManifest) -> int:
        """Bound trained checkpoints from the token-based development cadence."""
        if not self.development_examples:
            return 0
        return math.ceil(
            _student_input_token_upper_bound(self, manifest)
            / self.development_input_token_interval
        )


@dataclass(frozen=True)
class TokenizedDistillationExample:
    """Teacher hard targets represented in the student's tokenizer."""

    example_id: str
    input_tokens: Tuple[int, ...]
    target_tokens: Tuple[int, ...]
    weights: Tuple[float, ...]
    supervised_tokens: int


@dataclass(frozen=True)
class TeacherCandidate:
    """One sampled teacher completion before verifier filtering."""

    example_id: str
    response: str
    prompt_tokens: int
    output_tokens: int
    scored: Any


@dataclass(frozen=True)
class AcceptedTeacherTrace:
    """One audited response selected as a hard CE target."""

    example_id: str
    response: str
    prompt_tokens: int
    output_tokens: int
    student_input_tokens: int
    student_supervised_tokens: int
    parsed_answer: Optional[str]


@dataclass(frozen=True)
class DevelopmentReport:
    """Generation metrics, audit rows, and inference cost for one checkpoint."""

    metrics: Dict[str, float]
    prompt_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    table_rows: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class CheckpointRecord:
    """One generation-scored checkpoint, including initialization at step zero."""

    step: int
    state_path: Optional[str]
    sampler_path: Optional[str]
    development_pass_at_1: Optional[float]
    development_pass_at_4: Optional[float]
    development_metrics: Optional[Dict[str, float]]


def _print_progress(message: str) -> None:
    print(f"[gsm8k-kd] {message}", file=sys.stderr, flush=True)


def load_local_env() -> None:
    """Load ignored credentials without replacing shell values."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    return [int(token) for token in tokens]


def tokenize_teacher_response(
    row: Mapping[str, object],
    teacher_response: str,
    tokenizer: Any,
    config: KDConfig,
) -> Optional[TokenizedDistillationExample]:
    """Mask the prompt and apply hard CE only to a verified teacher trace."""
    prompt_tokens = _encode(tokenizer, build_prompt(str(row["question"])))
    if not prompt_tokens or len(prompt_tokens) > config.teacher_max_prompt_tokens:
        raise KDTrainingError("a student prompt exceeds the configured token limit")
    completion_tokens = _encode(tokenizer, teacher_response)
    if not completion_tokens:
        raise KDTrainingError("teacher returned an empty response")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_tokens.append(int(eos_token_id))
    full_tokens = prompt_tokens + completion_tokens
    if len(full_tokens) - 1 > config.max_sequence_tokens:
        return None
    input_tokens = tuple(full_tokens[:-1])
    target_tokens = tuple(full_tokens[1:])
    weights = tuple(
        [0.0] * (len(prompt_tokens) - 1)
        + [1.0] * (len(full_tokens) - len(prompt_tokens))
    )
    if not (input_tokens and len(input_tokens) == len(target_tokens) == len(weights)):
        raise KDTrainingError("tokenized KD tensors have inconsistent lengths")
    return TokenizedDistillationExample(
        example_id=content_id(row),
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=weights,
        supervised_tokens=int(sum(weights)),
    )


def _batches(
    examples: Sequence[TokenizedDistillationExample], batch_size: int
) -> Tuple[Tuple[TokenizedDistillationExample, ...], ...]:
    return tuple(
        tuple(examples[start : start + batch_size])
        for start in range(0, len(examples), batch_size)
    )


def _materialize_batch(
    examples: Sequence[TokenizedDistillationExample], tinker_module: Any
) -> Tuple[list[Any], int, int]:
    data = []
    input_tokens = 0
    supervised_tokens = 0
    for example in examples:
        data.append(
            tinker_module.types.Datum(
                model_input=tinker_module.ModelInput.from_ints(
                    tokens=list(example.input_tokens)
                ),
                loss_fn_inputs={
                    "weights": list(example.weights),
                    "target_tokens": list(example.target_tokens),
                },
            )
        )
        input_tokens += len(example.input_tokens)
        supervised_tokens += example.supervised_tokens
    return data, input_tokens, supervised_tokens


def _loss_sum(result: Any) -> float:
    metrics = getattr(result, "metrics", {})
    if "loss:sum" in metrics:
        return float(metrics["loss:sum"])
    if getattr(result, "loss", None) is not None:
        return float(result.loss)
    raise KDTrainingError("Tinker did not return a readable loss diagnostic")


def _perplexity(nll: float) -> float:
    return math.exp(min(nll, 80.0))


def _teacher_cost(prompt_tokens: int, output_tokens: int, config: KDConfig) -> float:
    return (
        prompt_tokens * config.teacher_prefill_usd_per_million
        + output_tokens * config.teacher_sample_usd_per_million
    ) / 1_000_000


def _student_monitor_cost(prompt_tokens: int, output_tokens: int) -> float:
    return (
        prompt_tokens * PREFILL_USD_PER_MILLION + output_tokens * SAMPLE_USD_PER_MILLION
    ) / 1_000_000


def _student_train_cost(input_tokens: int, config: KDConfig) -> float:
    return input_tokens * config.train_usd_per_million / 1_000_000


def _trace_digest(traces: Sequence[AcceptedTeacherTrace]) -> str:
    payload = [
        {"example_id": trace.example_id, "response": trace.response} for trace in traces
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ids_hash(example_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(example_ids).encode()).hexdigest()


def _candidate_ids_hash(rows: Sequence[Mapping[str, object]]) -> str:
    return _ids_hash(tuple(content_id(row) for row in rows))


def _teacher_candidate_ids(
    config: KDConfig, manifest: SplitManifest
) -> Tuple[str, ...]:
    """Return the exact ordered full-corpus E9 candidate IDs from the manifest."""
    ids_by_partition = {
        "sft_train": manifest.sft_train_ids,
        "sft_validation": manifest.sft_validation_ids,
        "rl_train": manifest.rl_train_ids,
        "rl_monitor": manifest.rl_monitor_ids,
    }
    return tuple(
        example_id
        for partition in config.teacher_candidate_partitions
        for example_id in ids_by_partition[partition]
    )


def _teacher_candidate_count(config: KDConfig, manifest: SplitManifest) -> int:
    return len(_teacher_candidate_ids(config, manifest))


def _student_input_token_upper_bound(config: KDConfig, manifest: SplitManifest) -> int:
    """Bound one CE pass over every accepted trace without truncating the corpus."""
    return _teacher_candidate_count(config, manifest) * config.max_sequence_tokens


def _max_development_runs(config: KDConfig, manifest: SplitManifest) -> int:
    """Count initialization plus every token-cadenced development evaluation."""
    if not config.development_examples:
        return 0
    return 1 + config.development_evaluations_upper_bound(manifest)


def _development_ids_hash(config: KDConfig, manifest: SplitManifest) -> Optional[str]:
    if not config.development_examples:
        return None
    return hashlib.sha256(
        "\n".join(manifest.sft_validation_ids[: config.development_examples]).encode()
    ).hexdigest()


def estimate_max_token_cost_usd(config: KDConfig, manifest: SplitManifest) -> float:
    """Bound teacher sampling, student training, and development rollout calls."""
    config.validate(manifest)
    candidate_count = _teacher_candidate_count(config, manifest)
    teacher = _teacher_cost(
        candidate_count * config.teacher_max_prompt_tokens,
        candidate_count * config.teacher_max_output_tokens,
        config,
    )
    student_training = _student_train_cost(
        _student_input_token_upper_bound(config, manifest), config
    )
    development = 0.0
    if config.development_examples:
        development = _student_monitor_cost(
            _max_development_runs(config, manifest)
            * config.development_examples
            * config.development_group_size
            * MAX_PROMPT_TOKENS,
            _max_development_runs(config, manifest)
            * config.development_examples
            * config.development_group_size
            * MAX_OUTPUT_TOKENS,
        )
    return teacher + student_training + development


def _cost_breakdown(config: KDConfig, manifest: SplitManifest) -> Dict[str, float]:
    total = estimate_max_token_cost_usd(config, manifest)
    candidate_count = _teacher_candidate_count(config, manifest)
    teacher = _teacher_cost(
        candidate_count * config.teacher_max_prompt_tokens,
        candidate_count * config.teacher_max_output_tokens,
        config,
    )
    student_training = _student_train_cost(
        _student_input_token_upper_bound(config, manifest), config
    )
    return {
        "teacher_generation_max_usd": teacher,
        "student_training_max_usd": student_training,
        "development_inference_max_usd": total - teacher - student_training,
        "total_max_usd": total,
    }


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _tracking_config(config: KDConfig, manifest: SplitManifest) -> Dict[str, Any]:
    method = method_spec(config.signal_kind)
    return {
        "experiment_id": config.experiment_id,
        "attempt": config.attempt,
        "suite_id": config.suite_id,
        "signal_kind": config.signal_kind,
        "distillation_schema_version": DISTILLATION_SCHEMA_VERSION,
        "distillation_method": asdict(method),
        "metric_schema": metric_schema_dict(config.signal_kind),
        "teacher_model_id": config.teacher_model_id,
        "teacher_temperature": config.teacher_temperature,
        "teacher_filter": "verifier_correct_and_boxed",
        "teacher_candidate_partitions": list(config.teacher_candidate_partitions),
        "teacher_candidate_data_policy": FULL_TRAINING_DATA_LABEL,
        "teacher_candidate_count": _teacher_candidate_count(config, manifest),
        "teacher_candidate_ids_hash": _ids_hash(
            _teacher_candidate_ids(config, manifest)
        ),
        "student_model_id": config.model_id,
        "initialization_source": config.initialization_source,
        "initialization_label": config.initialization_label,
        "parent_checkpoint": None,
        "parent_state_path": None,
        "parent_sampler_path": None,
        "reference_e4_checkpoint": E4_COMPARISON_CHECKPOINT,
        "student_training_data_policy": "all_accepted_teacher_traces_once",
        "student_input_token_upper_bound": _student_input_token_upper_bound(
            config, manifest
        ),
        "reference_e4_formal_pass_at_1": E4_FORMAL_PASS_AT_1,
        "reference_e4_formal_pass_at_4": E4_FORMAL_PASS_AT_4,
        "dataset_id": manifest.dataset_id,
        "dataset_revision": manifest.dataset_revision,
        "manifest_hash": manifest.manifest_hash,
        "prompt_version": PROMPT_VERSION,
        "max_sequence_tokens": config.max_sequence_tokens,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_student_steps": config.max_student_steps,
        "development_partition": config.development_partition,
        "development_data_status": (
            "held_out_from_kd_training_reused_sft_validation_not_cross_run_evidence"
        ),
        "development_examples": config.development_examples,
        "development_ids_hash": _development_ids_hash(config, manifest),
        "development_group_size": config.development_group_size,
        "development_input_token_interval": config.development_input_token_interval,
        "checkpoint_selection": "development_pass_at_4_then_pass_at_1",
        "teacher_pricing_per_million": {
            "prefill": config.teacher_prefill_usd_per_million,
            "sample": config.teacher_sample_usd_per_million,
        },
        "student_train_usd_per_million": config.train_usd_per_million,
        "hard_cap_usd": config.hard_cap_usd,
        "git_sha": _git_sha(),
        "hypothesis": (
            "One pass over verifier-filtered stronger-teacher traces from every "
            "allowed training prompt improves a Base student relative to the "
            "Base-to-SFT-to-GRPO E4 recipe."
        ),
        "expected_failure": (
            "Teacher traces are correct but the Base-to-KD student does not improve "
            "the held-out development behavior or formal performance over E4."
        ),
    }


def build_doctor_report(
    config: KDConfig,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_version: Optional[str] = None,
    wandb_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate E9's complete cost envelope without network or paid calls."""
    manifest = read_manifest() if manifest is None else manifest
    config.validate(manifest)
    tinker_sdk = (
        _package_version("tinker") if tinker_version is None else tinker_version
    )
    wandb_sdk = _package_version("wandb") if wandb_version is None else wandb_version
    estimate = estimate_max_token_cost_usd(config, manifest)
    return {
        "mode": "local-kd-preflight",
        "network_called": False,
        "run_name": config.run_name,
        "signal_kind": config.signal_kind,
        "distillation_schema_version": DISTILLATION_SCHEMA_VERSION,
        "distillation_method": asdict(method_spec(config.signal_kind)),
        "metric_schema": metric_schema_dict(config.signal_kind),
        "teacher_model_id": config.teacher_model_id,
        "teacher_candidate_partitions": list(config.teacher_candidate_partitions),
        "teacher_candidate_data_policy": FULL_TRAINING_DATA_LABEL,
        "teacher_candidate_count": _teacher_candidate_count(config, manifest),
        "student_model_id": config.model_id,
        "initialization_source": config.initialization_source,
        "initialization_label": config.initialization_label,
        "parent_checkpoint": None,
        "parent_state_path": None,
        "parent_sampler_path": None,
        "reference_e4_checkpoint": E4_COMPARISON_CHECKPOINT,
        "student_training_data_policy": "all_accepted_teacher_traces_once",
        "student_input_token_upper_bound": _student_input_token_upper_bound(
            config, manifest
        ),
        "reference_e4_formal_pass_at_1": E4_FORMAL_PASS_AT_1,
        "reference_e4_formal_pass_at_4": E4_FORMAL_PASS_AT_4,
        "max_student_steps": config.max_student_steps,
        "max_development_evaluations": _max_development_runs(config, manifest),
        "development_partition": config.development_partition,
        "development_examples": config.development_examples,
        "development_ids_hash": _development_ids_hash(config, manifest),
        "development_group_size": config.development_group_size,
        "development_input_token_interval": config.development_input_token_interval,
        "estimated_cost_breakdown_usd": _cost_breakdown(config, manifest),
        "tinker_sdk_version": tinker_sdk,
        "wandb_version": wandb_sdk,
        "tinker_api_key_configured": bool(environ.get("TINKER_API_KEY")),
        "wandb_api_key_configured": bool(environ.get("WANDB_API_KEY")),
        "hard_cap_usd": config.hard_cap_usd,
        "ready_for_paid_run": (
            sys.version_info[:2] >= (3, 11)
            and tinker_sdk is not None
            and wandb_sdk is not None
            and bool(environ.get("TINKER_API_KEY"))
            and bool(environ.get("WANDB_API_KEY"))
            and environ.get("WANDB_MODE", "").lower() != "offline"
            and estimate <= config.hard_cap_usd
        ),
    }


def _authorize(
    config: KDConfig,
    manifest: SplitManifest,
    allow_paid: bool,
    environ: Mapping[str, str],
) -> None:
    config.validate(manifest)
    if not allow_paid:
        raise KDTrainingError("training is blocked; pass --allow-paid after approval")
    if not environ.get("TINKER_API_KEY") or not environ.get("WANDB_API_KEY"):
        raise KDTrainingError("TINKER_API_KEY and WANDB_API_KEY are required")
    if environ.get("WANDB_MODE", "").lower() == "offline":
        raise KDTrainingError(
            "WANDB_MODE=offline cannot produce the required dashboard"
        )
    if estimate_max_token_cost_usd(config, manifest) > config.hard_cap_usd:
        raise KDTrainingError("estimated maximum token cost exceeds the hard cap")


async def _sample_teacher_candidate(
    index: int,
    row: Mapping[str, object],
    teacher_client: Any,
    teacher_tokenizer: Any,
    tinker_module: Any,
    config: KDConfig,
) -> TeacherCandidate:
    prompt_tokens = _encode(teacher_tokenizer, build_prompt(str(row["question"])))
    if not prompt_tokens or len(prompt_tokens) > config.teacher_max_prompt_tokens:
        raise KDTrainingError("a teacher prompt exceeds the configured token limit")
    result = await teacher_client.sample_async(
        prompt=tinker_module.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=1,
        sampling_params=tinker_module.SamplingParams(
            max_tokens=config.teacher_max_output_tokens,
            temperature=config.teacher_temperature,
            seed=config.seed + index,
        ),
    )
    if len(result.sequences) != 1:
        raise KDTrainingError("teacher returned the wrong number of samples")
    tokens = tuple(int(token) for token in result.sequences[0].tokens)
    if not tokens:
        raise KDTrainingError("teacher returned an empty completion")
    example_id = content_id(row)
    response = teacher_tokenizer.decode(list(tokens))
    scored = score_completion(
        Completion(
            example_id=example_id,
            response=response,
            ground_truth=str(row["answer"]),
            output_tokens=len(tokens),
            max_output_tokens=config.teacher_max_output_tokens,
        ),
        group_id=example_id,
    )
    return TeacherCandidate(
        example_id=example_id,
        response=response,
        prompt_tokens=len(prompt_tokens),
        output_tokens=len(tokens),
        scored=scored,
    )


async def _collect_teacher_traces(
    rows: Sequence[Mapping[str, object]],
    student_tokenizer: Any,
    teacher_client: Any,
    tinker_module: Any,
    config: KDConfig,
    progress: Callable[[str], None],
) -> Tuple[
    Tuple[TokenizedDistillationExample, ...],
    Tuple[AcceptedTeacherTrace, ...],
    Dict[str, int],
    int,
    int,
]:
    """Sample every candidate and retain each verifier-acceptable hard target."""
    candidate_rows = tuple(rows)
    teacher_tokenizer = teacher_client.get_tokenizer()
    accepted_examples = []
    accepted_traces = []
    outcomes = {
        "teacher_candidates_sampled": 0,
        "teacher_correct": 0,
        "teacher_rejected_incorrect": 0,
        "teacher_rejected_format": 0,
        "teacher_rejected_truncated": 0,
        "teacher_rejected_overlength": 0,
    }
    teacher_prompt_tokens = 0
    teacher_output_tokens = 0
    selected_input_tokens = 0

    for start in range(0, len(candidate_rows), config.teacher_batch_size):
        batch_rows = candidate_rows[start : start + config.teacher_batch_size]
        tasks = [
            asyncio.create_task(
                _sample_teacher_candidate(
                    start + offset,
                    row,
                    teacher_client,
                    teacher_tokenizer,
                    tinker_module,
                    config,
                )
            )
            for offset, row in enumerate(batch_rows)
        ]
        try:
            candidates = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for row, candidate in zip(batch_rows, candidates):
            outcomes["teacher_candidates_sampled"] += 1
            teacher_prompt_tokens += candidate.prompt_tokens
            teacher_output_tokens += candidate.output_tokens
            if not candidate.scored.correct:
                outcomes["teacher_rejected_incorrect"] += 1
                continue
            outcomes["teacher_correct"] += 1
            if (
                config.require_teacher_boxed_format
                and not candidate.scored.format_valid
            ):
                outcomes["teacher_rejected_format"] += 1
                continue
            if candidate.scored.truncated:
                outcomes["teacher_rejected_truncated"] += 1
                continue
            tokenized = tokenize_teacher_response(
                row, candidate.response, student_tokenizer, config
            )
            if tokenized is None:
                outcomes["teacher_rejected_overlength"] += 1
                continue
            accepted_examples.append(tokenized)
            accepted_traces.append(
                AcceptedTeacherTrace(
                    example_id=candidate.example_id,
                    response=candidate.response,
                    prompt_tokens=candidate.prompt_tokens,
                    output_tokens=candidate.output_tokens,
                    student_input_tokens=len(tokenized.input_tokens),
                    student_supervised_tokens=tokenized.supervised_tokens,
                    parsed_answer=candidate.scored.parsed_answer,
                )
            )
            selected_input_tokens += len(tokenized.input_tokens)

        progress(
            f"teacher candidates={outcomes['teacher_candidates_sampled']}/"
            f"{len(candidate_rows)} accepted={len(accepted_examples)} "
            f"student_input_tokens={selected_input_tokens}"
        )

    if not accepted_examples:
        raise KDTrainingError("teacher filtering produced no supervised traces")
    return (
        tuple(accepted_examples),
        tuple(accepted_traces),
        outcomes,
        teacher_prompt_tokens,
        teacher_output_tokens,
    )


DEVELOPMENT_TABLE_COLUMNS = (
    "example_id",
    "checkpoint_step",
    "rollout_id",
    "question",
    "ground_truth",
    "generated_response",
    "parsed_answer",
    "correct",
    "output_tokens",
    "format_valid",
    "truncated",
    "process_checked_steps",
    "process_valid_steps",
    "process_invalid_steps",
)


def _development_table_rows(
    report: Any,
    samples: Sequence[Mapping[str, Any]],
    checkpoint_step: int,
    config: KDConfig,
) -> Tuple[Tuple[Any, ...], ...]:
    """Keep each student development rollout inspectable in W&B."""
    sample_by_id = {str(sample["example_id"]): sample for sample in samples}
    return tuple(
        (
            row.example_id,
            checkpoint_step,
            rollout_id % config.development_group_size,
            sample_by_id[row.example_id]["question"],
            sample_by_id[row.example_id]["ground_truth"],
            row.response,
            row.parsed_answer,
            row.correct,
            row.output_tokens,
            row.format_valid,
            row.truncated,
            row.process.checked_steps,
            row.process.valid_steps,
            row.process.invalid_steps,
        )
        for rollout_id, row in enumerate(report.rows)
    )


async def _generation_development(
    service_client: Any,
    model_path: Optional[str],
    rows: Sequence[Mapping[str, object]],
    tinker_module: Any,
    config: KDConfig,
    checkpoint_step: int,
    label: str,
    progress: Callable[[str], None],
) -> DevelopmentReport:
    """Evaluate one student checkpoint on the frozen G=4 KD development set."""
    if model_path is None:
        client = await service_client.create_sampling_client_async(
            base_model=config.model_id
        )
    else:
        client = await service_client.create_sampling_client_async(
            model_path=model_path
        )
    tokenizer = client.get_tokenizer()

    async def sample_group(index: int, row: Mapping[str, object]) -> Dict[str, Any]:
        prompt_tokens = _encode(tokenizer, build_prompt(str(row["question"])))
        if not prompt_tokens or len(prompt_tokens) > MAX_PROMPT_TOKENS:
            raise KDTrainingError(
                "a development prompt exceeds the configured token limit"
            )
        result = await client.sample_async(
            prompt=tinker_module.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=config.development_group_size,
            sampling_params=tinker_module.SamplingParams(
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=1.0,
                seed=config.seed + index,
            ),
        )
        if len(result.sequences) != config.development_group_size:
            raise KDTrainingError("development received the wrong number of samples")
        return {
            "example_id": content_id(row),
            "question": str(row["question"]),
            "ground_truth": str(row["answer"]),
            "prompt_tokens": len(prompt_tokens),
            "responses": tuple(
                (tokenizer.decode(list(sequence.tokens)), len(sequence.tokens))
                for sequence in result.sequences
            ),
        }

    tasks = [
        asyncio.create_task(sample_group(index, row)) for index, row in enumerate(rows)
    ]
    try:
        samples = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    groups = {
        sample["example_id"]: tuple(
            Completion(
                example_id=sample["example_id"],
                response=response,
                ground_truth=sample["ground_truth"],
                output_tokens=output_tokens,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            for response, output_tokens in sample["responses"]
        )
        for sample in samples
    }
    report = evaluate_groups(groups, pass_k=config.development_group_size)
    prompt_total = sum(
        sample["prompt_tokens"] * config.development_group_size for sample in samples
    )
    output_total = sum(
        output_tokens for sample in samples for _, output_tokens in sample["responses"]
    )
    progress(
        f"development {label} pass_at_1={report.metrics['eval/pass_at_1']:.4f} "
        f"pass_at_4={report.metrics['eval/pass_at_4']:.4f}"
    )
    return DevelopmentReport(
        metrics=dict(report.metrics),
        prompt_tokens=prompt_total,
        output_tokens=output_total,
        estimated_cost_usd=_student_monitor_cost(prompt_total, output_total),
        table_rows=_development_table_rows(report, samples, checkpoint_step, config),
    )


def _development_metrics(report: DevelopmentReport) -> Dict[str, float]:
    metric_map = {
        "eval/pass_at_1": "dev/pass_at_1",
        "eval/pass_at_4": "dev/pass_at_4",
        "eval/format_accuracy": "dev/format_accuracy",
        "eval/truncation_rate": "dev/truncation_rate",
        "eval/avg_output_tokens": "dev/avg_output_tokens",
        "eval/group_all_correct_frac": "dev/group_all_correct_frac",
        "eval/group_all_wrong_frac": "dev/group_all_wrong_frac",
        "eval/group_mixed_frac": "dev/group_mixed_frac",
        "eval/group_reward_std_mean": "dev/group_reward_std_mean",
        "eval/group_unique_response_frac": "dev/group_unique_response_frac",
        "eval/process_check_coverage": "dev/process_check_coverage",
        "eval/process_validity_rate": "dev/process_validity_rate",
        "eval/final_correct_process_invalid": "dev/final_correct_process_invalid",
        "eval/final_correct_process_valid": "dev/final_correct_process_valid",
    }
    return {
        destination: float(report.metrics[source])
        for source, destination in metric_map.items()
    }


def _log_schema_metrics(
    wandb_run: Any, config: KDConfig, payload: Mapping[str, Any], step: int
) -> None:
    """Reject accidental, undocumented dashboard metrics before they reach W&B."""
    unregistered = validate_logged_metric_keys(config.signal_kind, payload.keys())
    if unregistered:
        raise KDTrainingError(
            "distillation metric schema is missing: " + ", ".join(unregistered)
        )
    wandb_run.log(dict(payload), step=step)


def _is_better_checkpoint(
    candidate: CheckpointRecord, current: CheckpointRecord
) -> bool:
    if (
        candidate.development_pass_at_4 is None
        or candidate.development_pass_at_1 is None
    ):
        return candidate.step > current.step
    if current.development_pass_at_4 is None or current.development_pass_at_1 is None:
        return True
    return (candidate.development_pass_at_4, candidate.development_pass_at_1) > (
        current.development_pass_at_4,
        current.development_pass_at_1,
    )


def _trace_path(config: KDConfig, run_id: str) -> Path:
    safe_run_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id)
    return (
        Path(config.trace_output_dir)
        / f"{config.experiment_id}_teacher_traces_{safe_run_id}.jsonl"
    )


def _report_path(config: KDConfig, run_id: str) -> Path:
    safe_run_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id)
    return (
        Path(config.trace_output_dir)
        / f"{config.experiment_id}_kd_report_{safe_run_id}.json"
    )


def _metric_dictionary_path(config: KDConfig, run_id: str) -> Path:
    safe_run_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id)
    return (
        Path(config.trace_output_dir)
        / f"{config.experiment_id}_metric_dictionary_{safe_run_id}.md"
    )


def _write_trace_artifact(path: Path, traces: Sequence[AcceptedTeacherTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for trace in traces:
            handle.write(json.dumps(asdict(trace), sort_keys=True) + "\n")


async def run_kd_training(
    config: KDConfig,
    allow_paid: bool,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
    train_rows: Optional[Sequence[Mapping[str, object]]] = None,
    development_rows: Optional[Sequence[Mapping[str, object]]] = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = _print_progress,
) -> Dict[str, Any]:
    """Run E9 only after an explicit paid-run acknowledgement."""
    manifest = read_manifest() if manifest is None else manifest
    _authorize(config, manifest, allow_paid, environ)
    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise KDTrainingError(
                "Tinker SDK is unavailable; run with `uv run --extra tinker`"
            ) from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise KDTrainingError("Weights & Biases is unavailable") from exc
    if train_rows is None:
        progress("loading frozen full-corpus teacher-candidate rows")
        train_rows = load_official_train_rows_for_partitions(
            manifest, config.teacher_candidate_partitions
        )
    if development_rows is None:
        progress("loading frozen sft_validation KD-development rows")
        development_rows = load_official_train_rows(
            manifest, config.development_partition
        )
    if len(train_rows) != _teacher_candidate_count(config, manifest):
        raise KDTrainingError(
            "loaded teacher-candidate rows do not match the full E9 manifest scope"
        )
    if tuple(content_id(row) for row in train_rows) != _teacher_candidate_ids(
        config, manifest
    ):
        raise KDTrainingError(
            "teacher-candidate rows do not exactly match the ordered full E9 split"
        )
    if len(development_rows) != len(manifest.sft_validation_ids):
        raise KDTrainingError(
            "loaded sft_validation rows do not match the frozen KD development split"
        )
    if (
        tuple(content_id(row) for row in development_rows)
        != manifest.sft_validation_ids
    ):
        raise KDTrainingError(
            "development rows do not exactly match the ordered frozen sft_validation split"
        )
    development_rows = tuple(development_rows[: config.development_examples])

    owned_http_client = None
    if service_client is None:
        import httpx

        owned_http_client = httpx.AsyncClient(follow_redirects=True)
        service_client = tinker_module.ServiceClient(
            user_metadata={
                "experiment_id": config.experiment_id,
                "suite_id": config.suite_id,
            },
            http_client=owned_http_client,
        )

    wandb_run = None
    try:
        progress(
            f"authorized run={config.run_name} teacher_candidates="
            f"{len(train_rows)} training_policy=all_accepted_once max_cost="
            f"${estimate_max_token_cost_usd(config, manifest):.4f}"
        )
        progress("initializing a fresh Base LoRA with a fresh KD optimizer")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_id,
            rank=config.lora_rank,
            seed=config.seed,
            user_metadata={"experiment_id": config.experiment_id},
        )
        student_tokenizer = training_client.get_tokenizer()
        wandb_run = wandb_module.init(
            project=config.project,
            entity=environ.get("WANDB_ENTITY") or None,
            name=config.run_name,
            group=config.suite_id,
            job_type="teacher-response-kd",
            tags=["gsm8k", "kd", config.experiment_id, config.signal_kind, "from-base"],
            config=_tracking_config(config, manifest),
        )
        configure_wandb_metrics(wandb_run, config.signal_kind)
        metric_dictionary_path = _metric_dictionary_path(
            config, str(getattr(wandb_run, "id", "run"))
        )
        write_metric_dictionary(metric_dictionary_path, config.signal_kind)
        progress(f"started W&B run={getattr(wandb_run, 'url', None)}")
        progress("creating frozen teacher sampling client")
        teacher_client = await service_client.create_sampling_client_async(
            base_model=config.teacher_model_id
        )
        (
            prepared,
            accepted_traces,
            outcomes,
            teacher_prompt_tokens,
            teacher_output_tokens,
        ) = await _collect_teacher_traces(
            train_rows,
            student_tokenizer,
            teacher_client,
            tinker_module,
            config,
            progress,
        )
        train_batches = _batches(prepared, config.batch_size)
        if len(train_batches) > config.max_student_steps:
            raise KDTrainingError(
                "all accepted teacher traces need more batches than max_student_steps; "
                "increase the explicit full-corpus safety bound before training"
            )
        trace_path = _trace_path(config, str(getattr(wandb_run, "id", "run")))
        _write_trace_artifact(trace_path, accepted_traces)
        trace_digest = _trace_digest(accepted_traces)
        selected_input_tokens = sum(len(example.input_tokens) for example in prepared)
        selected_supervised_tokens = sum(
            example.supervised_tokens for example in prepared
        )
        teacher_cost = _teacher_cost(
            teacher_prompt_tokens, teacher_output_tokens, config
        )
        _log_schema_metrics(
            wandb_run,
            config,
            {
                "train/optimizer_step": 0.0,
                "train/optimized_input_tokens": 0.0,
                "train/supervised_or_weighted_tokens": 0.0,
                "data/teacher_candidate_count": float(
                    outcomes["teacher_candidates_sampled"]
                ),
                "data/teacher_accepted_count": float(len(prepared)),
                "data/teacher_accept_rate": len(prepared)
                / outcomes["teacher_candidates_sampled"],
                "data/teacher_rejected_incorrect_count": float(
                    outcomes["teacher_rejected_incorrect"]
                ),
                "data/teacher_rejected_format_count": float(
                    outcomes["teacher_rejected_format"]
                ),
                "data/teacher_rejected_truncated_count": float(
                    outcomes["teacher_rejected_truncated"]
                ),
                "data/teacher_prompt_tokens": float(teacher_prompt_tokens),
                "data/teacher_output_tokens": float(teacher_output_tokens),
                "cost/teacher_generation_usd": teacher_cost,
                "cost/student_training_usd": 0.0,
                "cost/dev_inference_usd": 0.0,
                "cost/cumulative_usd": teacher_cost,
            },
            step=0,
        )
        progress(
            f"teacher data ready accepted={len(prepared)} input_tokens="
            f"{selected_input_tokens} policy=all_accepted_once "
            f"teacher_cost=${teacher_cost:.4f}"
        )

        development_cost = 0.0
        checkpoints = []
        if development_rows:
            parent_development = await _generation_development(
                service_client,
                None,
                development_rows,
                tinker_module,
                config,
                0,
                "step=0 initialization=Base",
                progress,
            )
            development_cost += parent_development.estimated_cost_usd
            parent_record = CheckpointRecord(
                step=0,
                state_path=None,
                sampler_path=None,
                development_pass_at_1=parent_development.metrics["eval/pass_at_1"],
                development_pass_at_4=parent_development.metrics["eval/pass_at_4"],
                development_metrics=_development_metrics(parent_development),
            )
            checkpoints.append(parent_record)
            parent_metrics: Dict[str, Any] = {
                **parent_record.development_metrics,
                "dev/checkpoint_step": 0.0,
                "dev/optimized_input_tokens": 0.0,
                "dev/generated_rollouts": float(len(parent_development.table_rows)),
                "dev/is_initialization_policy": 1.0,
                "cost/teacher_generation_usd": teacher_cost,
                "cost/student_training_usd": 0.0,
                "cost/dev_inference_usd": development_cost,
                "cost/cumulative_usd": teacher_cost + development_cost,
            }
            if hasattr(wandb_module, "Table"):
                parent_metrics["tables/development_rollouts"] = wandb_module.Table(
                    columns=list(DEVELOPMENT_TABLE_COLUMNS),
                    data=list(parent_development.table_rows),
                )
            _log_schema_metrics(wandb_run, config, parent_metrics, step=0)

        completed_input_tokens = 0
        completed_supervised_tokens = 0
        elapsed_started_at = clock()
        next_development_token_threshold = config.development_input_token_interval
        for step, batch in enumerate(train_batches, start=1):
            data, batch_input_tokens, batch_supervised_tokens = _materialize_batch(
                batch, tinker_module
            )
            step_started_at = clock()
            forward_backward = await training_client.forward_backward_async(
                data=data, loss_fn="cross_entropy"
            )
            forward_backward_result = await forward_backward.result_async()
            optimizer = await training_client.optim_step_async(
                tinker_module.types.AdamParams(learning_rate=config.learning_rate)
            )
            await optimizer.result_async()
            completed_input_tokens += batch_input_tokens
            completed_supervised_tokens += batch_supervised_tokens
            nll = _loss_sum(forward_backward_result) / batch_supervised_tokens
            step_seconds = max(clock() - step_started_at, 1e-9)
            elapsed_seconds = clock() - elapsed_started_at
            total_estimate = (
                teacher_cost
                + development_cost
                + _student_train_cost(completed_input_tokens, config)
            )
            _log_schema_metrics(
                wandb_run,
                config,
                {
                    "train/optimizer_step": float(step),
                    "train/hard_kd_nll": nll,
                    "train/hard_kd_ppl": _perplexity(nll),
                    "train/learning_rate": config.learning_rate,
                    "train/optimized_input_tokens": float(completed_input_tokens),
                    "train/supervised_or_weighted_tokens": float(
                        completed_supervised_tokens
                    ),
                    "timing/step_seconds": step_seconds,
                    "timing/elapsed_seconds": elapsed_seconds,
                    "cost/teacher_generation_usd": teacher_cost,
                    "cost/student_training_usd": _student_train_cost(
                        completed_input_tokens, config
                    ),
                    "cost/dev_inference_usd": development_cost,
                    "cost/cumulative_usd": total_estimate,
                },
                step=step,
            )
            if (
                step == 1
                or step % config.progress_every == 0
                or step == len(train_batches)
            ):
                progress(
                    f"step={step}/{len(train_batches)} nll={nll:.5f} "
                    f"input_tokens={completed_input_tokens}/{selected_input_tokens} "
                    f"estimated_cost=${total_estimate:.4f}"
                )
            development_due = bool(development_rows) and (
                completed_input_tokens >= next_development_token_threshold
                or step == len(train_batches)
            )
            if not development_due:
                continue
            while completed_input_tokens >= next_development_token_threshold:
                next_development_token_threshold += (
                    config.development_input_token_interval
                )
            checkpoint_name = f"{config.run_name}-step{step}"
            state_future = await training_client.save_state_async(
                checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
            )
            state_result = await state_future.result_async()
            sampler_future = await training_client.save_weights_for_sampler_async(
                checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
            )
            sampler_result = await sampler_future.result_async()
            development = None
            if development_rows:
                development = await _generation_development(
                    service_client,
                    str(sampler_result.path),
                    development_rows,
                    tinker_module,
                    config,
                    step,
                    f"step={step}",
                    progress,
                )
                development_cost += development.estimated_cost_usd
            record = CheckpointRecord(
                step=step,
                state_path=str(state_result.path),
                sampler_path=str(sampler_result.path),
                development_pass_at_1=(
                    development.metrics["eval/pass_at_1"]
                    if development is not None
                    else None
                ),
                development_pass_at_4=(
                    development.metrics["eval/pass_at_4"]
                    if development is not None
                    else None
                ),
                development_metrics=(
                    _development_metrics(development)
                    if development is not None
                    else None
                ),
            )
            checkpoints.append(record)
            checkpoint_metrics = {
                "dev/checkpoint_step": float(step),
                "dev/optimized_input_tokens": float(completed_input_tokens),
                "dev/is_initialization_policy": 0.0,
                "checkpoint/state_path": record.state_path,
                "checkpoint/sampler_path": record.sampler_path,
                "cost/teacher_generation_usd": teacher_cost,
                "cost/student_training_usd": _student_train_cost(
                    completed_input_tokens, config
                ),
                "cost/dev_inference_usd": development_cost,
                "cost/cumulative_usd": teacher_cost
                + development_cost
                + _student_train_cost(completed_input_tokens, config),
            }
            if development is not None:
                checkpoint_metrics.update(record.development_metrics or {})
                checkpoint_metrics["dev/generated_rollouts"] = float(
                    len(development.table_rows)
                )
                if hasattr(wandb_module, "Table"):
                    checkpoint_metrics["tables/development_rollouts"] = (
                        wandb_module.Table(
                            columns=list(DEVELOPMENT_TABLE_COLUMNS),
                            data=list(development.table_rows),
                        )
                    )
            _log_schema_metrics(wandb_run, config, checkpoint_metrics, step=step)

        if not checkpoints:
            selected = CheckpointRecord(
                step=len(train_batches),
                state_path=None,
                sampler_path=None,
                development_pass_at_1=None,
                development_pass_at_4=None,
                development_metrics=None,
            )
        else:
            selected = checkpoints[0]
            for candidate in checkpoints[1:]:
                if _is_better_checkpoint(candidate, selected):
                    selected = candidate
        total_cost = (
            teacher_cost
            + development_cost
            + _student_train_cost(completed_input_tokens, config)
        )
        if total_cost > config.hard_cap_usd:
            raise KDTrainingError(
                "observed token cost exceeded the configured hard cap"
            )
        report = {
            "distillation_schema_version": DISTILLATION_SCHEMA_VERSION,
            "distillation_method": asdict(method_spec(config.signal_kind)),
            "metric_schema": metric_schema_dict(config.signal_kind),
            "metric_dictionary_path": str(metric_dictionary_path),
            "mode": "remote-teacher-response-kd",
            "network_called": True,
            "run_name": config.run_name,
            "signal_kind": config.signal_kind,
            "teacher_model_id": config.teacher_model_id,
            "teacher_candidate_partitions": list(config.teacher_candidate_partitions),
            "teacher_candidate_data_policy": FULL_TRAINING_DATA_LABEL,
            "teacher_candidate_count": len(train_rows),
            "teacher_candidate_ids_hash": _candidate_ids_hash(train_rows),
            "teacher_filter": "verifier_correct_and_boxed",
            "teacher_outcomes": outcomes,
            "teacher_prompt_tokens": teacher_prompt_tokens,
            "teacher_output_tokens": teacher_output_tokens,
            "teacher_estimated_generation_usd": teacher_cost,
            "accepted_teacher_trace_digest": trace_digest,
            "accepted_teacher_trace_path": str(trace_path),
            "student_model_id": config.model_id,
            "initialization_source": config.initialization_source,
            "initialization_label": config.initialization_label,
            "parent_checkpoint": None,
            "parent_state_path": None,
            "parent_sampler_path": None,
            "reference_e4_checkpoint": E4_COMPARISON_CHECKPOINT,
            "student_selected_examples": len(prepared),
            "student_training_data_policy": "all_accepted_teacher_traces_once",
            "student_input_token_upper_bound": _student_input_token_upper_bound(
                config, manifest
            ),
            "student_selected_input_tokens": selected_input_tokens,
            "student_selected_supervised_tokens": selected_supervised_tokens,
            "student_optimized_input_tokens": completed_input_tokens,
            "student_supervised_tokens": completed_supervised_tokens,
            "training_steps": len(train_batches),
            "completed_training_steps": len(train_batches),
            "development_partition": config.development_partition,
            "development_examples": len(development_rows),
            "development_ids_hash": _development_ids_hash(config, manifest),
            "development_group_size": config.development_group_size,
            "development_input_token_interval": config.development_input_token_interval,
            "development_estimated_cost_usd": development_cost,
            "selected_checkpoint": asdict(selected),
            "selected_checkpoint_is_initialization": selected.step == 0,
            "checkpoints": [asdict(record) for record in checkpoints],
            "reference_e4_formal_pass_at_1": E4_FORMAL_PASS_AT_1,
            "reference_e4_formal_pass_at_4": E4_FORMAL_PASS_AT_4,
            "estimated_total_usd": total_cost,
            "hard_cap_usd": config.hard_cap_usd,
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        report_path = _report_path(config, str(getattr(wandb_run, "id", "run")))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        summary = {
            "schema/version": DISTILLATION_SCHEMA_VERSION,
            "schema/method": config.signal_kind,
            "data/teacher_accepted_count": len(prepared),
            "data/teacher_trace_digest": trace_digest,
            "cost/teacher_generation_usd": teacher_cost,
            "train/optimized_input_tokens": completed_input_tokens,
            "selection/selected_checkpoint_step": selected.step,
            "selection/selected_is_initialization": float(selected.step == 0),
            "cost/cumulative_usd": total_cost,
        }
        if selected.development_metrics is not None:
            selected_dev_pass_at_1 = selected.development_metrics["dev/pass_at_1"]
            selected_dev_pass_at_4 = selected.development_metrics["dev/pass_at_4"]
            summary.update(
                {
                    "dev/pass_at_1": selected_dev_pass_at_1,
                    "dev/pass_at_4": selected_dev_pass_at_4,
                    "selection/selected_dev_pass_at_1": selected_dev_pass_at_1,
                    "selection/selected_dev_pass_at_4": selected_dev_pass_at_4,
                }
            )
        if selected.state_path is not None:
            summary["checkpoint/selected_state_path"] = selected.state_path
        if selected.sampler_path is not None:
            summary["checkpoint/selected_sampler_path"] = selected.sampler_path
        wandb_run.summary.update(summary)
        progress(
            f"complete selected_step={selected.step} selected_initialization="
            f"{selected.step == 0} optimized_input_tokens={completed_input_tokens} "
            f"estimated_cost=${total_cost:.4f}"
        )
        return report
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if owned_http_client is not None:
            await owned_http_client.aclose()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or run verifier-filtered GSM8K teacher-response KD."
    )
    parser.add_argument("--run", action="store_true", help="Start the paid E9 run.")
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Acknowledge approval for the cost-gated Tinker requests.",
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--signal-kind", choices=DISTILLATION_SIGNAL_KINDS)
    parser.add_argument("--teacher-model-id")
    parser.add_argument("--teacher-temperature", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-student-steps", type=int)
    parser.add_argument("--development-examples", type=int)
    parser.add_argument("--development-input-token-interval", type=int)
    parser.add_argument("--hard-cap-usd", type=float)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--trace-output-dir")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> KDConfig:
    overrides = {
        field: value
        for field, value in {
            "attempt": args.attempt,
            "signal_kind": args.signal_kind,
            "teacher_model_id": args.teacher_model_id,
            "teacher_temperature": args.teacher_temperature,
            "learning_rate": args.learning_rate,
            "max_student_steps": args.max_student_steps,
            "development_examples": args.development_examples,
            "development_input_token_interval": (args.development_input_token_interval),
            "hard_cap_usd": args.hard_cap_usd,
            "progress_every": args.progress_every,
            "trace_output_dir": args.trace_output_dir,
        }.items()
        if value is not None
    }
    return KDConfig(**overrides)


async def _async_main(args: argparse.Namespace) -> Dict[str, Any]:
    config = _config_from_args(args)
    if args.run:
        return await run_kd_training(config, allow_paid=args.allow_paid)
    if args.allow_paid:
        raise KDTrainingError("--allow-paid requires --run")
    return build_doctor_report(config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    try:
        report = asyncio.run(_async_main(parse_args(argv)))
    except (KDTrainingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
