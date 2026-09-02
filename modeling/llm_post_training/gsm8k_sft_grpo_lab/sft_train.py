"""One-epoch, cost-gated GSM8K SFT on frozen train and validation splits."""

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
from dataclasses import asdict, dataclass, replace
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
    read_manifest,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import (
    Completion,
    evaluate_groups,
    normalize_number,
)


EXPERIMENT_ID = "e1"
E2_EXPERIMENT_ID = "e2"
E3_EXPERIMENT_ID = "e3"
LORA_RANK = 32
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
E2_LEARNING_RATE = 3e-4
MAX_SEQUENCE_TOKENS = 1024
VALIDATION_EVERY = 250
E2_VALIDATION_EVERY = 125
E2_GENERATION_MONITOR_EXAMPLES = 128
GENERATION_MONITOR_GROUP_SIZE = 4
GENERATION_MONITOR_TEMPERATURE = 1.0
LINEAR_FINAL_LR_FACTOR = 0.01
TRAINING_SEED = 20260901
PROGRESS_EVERY = 25
CHECKPOINT_TTL_SECONDS = 30 * 24 * 60 * 60
HARD_CAP_USD = 12.0
E2_HARD_CAP_USD = 18.0
E3_EARLY_STOPPING_PATIENCE = 1
E3_EARLY_STOPPING_MAX_REGRESSION = 4 / E2_GENERATION_MONITOR_EXAMPLES
TRAIN_USD_PER_MILLION = 1.463
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
OUTPUT_DIR = Path(__file__).parent / "outputs"


class SFTTrainingError(RuntimeError):
    """Raised when an SFT run would be invalid, unsafe, or incomplete."""


@dataclass(frozen=True)
class SFTConfig:
    """Frozen conditions for a GSM8K SFT experiment."""

    model_id: str = MODEL_ID
    project: str = WANDB_PROJECT
    suite_id: str = SUITE_ID
    experiment_id: str = EXPERIMENT_ID
    attempt: int = 1
    lora_rank: int = LORA_RANK
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    learning_rate_schedule: str = "constant"
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS
    validation_every: int = VALIDATION_EVERY
    generation_monitor_examples: int = 0
    generation_monitor_group_size: int = GENERATION_MONITOR_GROUP_SIZE
    checkpoint_selection: str = "validation_nll"
    early_stopping_patience: Optional[int] = None
    early_stopping_max_regression: float = 0.0
    progress_every: int = PROGRESS_EVERY
    checkpoint_ttl_seconds: int = CHECKPOINT_TTL_SECONDS
    hard_cap_usd: float = HARD_CAP_USD
    train_usd_per_million: float = TRAIN_USD_PER_MILLION

    def validate(self, manifest: Optional[SplitManifest] = None) -> None:
        if not self.model_id or not self.project or not self.suite_id:
            raise SFTTrainingError("model, project, and suite identifiers are required")
        if self.experiment_id not in {
            EXPERIMENT_ID,
            E2_EXPERIMENT_ID,
            E3_EXPERIMENT_ID,
        }:
            raise SFTTrainingError("SFT experiment ID must be e1, e2, or e3")
        positive_ints = {
            "attempt": self.attempt,
            "lora_rank": self.lora_rank,
            "batch_size": self.batch_size,
            "max_sequence_tokens": self.max_sequence_tokens,
            "validation_every": self.validation_every,
            "generation_monitor_group_size": self.generation_monitor_group_size,
            "progress_every": self.progress_every,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise SFTTrainingError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise SFTTrainingError("learning_rate must be positive")
        if self.learning_rate_schedule not in {"constant", "linear"}:
            raise SFTTrainingError("learning_rate_schedule must be constant or linear")
        if self.generation_monitor_examples < 0:
            raise SFTTrainingError("generation_monitor_examples cannot be negative")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise SFTTrainingError("early_stopping_patience must be positive")
        if self.early_stopping_max_regression < 0:
            raise SFTTrainingError("early_stopping_max_regression cannot be negative")
        if self.checkpoint_selection not in {"validation_nll", "generation_pass_at_4"}:
            raise SFTTrainingError("checkpoint_selection is not supported")
        if (
            self.checkpoint_selection == "generation_pass_at_4"
            and self.generation_monitor_examples == 0
        ):
            raise SFTTrainingError("generation selection requires a monitor")
        if (
            self.early_stopping_patience is not None
            and self.generation_monitor_examples == 0
        ):
            raise SFTTrainingError("early stopping requires a generation monitor")
        if self.hard_cap_usd <= 0 or self.train_usd_per_million <= 0:
            raise SFTTrainingError("cost settings must be positive")
        if manifest is not None:
            manifest.validate()
            if len(manifest.sft_train_ids) < self.batch_size:
                raise SFTTrainingError("sft_train must contain at least one batch")
            if self.generation_monitor_examples > len(manifest.sft_validation_ids):
                raise SFTTrainingError("generation monitor exceeds sft_validation")

    @property
    def run_name(self) -> str:
        model_slug = self.model_id.lower().replace("/", "-").replace(".", "-")
        if self.experiment_id == EXPERIMENT_ID:
            return (
                f"{self.experiment_id}-sft-{model_slug}-r{self.lora_rank}"
                f"-b{self.batch_size}-a{self.attempt:02d}"
            )
        lr_slug = f"{self.learning_rate:.0e}".replace("-0", "-")
        early_stop_slug = (
            ""
            if self.early_stopping_patience is None
            else (
                f"-es{self.early_stopping_patience}"
                f"-tol{self.early_stopping_max_regression:.4f}".replace(".", "p")
            )
        )
        return (
            f"{self.experiment_id}-sft-{model_slug}-r{self.lora_rank}"
            f"-b{self.batch_size}-lr{lr_slug}-{self.learning_rate_schedule}"
            f"-gm{self.generation_monitor_examples}{early_stop_slug}-a{self.attempt:02d}"
        )

    def training_steps(self, manifest: SplitManifest) -> int:
        self.validate(manifest)
        return math.ceil(len(manifest.sft_train_ids) / self.batch_size)

    def validation_steps(self, manifest: SplitManifest) -> Tuple[int, ...]:
        steps = self.training_steps(manifest)
        scheduled = tuple(
            range(self.validation_every, steps + 1, self.validation_every)
        )
        return (
            scheduled if scheduled and scheduled[-1] == steps else scheduled + (steps,)
        )


def e2_config() -> SFTConfig:
    """Return the generation-selected follow-up to E1."""
    return SFTConfig(
        experiment_id=E2_EXPERIMENT_ID,
        learning_rate=E2_LEARNING_RATE,
        learning_rate_schedule="linear",
        validation_every=E2_VALIDATION_EVERY,
        generation_monitor_examples=E2_GENERATION_MONITOR_EXAMPLES,
        checkpoint_selection="generation_pass_at_4",
        hard_cap_usd=E2_HARD_CAP_USD,
    )


def e3_config() -> SFTConfig:
    """Return E2's generation monitor with buffered early stopping."""
    return replace(
        e2_config(),
        experiment_id=E3_EXPERIMENT_ID,
        early_stopping_patience=E3_EARLY_STOPPING_PATIENCE,
        early_stopping_max_regression=E3_EARLY_STOPPING_MAX_REGRESSION,
    )


@dataclass(frozen=True)
class TokenizedSFTExample:
    """One raw-completion example with loss masked outside its answer."""

    example_id: str
    input_tokens: Tuple[int, ...]
    target_tokens: Tuple[int, ...]
    weights: Tuple[float, ...]
    supervised_tokens: int


@dataclass(frozen=True)
class CheckpointRecord:
    """One validation-scored remote checkpoint."""

    step: int
    nll: float
    perplexity: float
    state_path: str
    sampler_path: str
    generation_pass_at_1: Optional[float] = None
    generation_pass_at_4: Optional[float] = None


@dataclass(frozen=True)
class GenerationMonitorReport:
    """Generation metrics and token cost for one fixed validation subset."""

    metrics: Dict[str, float]
    prompt_tokens: int
    output_tokens: int
    estimated_cost_usd: float


def _print_progress(message: str) -> None:
    print(f"[gsm8k-sft] {message}", file=sys.stderr, flush=True)


def load_local_env() -> None:
    """Load ignored credentials without replacing shell values."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def estimate_max_token_cost_usd(config: SFTConfig, manifest: SplitManifest) -> float:
    """Bound training, NLL validation, and optional generation monitoring."""
    config.validate(manifest)
    train_tokens = (
        config.training_steps(manifest) * config.batch_size * config.max_sequence_tokens
    )
    validation_tokens = (
        (1 + len(config.validation_steps(manifest)))
        * len(manifest.sft_validation_ids)
        * config.max_sequence_tokens
    )
    train_and_nll_cost = (
        (train_tokens + validation_tokens) * config.train_usd_per_million / 1_000_000
    )
    monitor_runs = 1 + len(config.validation_steps(manifest))
    monitor_cost = (
        monitor_runs
        * config.generation_monitor_examples
        * config.generation_monitor_group_size
        * (
            MAX_PROMPT_TOKENS * PREFILL_USD_PER_MILLION
            + MAX_OUTPUT_TOKENS * SAMPLE_USD_PER_MILLION
        )
        / 1_000_000
    )
    return train_and_nll_cost + monitor_cost


def _generation_monitor_ids_hash(
    config: SFTConfig, manifest: SplitManifest
) -> Optional[str]:
    if not config.generation_monitor_examples:
        return None
    encoded = "\n".join(
        manifest.sft_validation_ids[: config.generation_monitor_examples]
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_doctor_report(
    config: SFTConfig,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_version: Optional[str] = None,
    wandb_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Check a complete SFT request without fetching data or training remotely."""
    manifest = read_manifest() if manifest is None else manifest
    config.validate(manifest)
    estimated_cost = estimate_max_token_cost_usd(config, manifest)
    tinker_sdk = (
        _package_version("tinker") if tinker_version is None else tinker_version
    )
    wandb_sdk = _package_version("wandb") if wandb_version is None else wandb_version
    return {
        "mode": "local-sft-preflight",
        "network_called": False,
        "run_name": config.run_name,
        "model_id": config.model_id,
        "sft_train_examples": len(manifest.sft_train_ids),
        "sft_validation_examples": len(manifest.sft_validation_ids),
        "training_steps": config.training_steps(manifest),
        "validation_steps": list(config.validation_steps(manifest)),
        "generation_monitor_examples": config.generation_monitor_examples,
        "generation_monitor_group_size": config.generation_monitor_group_size,
        "generation_monitor_ids_hash": _generation_monitor_ids_hash(config, manifest),
        "checkpoint_selection": config.checkpoint_selection,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_max_regression": config.early_stopping_max_regression,
        "learning_rate_schedule": config.learning_rate_schedule,
        "tinker_sdk_version": tinker_sdk,
        "wandb_version": wandb_sdk,
        "hf_token_configured": bool(environ.get("HF_TOKEN")),
        "tinker_api_key_configured": bool(environ.get("TINKER_API_KEY")),
        "wandb_api_key_configured": bool(environ.get("WANDB_API_KEY")),
        "estimated_max_token_cost_usd": estimated_cost,
        "hard_cap_usd": config.hard_cap_usd,
        "ready_for_paid_run": (
            sys.version_info[:2] >= (3, 11)
            and tinker_sdk is not None
            and wandb_sdk is not None
            and bool(environ.get("TINKER_API_KEY"))
            and bool(environ.get("WANDB_API_KEY"))
            and environ.get("WANDB_MODE", "").lower() != "offline"
            and estimated_cost <= config.hard_cap_usd
        ),
    }


def _authorize(
    config: SFTConfig,
    manifest: SplitManifest,
    allow_paid: bool,
    environ: Mapping[str, str],
) -> None:
    config.validate(manifest)
    if not allow_paid:
        raise SFTTrainingError("training is blocked; pass --allow-paid after approval")
    if not environ.get("TINKER_API_KEY") or not environ.get("WANDB_API_KEY"):
        raise SFTTrainingError("TINKER_API_KEY and WANDB_API_KEY are required")
    if environ.get("WANDB_MODE", "").lower() == "offline":
        raise SFTTrainingError(
            "WANDB_MODE=offline cannot produce the required dashboard"
        )
    if estimate_max_token_cost_usd(config, manifest) > config.hard_cap_usd:
        raise SFTTrainingError("estimated maximum token cost exceeds the hard cap")


def build_sft_completion(answer: str) -> str:
    """Keep GSM8K's worked solution while teaching the evaluation answer format."""
    marker = answer.rfind("####")
    if marker < 0:
        raise SFTTrainingError("a GSM8K answer is missing its final-answer marker")
    reasoning = answer[:marker].strip()
    final_answer = answer[marker + len("####") :].strip().splitlines()[0].strip()
    if normalize_number(final_answer) is None:
        raise SFTTrainingError("a GSM8K final answer is not numeric")
    return (
        f"{reasoning}\n\\boxed{{{final_answer}}}"
        if reasoning
        else f"\\boxed{{{final_answer}}}"
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    return [int(token) for token in tokens]


def tokenize_sft_example(
    row: Mapping[str, object], tokenizer: Any, max_sequence_tokens: int
) -> Optional[TokenizedSFTExample]:
    """Tokenize one example with loss applied only to the worked solution."""
    prompt_tokens = _encode(tokenizer, build_prompt(str(row["question"])))
    completion_tokens = _encode(tokenizer, build_sft_completion(str(row["answer"])))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_tokens.append(int(eos_token_id))
    full_tokens = prompt_tokens + completion_tokens
    if len(full_tokens) - 1 > max_sequence_tokens:
        return None
    input_tokens = tuple(full_tokens[:-1])
    target_tokens = tuple(full_tokens[1:])
    weights = tuple(
        [0.0] * (len(prompt_tokens) - 1)
        + [1.0] * (len(full_tokens) - len(prompt_tokens))
    )
    if not (input_tokens and len(input_tokens) == len(target_tokens) == len(weights)):
        raise SFTTrainingError("tokenized SFT tensors have inconsistent lengths")
    return TokenizedSFTExample(
        example_id=content_id(row),
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=weights,
        supervised_tokens=int(sum(weights)),
    )


def prepare_sft_examples(
    rows: Sequence[Mapping[str, object]], tokenizer: Any, max_sequence_tokens: int
) -> Tuple[TokenizedSFTExample, ...]:
    """Prepare one frozen partition, rejecting silent length-based data loss."""
    prepared = tuple(
        tokenize_sft_example(row, tokenizer, max_sequence_tokens) for row in rows
    )
    skipped = sum(item is None for item in prepared)
    if skipped:
        raise SFTTrainingError(
            f"{skipped} frozen SFT examples exceed max_sequence_tokens={max_sequence_tokens}"
        )
    return tuple(item for item in prepared if item is not None)


def _materialize_batch(
    examples: Sequence[TokenizedSFTExample], tinker_module: Any
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
    raise SFTTrainingError("Tinker did not return a readable loss diagnostic")


def _perplexity(nll: float) -> float:
    return math.exp(min(nll, 80.0))


def _batches(
    examples: Sequence[TokenizedSFTExample], batch_size: int
) -> Tuple[Tuple[TokenizedSFTExample, ...], ...]:
    return tuple(
        tuple(examples[start : start + batch_size])
        for start in range(0, len(examples), batch_size)
    )


async def _validation_nll(
    training_client: Any,
    examples: Sequence[TokenizedSFTExample],
    config: SFTConfig,
    tinker_module: Any,
    label: str,
    progress: Callable[[str], None],
) -> Tuple[float, int]:
    total_loss = 0.0
    total_supervised_tokens = 0
    input_tokens = 0
    batches = _batches(examples, config.batch_size)
    for completed, batch in enumerate(batches, start=1):
        data, batch_input_tokens, batch_supervised_tokens = _materialize_batch(
            batch, tinker_module
        )
        future = await training_client.forward_async(data=data, loss_fn="cross_entropy")
        result = await future.result_async()
        total_loss += _loss_sum(result)
        total_supervised_tokens += batch_supervised_tokens
        input_tokens += batch_input_tokens
        if completed == 1 or completed % 10 == 0 or completed == len(batches):
            partial_nll = total_loss / total_supervised_tokens
            progress(
                f"validating {label} batches={completed}/{len(batches)} "
                f"partial_nll={partial_nll:.5f}"
            )
    if total_supervised_tokens <= 0:
        raise SFTTrainingError("sft_validation has no supervised tokens")
    return total_loss / total_supervised_tokens, input_tokens


def _effective_learning_rate(config: SFTConfig, step: int, total_steps: int) -> float:
    if config.learning_rate_schedule == "constant" or total_steps <= 1:
        return config.learning_rate
    fraction = (step - 1) / (total_steps - 1)
    return config.learning_rate * max(LINEAR_FINAL_LR_FACTOR, 1.0 - fraction)


def _monitor_cost(prompt_tokens: int, output_tokens: int) -> float:
    return (
        prompt_tokens * PREFILL_USD_PER_MILLION + output_tokens * SAMPLE_USD_PER_MILLION
    ) / 1_000_000


async def _generation_monitor(
    service_client: Any,
    model_path: Optional[str],
    rows: Sequence[Mapping[str, object]],
    config: SFTConfig,
    tinker_module: Any,
    label: str,
    progress: Callable[[str], None],
) -> GenerationMonitorReport:
    """Sample a fixed held-out subset without exposing its answers to training."""
    client_kwargs = (
        {"model_path": model_path}
        if model_path is not None
        else {"base_model": config.model_id}
    )
    client = await service_client.create_sampling_client_async(**client_kwargs)
    tokenizer = client.get_tokenizer()

    async def sample_group(index: int, row: Mapping[str, object]) -> Dict[str, Any]:
        prompt_tokens = list(tokenizer.encode(build_prompt(str(row["question"]))))
        if len(prompt_tokens) > MAX_PROMPT_TOKENS:
            raise SFTTrainingError("generation monitor prompt exceeds token limit")
        result = await client.sample_async(
            prompt=tinker_module.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=config.generation_monitor_group_size,
            sampling_params=tinker_module.SamplingParams(
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=GENERATION_MONITOR_TEMPERATURE,
                seed=SEED + index,
            ),
        )
        if len(result.sequences) != config.generation_monitor_group_size:
            raise SFTTrainingError("generation monitor received the wrong group size")
        responses = tuple(
            (tokenizer.decode(sequence.tokens), len(sequence.tokens))
            for sequence in result.sequences
        )
        return {
            "index": index,
            "example_id": content_id(row),
            "ground_truth": str(row["answer"]),
            "prompt_tokens": len(prompt_tokens),
            "responses": responses,
        }

    tasks = [
        asyncio.create_task(sample_group(index, row)) for index, row in enumerate(rows)
    ]
    samples_by_index: Dict[int, Dict[str, Any]] = {}
    try:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            sample = await task
            samples_by_index[int(sample["index"])] = sample
            if completed == 1 or completed % 32 == 0 or completed == len(tasks):
                progress(
                    f"generation validation {label} prompts={completed}/{len(tasks)} "
                    f"rollouts={completed * config.generation_monitor_group_size}/"
                    f"{len(tasks) * config.generation_monitor_group_size}"
                )
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    samples = tuple(samples_by_index[index] for index in range(len(rows)))
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
    report = evaluate_groups(groups, pass_k=config.generation_monitor_group_size)
    prompt_total = sum(
        sample["prompt_tokens"] * config.generation_monitor_group_size
        for sample in samples
    )
    output_total = sum(
        output_tokens for sample in samples for _, output_tokens in sample["responses"]
    )
    return GenerationMonitorReport(
        metrics=dict(report.metrics),
        prompt_tokens=prompt_total,
        output_tokens=output_total,
        estimated_cost_usd=_monitor_cost(prompt_total, output_total),
    )


def _monitor_metrics(report: GenerationMonitorReport) -> Dict[str, float]:
    return {
        key.replace("eval/", "sft_generation_validation/"): value
        for key, value in report.metrics.items()
    }


def _is_better_generation_score(
    pass_at_1: float,
    pass_at_4: float,
    best_pass_at_1: float,
    best_pass_at_4: float,
) -> bool:
    return (pass_at_4, pass_at_1) > (best_pass_at_4, best_pass_at_1)


def _is_material_generation_regression(
    pass_at_1: float,
    pass_at_4: float,
    best_pass_at_1: float,
    best_pass_at_4: float,
    tolerance: float,
) -> bool:
    return (
        pass_at_1 < best_pass_at_1 - tolerance
        and pass_at_4 < best_pass_at_4 - tolerance
    )


def _select_checkpoint(
    checkpoints: Sequence[CheckpointRecord], config: SFTConfig
) -> CheckpointRecord:
    if config.checkpoint_selection == "validation_nll":
        return min(checkpoints, key=lambda record: record.nll)
    if any(record.generation_pass_at_4 is None for record in checkpoints):
        raise SFTTrainingError("generation monitor is missing at a checkpoint")
    return min(
        checkpoints,
        key=lambda record: (
            -float(record.generation_pass_at_4),
            -float(record.generation_pass_at_1),
            record.nll,
        ),
    )


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


def _tracking_config(config: SFTConfig, manifest: SplitManifest) -> Dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "attempt": config.attempt,
        "suite_id": config.suite_id,
        "checkpoint": "base",
        "parent_checkpoint": "base",
        "model_id": config.model_id,
        "dataset_id": manifest.dataset_id,
        "dataset_revision": manifest.dataset_revision,
        "manifest_hash": manifest.manifest_hash,
        "sft_train_examples": len(manifest.sft_train_ids),
        "sft_validation_examples": len(manifest.sft_validation_ids),
        "prompt_version": PROMPT_VERSION,
        "target_format": "gsm8k-worked-solution-boxed-v1",
        "lora_rank": config.lora_rank,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "learning_rate_schedule": config.learning_rate_schedule,
        "linear_final_lr_factor": LINEAR_FINAL_LR_FACTOR,
        "training_seed": TRAINING_SEED,
        "max_sequence_tokens": config.max_sequence_tokens,
        "training_steps": config.training_steps(manifest),
        "validation_steps": list(config.validation_steps(manifest)),
        "generation_monitor_examples": config.generation_monitor_examples,
        "generation_monitor_group_size": config.generation_monitor_group_size,
        "generation_monitor_temperature": GENERATION_MONITOR_TEMPERATURE,
        "generation_monitor_max_prompt_tokens": MAX_PROMPT_TOKENS,
        "generation_monitor_max_output_tokens": MAX_OUTPUT_TOKENS,
        "generation_monitor_ids_hash": _generation_monitor_ids_hash(config, manifest),
        "checkpoint_selection": config.checkpoint_selection,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_max_regression": config.early_stopping_max_regression,
        "train_usd_per_million": config.train_usd_per_million,
        "hard_cap_usd": config.hard_cap_usd,
        "git_sha": _git_sha(),
        "hypothesis": (
            "One clean SFT epoch improves answer format and GSM8K accuracy."
            if config.experiment_id == EXPERIMENT_ID
            else "Generation-selected SFT avoids the E1 formal regression."
        ),
        "expected_failure": (
            "Validation NLL improves without formal generation gain."
            if config.experiment_id == EXPERIMENT_ID
            else "Generation monitoring regresses despite a lower learning rate."
        ),
    }


def _estimated_cost(processed_tokens: int, config: SFTConfig) -> float:
    return processed_tokens * config.train_usd_per_million / 1_000_000


def _total_estimated_cost(
    processed_tokens: int, generation_monitor_cost: float, config: SFTConfig
) -> float:
    return _estimated_cost(processed_tokens, config) + generation_monitor_cost


def _report_path(experiment_id: str, run_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id)
    return OUTPUT_DIR / f"{experiment_id}_sft_report_{safe_id}.json"


async def run_sft_training(
    config: SFTConfig,
    allow_paid: bool,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
    train_rows: Optional[Sequence[Mapping[str, object]]] = None,
    validation_rows: Optional[Sequence[Mapping[str, object]]] = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = _print_progress,
) -> Dict[str, Any]:
    """Run one frozen SFT epoch and select its configured best checkpoint."""
    manifest = read_manifest() if manifest is None else manifest
    _authorize(config, manifest, allow_paid, environ)
    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise SFTTrainingError(
                "Tinker SDK is unavailable; run with `uv run --extra tinker`"
            ) from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise SFTTrainingError("Weights & Biases is unavailable") from exc
    if train_rows is None or validation_rows is None:
        progress("loading frozen sft_train and sft_validation rows")
        train_rows = load_official_train_rows(manifest, "sft_train")
        validation_rows = load_official_train_rows(manifest, "sft_validation")
    if len(train_rows) != len(manifest.sft_train_ids):
        raise SFTTrainingError("loaded sft_train rows do not match the manifest")
    if len(validation_rows) != len(manifest.sft_validation_ids):
        raise SFTTrainingError("loaded sft_validation rows do not match the manifest")

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
            f"authorized run={config.run_name} train={len(train_rows)} "
            f"validation={len(validation_rows)} steps={config.training_steps(manifest)} "
            f"max_cost=${estimate_max_token_cost_usd(config, manifest):.4f}"
        )
        progress("initializing Tinker LoRA training client")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_id,
            rank=config.lora_rank,
            seed=TRAINING_SEED,
            user_metadata={"experiment_id": config.experiment_id},
        )
        tokenizer = training_client.get_tokenizer()
        train = prepare_sft_examples(train_rows, tokenizer, config.max_sequence_tokens)
        validation = prepare_sft_examples(
            validation_rows, tokenizer, config.max_sequence_tokens
        )
        monitor_rows = validation_rows[: config.generation_monitor_examples]
        progress(
            f"data ready train={len(train)} validation={len(validation)} "
            f"generation_monitor={len(monitor_rows)}"
        )
        wandb_run = wandb_module.init(
            project=config.project,
            entity=environ.get("WANDB_ENTITY") or None,
            name=config.run_name,
            group=config.suite_id,
            job_type="sft-training",
            tags=["gsm8k", "sft", config.experiment_id, f"rank-{config.lora_rank}"],
            config=_tracking_config(config, manifest),
        )
        progress(f"started W&B run={getattr(wandb_run, 'url', None)}")

        processed_tokens = 0
        generation_monitor_cost = 0.0
        baseline_generation: Optional[GenerationMonitorReport] = None
        best_generation_pass_at_1: Optional[float] = None
        best_generation_pass_at_4: Optional[float] = None
        generation_regression_streak = 0
        early_stopping_triggered = False
        early_stopping_step: Optional[int] = None
        elapsed_started_at = clock()
        progress("validating untrained adapter")
        baseline_nll, baseline_tokens = await _validation_nll(
            training_client,
            validation,
            config,
            tinker_module,
            label=f"step=0/{config.training_steps(manifest)}",
            progress=progress,
        )
        processed_tokens += baseline_tokens
        wandb_run.log(
            {
                "sft_validation/nll": baseline_nll,
                "sft_validation/perplexity": _perplexity(baseline_nll),
                "sft_validation/is_baseline": 1.0,
                "run_stats/cumulative_processed_tokens": float(processed_tokens),
                "run_stats/estimated_cumulative_usd": _total_estimated_cost(
                    processed_tokens, generation_monitor_cost, config
                ),
            },
            step=0,
        )
        progress(
            f"validation step=0/{config.training_steps(manifest)} "
            f"nll={baseline_nll:.5f} perplexity={_perplexity(baseline_nll):.3f}"
        )
        if monitor_rows:
            progress("generation validating Base model")
            baseline_generation = await _generation_monitor(
                service_client,
                None,
                monitor_rows,
                config,
                tinker_module,
                label=f"step=0/{config.training_steps(manifest)}",
                progress=progress,
            )
            generation_monitor_cost += baseline_generation.estimated_cost_usd
            wandb_run.log(
                {
                    **_monitor_metrics(baseline_generation),
                    "sft_generation_validation/is_baseline": 1.0,
                    "sft_generation_validation/estimated_cumulative_usd": (
                        generation_monitor_cost
                    ),
                    "run_stats/estimated_cumulative_usd": _total_estimated_cost(
                        processed_tokens, generation_monitor_cost, config
                    ),
                },
                step=0,
            )
            progress(
                f"generation validation step=0/{config.training_steps(manifest)} "
                f"pass_at_1={baseline_generation.metrics['eval/pass_at_1']:.4f} "
                f"pass_at_4={baseline_generation.metrics['eval/pass_at_4']:.4f} "
                f"estimated_cost=${generation_monitor_cost:.4f}"
            )
            best_generation_pass_at_1 = baseline_generation.metrics["eval/pass_at_1"]
            best_generation_pass_at_4 = baseline_generation.metrics["eval/pass_at_4"]

        checkpoints: list[CheckpointRecord] = []
        validation_steps = set(config.validation_steps(manifest))
        train_batches = _batches(train, config.batch_size)
        total_steps = len(train_batches)
        completed_training_steps = 0
        examples_seen = 0
        for step, batch in enumerate(train_batches, start=1):
            data, batch_tokens, supervised_tokens = _materialize_batch(
                batch, tinker_module
            )
            step_started_at = clock()
            forward_backward = await training_client.forward_backward_async(
                data=data, loss_fn="cross_entropy"
            )
            forward_backward_result = await forward_backward.result_async()
            current_learning_rate = _effective_learning_rate(config, step, total_steps)
            optimizer = await training_client.optim_step_async(
                tinker_module.types.AdamParams(learning_rate=current_learning_rate)
            )
            await optimizer.result_async()
            processed_tokens += batch_tokens
            completed_training_steps = step
            examples_seen += len(batch)
            nll = _loss_sum(forward_backward_result) / supervised_tokens
            step_seconds = max(clock() - step_started_at, 1e-9)
            elapsed_seconds = clock() - elapsed_started_at
            steps_per_second = step / max(elapsed_seconds, 1e-9)
            eta_seconds = (total_steps - step) / steps_per_second
            metrics = {
                "train/nll": nll,
                "train/perplexity": _perplexity(nll),
                "train/learning_rate": current_learning_rate,
                "train/supervised_tokens": float(supervised_tokens),
                "train/examples_seen": float(examples_seen),
                "train/fraction_examples_seen": examples_seen / len(train),
                "train/tokens_per_second": batch_tokens / step_seconds,
                "timing/step_seconds": step_seconds,
                "timing/elapsed_seconds": elapsed_seconds,
                "timing/eta_seconds": eta_seconds,
                "run_stats/cumulative_processed_tokens": float(processed_tokens),
                "run_stats/estimated_cumulative_usd": _total_estimated_cost(
                    processed_tokens, generation_monitor_cost, config
                ),
            }
            wandb_run.log(metrics, step=step)
            if step == 1 or step % config.progress_every == 0 or step == total_steps:
                progress(
                    f"step={step}/{total_steps} nll={nll:.5f} "
                    f"perplexity={_perplexity(nll):.3f} lr={current_learning_rate:.2g} "
                    f"throughput={batch_tokens / step_seconds:.1f}tok/s "
                    f"elapsed={elapsed_seconds:.1f}s eta={eta_seconds:.1f}s "
                    f"estimated_cost=${_total_estimated_cost(processed_tokens, generation_monitor_cost, config):.4f}"
                )
            if step not in validation_steps:
                continue

            progress(f"validating checkpoint step={step}/{total_steps}")
            validation_nll, validation_tokens = await _validation_nll(
                training_client,
                validation,
                config,
                tinker_module,
                label=f"step={step}/{total_steps}",
                progress=progress,
            )
            processed_tokens += validation_tokens
            validation_ppl = _perplexity(validation_nll)
            checkpoint_name = f"{config.run_name}-step{step}"
            state_future = await training_client.save_state_async(
                checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
            )
            state_result = await state_future.result_async()
            sampler_future = await training_client.save_weights_for_sampler_async(
                checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
            )
            sampler_result = await sampler_future.result_async()
            generation: Optional[GenerationMonitorReport] = None
            if monitor_rows:
                progress(
                    f"generation validating checkpoint step={step}/{total_steps} "
                    f"prompts={len(monitor_rows)}"
                )
                generation = await _generation_monitor(
                    service_client,
                    str(sampler_result.path),
                    monitor_rows,
                    config,
                    tinker_module,
                    label=f"step={step}/{total_steps}",
                    progress=progress,
                )
                generation_monitor_cost += generation.estimated_cost_usd
            record = CheckpointRecord(
                step=step,
                nll=validation_nll,
                perplexity=validation_ppl,
                state_path=str(state_result.path),
                sampler_path=str(sampler_result.path),
                generation_pass_at_1=(
                    generation.metrics["eval/pass_at_1"] if generation else None
                ),
                generation_pass_at_4=(
                    generation.metrics["eval/pass_at_4"] if generation else None
                ),
            )
            checkpoints.append(record)
            should_stop = False
            early_stopping_metrics: Dict[str, float] = {}
            checkpoint_metrics = {
                "sft_validation/nll": validation_nll,
                "sft_validation/perplexity": validation_ppl,
                "sft_validation/is_baseline": 0.0,
                "sft_validation/nll_delta_from_base": validation_nll - baseline_nll,
                "run_stats/cumulative_processed_tokens": float(processed_tokens),
                "run_stats/estimated_cumulative_usd": _total_estimated_cost(
                    processed_tokens, generation_monitor_cost, config
                ),
            }
            if generation is not None:
                generation_pass_at_1 = generation.metrics["eval/pass_at_1"]
                generation_pass_at_4 = generation.metrics["eval/pass_at_4"]
                checkpoint_metrics.update(_monitor_metrics(generation))
                checkpoint_metrics.update(
                    {
                        "sft_generation_validation/is_baseline": 0.0,
                        "sft_generation_validation/estimated_cumulative_usd": (
                            generation_monitor_cost
                        ),
                    }
                )
                if baseline_generation is not None:
                    checkpoint_metrics.update(
                        {
                            "sft_generation_validation/pass_at_1_delta_from_base": (
                                generation_pass_at_1
                                - baseline_generation.metrics["eval/pass_at_1"]
                            ),
                            "sft_generation_validation/pass_at_4_delta_from_base": (
                                generation_pass_at_4
                                - baseline_generation.metrics["eval/pass_at_4"]
                            ),
                        }
                    )
                if config.early_stopping_patience is not None:
                    if (
                        best_generation_pass_at_1 is None
                        or best_generation_pass_at_4 is None
                    ):
                        raise SFTTrainingError(
                            "early stopping requires a baseline generation score"
                        )
                    pass_at_1_delta = generation_pass_at_1 - best_generation_pass_at_1
                    pass_at_4_delta = generation_pass_at_4 - best_generation_pass_at_4
                    material_regression = _is_material_generation_regression(
                        generation_pass_at_1,
                        generation_pass_at_4,
                        best_generation_pass_at_1,
                        best_generation_pass_at_4,
                        config.early_stopping_max_regression,
                    )
                    generation_regression_streak = (
                        generation_regression_streak + 1 if material_regression else 0
                    )
                    if _is_better_generation_score(
                        generation_pass_at_1,
                        generation_pass_at_4,
                        best_generation_pass_at_1,
                        best_generation_pass_at_4,
                    ):
                        best_generation_pass_at_1 = generation_pass_at_1
                        best_generation_pass_at_4 = generation_pass_at_4
                    should_stop = (
                        generation_regression_streak >= config.early_stopping_patience
                    )
                    early_stopping_metrics = {
                        "early_stopping/best_pass_at_1": best_generation_pass_at_1,
                        "early_stopping/best_pass_at_4": best_generation_pass_at_4,
                        "early_stopping/pass_at_1_delta_from_best": pass_at_1_delta,
                        "early_stopping/pass_at_4_delta_from_best": pass_at_4_delta,
                        "early_stopping/material_regression": float(
                            material_regression
                        ),
                        "early_stopping/regression_streak": float(
                            generation_regression_streak
                        ),
                        "early_stopping/stop_triggered": float(should_stop),
                    }
                    checkpoint_metrics.update(early_stopping_metrics)
            wandb_run.log(checkpoint_metrics, step=step)
            progress(
                f"validation step={step}/{total_steps} nll={validation_nll:.5f} "
                f"perplexity={validation_ppl:.3f} "
                f"estimated_cost=${_total_estimated_cost(processed_tokens, generation_monitor_cost, config):.4f}"
            )
            if generation is not None:
                progress(
                    f"generation validation step={step}/{total_steps} "
                    f"pass_at_1={generation.metrics['eval/pass_at_1']:.4f} "
                    f"pass_at_4={generation.metrics['eval/pass_at_4']:.4f} "
                    f"monitor_cost=${generation_monitor_cost:.4f}"
                )
            if config.early_stopping_patience is not None:
                progress(
                    f"early stopping step={step}/{total_steps} "
                    f"decision={'stop' if should_stop else 'continue'} "
                    f"streak={generation_regression_streak}/"
                    f"{config.early_stopping_patience} "
                    f"best_pass_at_1={best_generation_pass_at_1:.4f} "
                    f"best_pass_at_4={best_generation_pass_at_4:.4f}"
                )
            if should_stop:
                early_stopping_triggered = True
                early_stopping_step = step
                break

        if not checkpoints:
            raise SFTTrainingError("training completed without a validation checkpoint")
        selected = _select_checkpoint(checkpoints, config)
        total_estimated_cost = _total_estimated_cost(
            processed_tokens, generation_monitor_cost, config
        )
        if total_estimated_cost > config.hard_cap_usd:
            raise SFTTrainingError(
                "observed token cost exceeded the configured hard cap"
            )
        report = {
            "mode": "remote-sft-training",
            "network_called": True,
            "run_name": config.run_name,
            "model_id": config.model_id,
            "manifest_hash": manifest.manifest_hash,
            "sft_train_examples": len(train),
            "sft_validation_examples": len(validation),
            "training_steps": total_steps,
            "completed_training_steps": completed_training_steps,
            "completed_examples_seen": examples_seen,
            "baseline_validation_nll": baseline_nll,
            "baseline_validation_perplexity": _perplexity(baseline_nll),
            "baseline_generation_monitor": (
                asdict(baseline_generation) if baseline_generation else None
            ),
            "selected_checkpoint": asdict(selected),
            "validation_checkpoints": [asdict(record) for record in checkpoints],
            "generation_monitor_estimated_cost_usd": generation_monitor_cost,
            "early_stopping": {
                "enabled": config.early_stopping_patience is not None,
                "patience": config.early_stopping_patience,
                "max_regression": config.early_stopping_max_regression,
                "triggered": early_stopping_triggered,
                "step": early_stopping_step,
                "final_regression_streak": generation_regression_streak,
            },
            "estimated_token_cost_usd": total_estimated_cost,
            "hard_cap_usd": config.hard_cap_usd,
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _report_path(config.experiment_id, str(getattr(wandb_run, "id", "run")))
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        wandb_run.summary.update(
            {
                "sft_validation/baseline_nll": baseline_nll,
                "sft_validation/best_nll": selected.nll,
                "sft_validation/best_perplexity": selected.perplexity,
                "checkpoint/selection_metric": config.checkpoint_selection,
                "checkpoint/selected_step": selected.step,
                "checkpoint/selected_state_path": selected.state_path,
                "checkpoint/selected_sampler_path": selected.sampler_path,
                "checkpoint/selected_generation_pass_at_1": (
                    selected.generation_pass_at_1
                ),
                "checkpoint/selected_generation_pass_at_4": (
                    selected.generation_pass_at_4
                ),
                "early_stopping/enabled": float(
                    config.early_stopping_patience is not None
                ),
                "early_stopping/triggered": float(early_stopping_triggered),
                "early_stopping/step": early_stopping_step,
                "early_stopping/final_regression_streak": generation_regression_streak,
                "run_stats/completed_training_steps": completed_training_steps,
                "run_stats/completed_examples_seen": examples_seen,
                "run_stats/estimated_total_usd": total_estimated_cost,
            }
        )
        progress(
            f"complete selected_step={selected.step} selection={config.checkpoint_selection} "
            f"completed_steps={completed_training_steps}/{total_steps} "
            f"best_validation_nll={selected.nll:.5f} "
            f"estimated_cost=${total_estimated_cost:.4f}"
        )
        return report
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if owned_http_client is not None:
            await owned_http_client.aclose()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or run a frozen GSM8K SFT experiment."
    )
    parser.add_argument("--run", action="store_true", help="Start the paid SFT run.")
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Acknowledge approval for the cost-gated Tinker request.",
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--recipe", choices=("e1", "e2", "e3"), default="e1")
    parser.add_argument("--hard-cap-usd", type=float)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-max-regression", type=float)
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> Dict[str, Any]:
    base_config = (
        e2_config()
        if args.recipe == E2_EXPERIMENT_ID
        else e3_config()
        if args.recipe == E3_EXPERIMENT_ID
        else SFTConfig()
    )
    config = replace(
        base_config,
        attempt=args.attempt,
        hard_cap_usd=(
            base_config.hard_cap_usd if args.hard_cap_usd is None else args.hard_cap_usd
        ),
        progress_every=args.progress_every,
        early_stopping_patience=(
            base_config.early_stopping_patience
            if args.early_stopping_patience is None
            else args.early_stopping_patience
        ),
        early_stopping_max_regression=(
            base_config.early_stopping_max_regression
            if args.early_stopping_max_regression is None
            else args.early_stopping_max_regression
        ),
    )
    if args.run:
        return await run_sft_training(config, allow_paid=args.allow_paid)
    if args.allow_paid:
        raise SFTTrainingError("--allow-paid requires --run")
    return build_doctor_report(config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    try:
        report = asyncio.run(_async_main(parse_args(argv)))
    except (SFTTrainingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
