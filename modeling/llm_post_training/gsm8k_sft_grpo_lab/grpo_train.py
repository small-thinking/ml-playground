"""Cost-gated GRPO on the frozen GSM8K RL split."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
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
    read_manifest,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import (
    Completion,
    evaluate_groups,
    score_completion,
)


EXPERIMENT_ID = "e4"
GRPO_EXPERIMENT_IDS = ("e4", "e5")
PARENT_CHECKPOINT = "e2-sft-qwen-qwen3-5-9b-base-r32-b8-lr3e-4-linear-gm128-a01-step250"
PARENT_STATE_PATH = (
    "tinker://5048e951-841f-53d9-9388-87cb865de0bb:train:0/weights/"
    "e2-sft-qwen-qwen3-5-9b-base-r32-b8-lr3e-4-linear-gm128-a01-step250"
)
PARENT_SAMPLER_PATH = (
    "tinker://5048e951-841f-53d9-9388-87cb865de0bb:train:0/sampler_weights/"
    "e2-sft-qwen-qwen3-5-9b-base-r32-b8-lr3e-4-linear-gm128-a01-step250"
)
LORA_RANK = 32
DEFAULT_STEPS = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_GROUP_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MONITOR_EXAMPLES = 64
DEFAULT_CHECKPOINT_EVERY = 25
DEFAULT_PROGRESS_EVERY = 5
DEFAULT_HARD_CAP_USD = 12.0
TRAIN_USD_PER_MILLION = 1.463
CHECKPOINT_TTL_SECONDS = 30 * 24 * 60 * 60
REWARD_VERSION = "gsm8k-final-answer-binary-v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
OUTPUT_DIR = Path(__file__).parent / "outputs"

MONITOR_COLUMNS = (
    "example_id",
    "group_id",
    "rollout_id",
    "question",
    "ground_truth",
    "generated_response",
    "parsed_answer",
    "correct",
    "format_valid",
    "output_tokens",
    "truncated",
    "reward",
    "group_mean_reward",
    "advantage",
)


class GRPOTrainingError(RuntimeError):
    """Raised when a GRPO run would be invalid, unsafe, or incomplete."""


@dataclass(frozen=True)
class GRPOConfig:
    """All conditions for one comparable GRPO training attempt."""

    model_id: str = MODEL_ID
    project: str = WANDB_PROJECT
    suite_id: str = SUITE_ID
    experiment_id: str = EXPERIMENT_ID
    attempt: int = 1
    init_source: str = "sft"
    initialization_label: Optional[str] = None
    parent_checkpoint: str = PARENT_CHECKPOINT
    parent_state_path: str = PARENT_STATE_PATH
    parent_sampler_path: str = PARENT_SAMPLER_PATH
    lora_rank: int = LORA_RANK
    steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    group_size: int = DEFAULT_GROUP_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    temperature: float = 1.0
    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    monitor_examples: int = DEFAULT_MONITOR_EXAMPLES
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    min_effective_groups: int = 0
    max_resample_rounds: int = 0
    early_stopping_patience: Optional[int] = None
    early_stopping_max_regression: float = 0.0
    progress_every: int = DEFAULT_PROGRESS_EVERY
    checkpoint_ttl_seconds: int = CHECKPOINT_TTL_SECONDS
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD
    train_usd_per_million: float = TRAIN_USD_PER_MILLION
    prefill_usd_per_million: float = PREFILL_USD_PER_MILLION
    sample_usd_per_million: float = SAMPLE_USD_PER_MILLION
    seed: int = SEED

    def validate(self, manifest: Optional[SplitManifest] = None) -> None:
        if self.experiment_id not in GRPO_EXPERIMENT_IDS:
            raise GRPOTrainingError("GRPO experiment_id must be e4 or e5")
        if not self.model_id or not self.project or not self.suite_id:
            raise GRPOTrainingError(
                "model, project, and suite identifiers are required"
            )
        if self.init_source not in {"sft", "base"}:
            raise GRPOTrainingError("init_source must be sft or base")
        if self.init_source == "sft":
            if "/weights/" not in self.parent_state_path:
                raise GRPOTrainingError(
                    "parent_state_path must be a Tinker training-state URI"
                )
            if "/sampler_weights/" not in self.parent_sampler_path:
                raise GRPOTrainingError(
                    "parent_sampler_path must be a Tinker sampler URI"
                )
        positive_ints = {
            "attempt": self.attempt,
            "lora_rank": self.lora_rank,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_output_tokens": self.max_output_tokens,
            "checkpoint_every": self.checkpoint_every,
            "progress_every": self.progress_every,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise GRPOTrainingError(f"{name} must be positive")
        if self.group_size < 4:
            raise GRPOTrainingError(
                "group_size must be at least four for pass@4 monitoring"
            )
        if min(self.learning_rate, self.temperature, self.hard_cap_usd) <= 0:
            raise GRPOTrainingError(
                "learning rate, temperature, and hard cap must be positive"
            )
        if (
            min(
                self.train_usd_per_million,
                self.prefill_usd_per_million,
                self.sample_usd_per_million,
            )
            <= 0
        ):
            raise GRPOTrainingError("token pricing inputs must be positive")
        if self.monitor_examples < 0:
            raise GRPOTrainingError("monitor_examples cannot be negative")
        if not 0 <= self.min_effective_groups <= self.batch_size:
            raise GRPOTrainingError(
                "min_effective_groups must be between zero and batch_size"
            )
        if self.max_resample_rounds < 0:
            raise GRPOTrainingError("max_resample_rounds cannot be negative")
        if self.min_effective_groups == 0 and self.max_resample_rounds:
            raise GRPOTrainingError(
                "max_resample_rounds requires min_effective_groups"
            )
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise GRPOTrainingError("early_stopping_patience must be positive")
            if not self.monitor_examples:
                raise GRPOTrainingError("early stopping requires rl_monitor examples")
        if self.early_stopping_max_regression < 0:
            raise GRPOTrainingError(
                "early_stopping_max_regression cannot be negative"
            )
        if manifest is not None:
            manifest.validate()
            if self.batch_size > len(manifest.rl_train_ids):
                raise GRPOTrainingError("batch_size exceeds the frozen rl_train split")
            if self.monitor_examples > len(manifest.rl_monitor_ids):
                raise GRPOTrainingError(
                    "monitor_examples exceeds the frozen rl_monitor split"
                )

    @property
    def run_name(self) -> str:
        model_slug = self.model_id.lower().replace("/", "-").replace(".", "-")
        lr_slug = f"{self.learning_rate:.0e}".replace("-0", "-")
        source_label = self.initialization_label or (
            "base" if self.init_source == "base" else "e2s250"
        )
        source_slug = re.sub(r"[^a-zA-Z0-9]+", "-", source_label).strip("-").lower()
        name = (
            f"{self.experiment_id}-grpo-{model_slug}-r{self.lora_rank}"
            f"-b{self.batch_size}-g{self.group_size}-lr{lr_slug}"
            f"-from-{source_slug}-s{self.steps}-m{self.monitor_examples}-a{self.attempt:02d}"
        )
        if self.min_effective_groups:
            name += f"-sig{self.min_effective_groups}x{self.max_resample_rounds + 1}"
        if self.early_stopping_patience is not None:
            name += f"-es{self.early_stopping_patience}"
        return name

    def checkpoint_steps(self) -> Tuple[int, ...]:
        scheduled = tuple(
            range(self.checkpoint_every, self.steps + 1, self.checkpoint_every)
        )
        return (
            scheduled
            if scheduled and scheduled[-1] == self.steps
            else scheduled + (self.steps,)
        )

    @property
    def max_candidate_groups_per_step(self) -> int:
        return self.batch_size * (1 + self.max_resample_rounds)


@dataclass(frozen=True)
class Rollout:
    """One sampled completion and its immutable verifier outcome."""

    tokens: Tuple[int, ...]
    logprobs: Optional[Tuple[float, ...]]
    response: str
    scored: Any


@dataclass(frozen=True)
class RolloutGroup:
    """All rollouts for one prompt, kept together for group-relative rewards."""

    example_id: str
    question: str
    ground_truth: str
    prompt_tokens: Tuple[int, ...]
    rollouts: Tuple[Rollout, ...]


@dataclass(frozen=True)
class MonitorReport:
    """Metrics and bounded token accounting for one held-out RL monitor pass."""

    metrics: Dict[str, float]
    prompt_tokens: int
    output_tokens: int
    table_rows: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class CheckpointRecord:
    """A persisted policy and the monitor score used to select it."""

    step: int
    state_path: Optional[str]
    sampler_path: Optional[str]
    monitor_pass_at_1: Optional[float]
    monitor_pass_at_4: Optional[float]


def _print_progress(message: str) -> None:
    print(f"[gsm8k-grpo] {message}", file=sys.stderr, flush=True)


def load_local_env() -> None:
    """Load ignored credentials without replacing shell values."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _sampling_cost(prompt_tokens: int, output_tokens: int, config: GRPOConfig) -> float:
    return (
        prompt_tokens * config.prefill_usd_per_million
        + output_tokens * config.sample_usd_per_million
    ) / 1_000_000


def _training_cost(input_tokens: int, config: GRPOConfig) -> float:
    return input_tokens * config.train_usd_per_million / 1_000_000


def estimate_max_token_cost_usd(config: GRPOConfig, manifest: SplitManifest) -> float:
    """Bound rollout, optimization, and held-out monitor tokens before a run."""
    config.validate(manifest)
    training_rollouts = (
        config.steps * config.max_candidate_groups_per_step * config.group_size
    )
    sample_cost = _sampling_cost(
        training_rollouts * config.max_prompt_tokens,
        training_rollouts * config.max_output_tokens,
        config,
    )
    optimization_cost = _training_cost(
        training_rollouts * (config.max_prompt_tokens + config.max_output_tokens),
        config,
    )
    monitor_runs = 1 + len(config.checkpoint_steps()) if config.monitor_examples else 0
    monitor_rollouts = monitor_runs * config.monitor_examples * config.group_size
    monitor_cost = _sampling_cost(
        monitor_rollouts * config.max_prompt_tokens,
        monitor_rollouts * config.max_output_tokens,
        config,
    )
    return sample_cost + optimization_cost + monitor_cost


def _monitor_ids_hash(config: GRPOConfig, manifest: SplitManifest) -> Optional[str]:
    if not config.monitor_examples:
        return None
    return hashlib.sha256(
        "\n".join(manifest.rl_monitor_ids[: config.monitor_examples]).encode()
    ).hexdigest()


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


def _tracking_config(config: GRPOConfig, manifest: SplitManifest) -> Dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "attempt": config.attempt,
        "suite_id": config.suite_id,
        "model_id": config.model_id,
        "initialization_source": config.init_source,
        "initialization_label": config.initialization_label
        or ("base" if config.init_source == "base" else "e2s250"),
        "parent_checkpoint": config.parent_checkpoint
        if config.init_source == "sft"
        else None,
        "parent_state_path": config.parent_state_path
        if config.init_source == "sft"
        else None,
        "parent_sampler_path": (
            config.parent_sampler_path if config.init_source == "sft" else None
        ),
        "dataset_id": manifest.dataset_id,
        "dataset_revision": manifest.dataset_revision,
        "manifest_hash": manifest.manifest_hash,
        "rl_train_examples": len(manifest.rl_train_ids),
        "rl_monitor_examples": config.monitor_examples,
        "rl_monitor_ids_hash": _monitor_ids_hash(config, manifest),
        "prompt_version": PROMPT_VERSION,
        "reward_version": REWARD_VERSION,
        "reward_definition": "1.0 for exact final-answer match, otherwise 0.0",
        "lora_rank": config.lora_rank,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "learning_rate": config.learning_rate,
        "temperature": config.temperature,
        "max_prompt_tokens": config.max_prompt_tokens,
        "max_output_tokens": config.max_output_tokens,
        "checkpoint_steps": list(config.checkpoint_steps()),
        "min_effective_groups": config.min_effective_groups,
        "max_resample_rounds": config.max_resample_rounds,
        "max_candidate_groups_per_step": config.max_candidate_groups_per_step,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_max_regression": config.early_stopping_max_regression,
        "checkpoint_ttl_seconds": config.checkpoint_ttl_seconds,
        "seed": config.seed,
        "hard_cap_usd": config.hard_cap_usd,
        "train_usd_per_million": config.train_usd_per_million,
        "prefill_usd_per_million": config.prefill_usd_per_million,
        "sample_usd_per_million": config.sample_usd_per_million,
        "git_sha": _git_sha(),
        "hypothesis": (
            "On-policy binary-answer GRPO improves held-out RL-monitor pass@4 "
            f"from the {config.init_source} initialization policy."
        ),
        "expected_failure": (
            "Too many degenerate groups leave too little learning signal, or "
            "monitor pass metrics regress despite rising training reward."
        ),
    }


def build_doctor_report(
    config: GRPOConfig,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_version: Optional[str] = None,
    wandb_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the whole paid plan without data downloads or remote calls."""
    manifest = read_manifest() if manifest is None else manifest
    config.validate(manifest)
    estimated_cost = estimate_max_token_cost_usd(config, manifest)
    tinker_sdk = (
        _package_version("tinker") if tinker_version is None else tinker_version
    )
    wandb_sdk = _package_version("wandb") if wandb_version is None else wandb_version
    monitor_runs = 1 + len(config.checkpoint_steps()) if config.monitor_examples else 0
    return {
        "mode": "local-grpo-preflight",
        "network_called": False,
        "run_name": config.run_name,
        "model_id": config.model_id,
        "initialization_source": config.init_source,
        "initialization_label": config.initialization_label
        or ("base" if config.init_source == "base" else "e2s250"),
        "parent_checkpoint": config.parent_checkpoint
        if config.init_source == "sft"
        else None,
        "parent_state_path": config.parent_state_path
        if config.init_source == "sft"
        else None,
        "parent_sampler_path": (
            config.parent_sampler_path if config.init_source == "sft" else None
        ),
        "manifest_hash": manifest.manifest_hash,
        "rl_train_examples": len(manifest.rl_train_ids),
        "rl_monitor_examples": config.monitor_examples,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "training_rollouts": config.steps * config.batch_size * config.group_size,
        "max_training_rollouts": (
            config.steps * config.max_candidate_groups_per_step * config.group_size
        ),
        "min_effective_groups": config.min_effective_groups,
        "max_resample_rounds": config.max_resample_rounds,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_max_regression": config.early_stopping_max_regression,
        "checkpoint_steps": list(config.checkpoint_steps()),
        "monitor_runs": monitor_runs,
        "estimated_max_token_cost_usd": estimated_cost,
        "hard_cap_usd": config.hard_cap_usd,
        "tinker_sdk_version": tinker_sdk,
        "wandb_version": wandb_sdk,
        "tinker_api_key_configured": bool(environ.get("TINKER_API_KEY")),
        "wandb_api_key_configured": bool(environ.get("WANDB_API_KEY")),
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
    config: GRPOConfig,
    manifest: SplitManifest,
    allow_paid: bool,
    environ: Mapping[str, str],
) -> None:
    config.validate(manifest)
    if not allow_paid:
        raise GRPOTrainingError("training is blocked; pass --allow-paid after approval")
    if not environ.get("TINKER_API_KEY") or not environ.get("WANDB_API_KEY"):
        raise GRPOTrainingError("TINKER_API_KEY and WANDB_API_KEY are required")
    if environ.get("WANDB_MODE", "").lower() == "offline":
        raise GRPOTrainingError(
            "WANDB_MODE=offline cannot produce the required dashboard"
        )
    if estimate_max_token_cost_usd(config, manifest) > config.hard_cap_usd:
        raise GRPOTrainingError("estimated maximum token cost exceeds the hard cap")


def _encode(tokenizer: Any, text: str) -> Tuple[int, ...]:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    return tuple(int(token) for token in tokens)


async def _sample_groups(
    client: Any,
    rows: Sequence[Mapping[str, object]],
    tokenizer: Any,
    tinker_module: Any,
    config: GRPOConfig,
    seed_offset: int,
    require_logprobs: bool,
    label: str,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[RolloutGroup, ...]:
    """Sample one G-way rollout group per row, retaining only local train state."""
    prompts: list[Tuple[int, ...]] = []
    for row in rows:
        prompt_tokens = _encode(tokenizer, build_prompt(str(row["question"])))
        if not prompt_tokens or len(prompt_tokens) > config.max_prompt_tokens:
            raise GRPOTrainingError(
                "a rollout prompt exceeds the configured token limit"
            )
        prompts.append(prompt_tokens)

    async def sample_one(index: int) -> Tuple[int, RolloutGroup]:
        row = rows[index]
        prompt_tokens = prompts[index]
        result = await client.sample_async(
            prompt=tinker_module.ModelInput.from_ints(tokens=list(prompt_tokens)),
            num_samples=config.group_size,
            sampling_params=tinker_module.SamplingParams(
                max_tokens=config.max_output_tokens,
                temperature=config.temperature,
                seed=config.seed + seed_offset + index,
            ),
        )
        if len(result.sequences) != config.group_size:
            raise GRPOTrainingError(
                "Tinker returned the wrong number of rollout samples"
            )
        example_id = content_id(row)
        rollouts = []
        for sequence in result.sequences:
            tokens = tuple(int(token) for token in sequence.tokens)
            logprobs = getattr(sequence, "logprobs", None)
            if require_logprobs and (logprobs is None or len(logprobs) != len(tokens)):
                raise GRPOTrainingError("on-policy rollout is missing aligned logprobs")
            if not tokens:
                raise GRPOTrainingError("on-policy rollout is empty")
            response = tokenizer.decode(list(tokens))
            scored = score_completion(
                Completion(
                    example_id=example_id,
                    response=response,
                    ground_truth=str(row["answer"]),
                    output_tokens=len(tokens),
                    max_output_tokens=config.max_output_tokens,
                ),
                group_id=example_id,
            )
            rollouts.append(
                Rollout(
                    tokens=tokens,
                    logprobs=(
                        tuple(float(logprob) for logprob in logprobs)
                        if logprobs is not None
                        else None
                    ),
                    response=response,
                    scored=scored,
                )
            )
        return index, RolloutGroup(
            example_id=example_id,
            question=str(row["question"]),
            ground_truth=str(row["answer"]),
            prompt_tokens=prompt_tokens,
            rollouts=tuple(rollouts),
        )

    tasks = [asyncio.create_task(sample_one(index)) for index in range(len(rows))]
    groups_by_index: Dict[int, RolloutGroup] = {}
    try:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            index, group = await task
            groups_by_index[index] = group
            if progress and (
                completed == 1 or completed % 32 == 0 or completed == len(tasks)
            ):
                progress(
                    f"{label} prompts={completed}/{len(tasks)} rollouts="
                    f"{completed * config.group_size}/{len(tasks) * config.group_size}"
                )
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return tuple(groups_by_index[index] for index in range(len(rows)))


def _group_metrics(
    groups: Sequence[RolloutGroup], config: GRPOConfig
) -> Dict[str, float]:
    if not groups:
        raise GRPOTrainingError("a GRPO batch must contain at least one rollout group")
    rewards = [
        [float(rollout.scored.correct) for rollout in group.rollouts]
        for group in groups
    ]
    flat_rewards = [reward for group in rewards for reward in group]
    advantages = [
        reward - sum(group) / len(group) for group in rewards for reward in group
    ]
    all_correct = sum(all(group) for group in rewards)
    all_wrong = sum(not any(group) for group in rewards)
    mixed = len(groups) - all_correct - all_wrong
    return {
        "train/reward_mean": sum(flat_rewards) / len(flat_rewards),
        "train/group_pass_at_4": sum(any(group[:4]) for group in rewards) / len(groups),
        "train/group_all_correct_frac": all_correct / len(groups),
        "train/group_all_wrong_frac": all_wrong / len(groups),
        "train/group_mixed_frac": mixed / len(groups),
        "train/degenerate_group_frac": (all_correct + all_wrong) / len(groups),
        "train/effective_group_count": float(mixed),
        "train/group_reward_std_mean": sum(
            statistics.pstdev(group) for group in rewards
        )
        / len(groups),
        "train/advantage_abs_mean": sum(abs(value) for value in advantages)
        / len(advantages),
        "train/format_accuracy": sum(
            rollout.scored.format_valid
            for group in groups
            for rollout in group.rollouts
        )
        / len(flat_rewards),
        "train/truncation_rate": sum(
            rollout.scored.truncated for group in groups for rollout in group.rollouts
        )
        / len(flat_rewards),
        "train/avg_output_tokens": sum(
            rollout.scored.output_tokens
            for group in groups
            for rollout in group.rollouts
        )
        / len(flat_rewards),
    }


def _materialize_rl_datums(
    groups: Sequence[RolloutGroup], tinker_module: Any
) -> Tuple[list[Any], int]:
    """Mask prompts and apply each group-relative advantage to completion tokens."""
    data = []
    input_tokens = 0
    for group in groups:
        rewards = [float(rollout.scored.correct) for rollout in group.rollouts]
        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        if all(advantage == 0.0 for advantage in advantages):
            continue
        prefix_length = len(group.prompt_tokens) - 1
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            if rollout.logprobs is None:
                raise GRPOTrainingError("importance sampling requires rollout logprobs")
            full_tokens = list(group.prompt_tokens + rollout.tokens)
            target_tokens = full_tokens[1:]
            padded_logprobs = [0.0] * prefix_length + list(rollout.logprobs)
            padded_advantages = [0.0] * prefix_length + [advantage] * len(
                rollout.tokens
            )
            if not (
                len(target_tokens) == len(padded_logprobs) == len(padded_advantages)
            ):
                raise GRPOTrainingError("RL datum tensors have inconsistent lengths")
            model_input = tinker_module.ModelInput.from_ints(tokens=full_tokens[:-1])
            data.append(
                tinker_module.types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": target_tokens,
                        "logprobs": padded_logprobs,
                        "advantages": padded_advantages,
                    },
                )
            )
            input_tokens += len(full_tokens) - 1
    return data, input_tokens


def _loss_sum(result: Any) -> float:
    metrics = getattr(result, "metrics", {})
    if "loss:sum" in metrics:
        return float(metrics["loss:sum"])
    if getattr(result, "loss", None) is not None:
        return float(result.loss)
    return 0.0


def _monitor_table_rows(groups: Sequence[RolloutGroup]) -> Tuple[Tuple[Any, ...], ...]:
    rows = []
    for group in groups:
        rewards = [float(rollout.scored.correct) for rollout in group.rollouts]
        mean_reward = sum(rewards) / len(rewards)
        for rollout_id, rollout in enumerate(group.rollouts):
            scored = rollout.scored
            rows.append(
                (
                    group.example_id,
                    group.example_id,
                    rollout_id,
                    group.question,
                    group.ground_truth,
                    rollout.response,
                    scored.parsed_answer,
                    scored.correct,
                    scored.format_valid,
                    scored.output_tokens,
                    scored.truncated,
                    float(scored.correct),
                    mean_reward,
                    float(scored.correct) - mean_reward,
                )
            )
    return tuple(rows)


async def _monitor_policy(
    service_client: Any,
    sampler_path: Optional[str],
    rows: Sequence[Mapping[str, object]],
    config: GRPOConfig,
    tinker_module: Any,
    label: str,
    progress: Callable[[str], None],
) -> MonitorReport:
    """Score a frozen held-out RL monitor without feeding it into GRPO updates."""
    client = await service_client.create_sampling_client_async(
        **(
            {"model_path": sampler_path}
            if sampler_path
            else {"base_model": config.model_id}
        )
    )
    tokenizer = client.get_tokenizer()
    groups = await _sample_groups(
        client,
        rows,
        tokenizer,
        tinker_module,
        config,
        seed_offset=1_000_000,
        require_logprobs=False,
        label=f"monitor {label}",
        progress=progress,
    )
    evaluation_groups = {
        group.example_id: tuple(
            Completion(
                example_id=group.example_id,
                response=rollout.response,
                ground_truth=group.ground_truth,
                output_tokens=rollout.scored.output_tokens,
                max_output_tokens=config.max_output_tokens,
            )
            for rollout in group.rollouts
        )
        for group in groups
    }
    report = evaluate_groups(evaluation_groups, pass_k=4)
    return MonitorReport(
        metrics=dict(report.metrics),
        prompt_tokens=sum(
            len(group.prompt_tokens) * config.group_size for group in groups
        ),
        output_tokens=sum(
            rollout.scored.output_tokens
            for group in groups
            for rollout in group.rollouts
        ),
        table_rows=_monitor_table_rows(groups),
    )


def _monitor_metrics(report: MonitorReport) -> Dict[str, float]:
    return {
        key.replace("eval/", "rl_monitor/"): value
        for key, value in report.metrics.items()
    }


def _monitor_score(report: MonitorReport) -> Tuple[float, float]:
    return (
        report.metrics["eval/pass_at_4"],
        report.metrics["eval/pass_at_1"],
    )


def _is_material_monitor_regression(
    current: MonitorReport,
    best: MonitorReport,
    max_regression: float,
) -> bool:
    current_pass_at_4, current_pass_at_1 = _monitor_score(current)
    best_pass_at_4, best_pass_at_1 = _monitor_score(best)
    return (
        current_pass_at_4 < best_pass_at_4 - max_regression
        and current_pass_at_1 < best_pass_at_1 - max_regression
    )


def _select_checkpoint(
    checkpoints: Sequence[CheckpointRecord], monitor_enabled: bool
) -> CheckpointRecord:
    if not checkpoints:
        raise GRPOTrainingError("training completed without a persistent checkpoint")
    if not monitor_enabled:
        return checkpoints[-1]
    if any(record.monitor_pass_at_4 is None for record in checkpoints):
        raise GRPOTrainingError("checkpoint monitor metrics are missing")
    return max(
        checkpoints,
        key=lambda record: (
            float(record.monitor_pass_at_4),
            float(record.monitor_pass_at_1),
            -record.step,
        ),
    )


def _report_path(experiment_id: str, run_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id)
    return OUTPUT_DIR / f"{experiment_id}_grpo_report_{safe_id}.json"


async def run_grpo_training(
    config: GRPOConfig,
    allow_paid: bool,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
    train_rows: Optional[Sequence[Mapping[str, object]]] = None,
    monitor_rows: Optional[Sequence[Mapping[str, object]]] = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = _print_progress,
) -> Dict[str, Any]:
    """Run on-policy GRPO from an SFT checkpoint or a fresh Base LoRA."""
    manifest = read_manifest() if manifest is None else manifest
    _authorize(config, manifest, allow_paid, environ)
    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise GRPOTrainingError(
                "Tinker SDK is unavailable; run with `uv run --extra tinker`"
            ) from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise GRPOTrainingError("Weights & Biases is unavailable") from exc
    if train_rows is None:
        progress("loading frozen rl_train and rl_monitor rows")
        train_rows = load_official_train_rows(manifest, "rl_train")
    if monitor_rows is None:
        monitor_rows = load_official_train_rows(manifest, "rl_monitor")
    if len(train_rows) != len(manifest.rl_train_ids):
        raise GRPOTrainingError("loaded rl_train rows do not match the manifest")
    if len(monitor_rows) != len(manifest.rl_monitor_ids):
        raise GRPOTrainingError("loaded rl_monitor rows do not match the manifest")
    monitor_rows = tuple(monitor_rows[: config.monitor_examples])

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
            f"authorized run={config.run_name} steps={config.steps} batch={config.batch_size} "
            f"groups={config.group_size} monitor={len(monitor_rows)} "
            f"max_cost=${estimate_max_token_cost_usd(config, manifest):.4f}"
        )
        if config.init_source == "sft":
            progress("restoring SFT training state with a fresh RL optimizer")
            training_client = (
                await service_client.create_training_client_from_state_async(
                    config.parent_state_path,
                    base_model=config.model_id,
                    user_metadata={"experiment_id": config.experiment_id},
                )
            )
        else:
            progress("initializing a fresh Base LoRA for the direct-RL ablation")
            training_client = await service_client.create_lora_training_client_async(
                base_model=config.model_id,
                rank=config.lora_rank,
                seed=config.seed,
                user_metadata={"experiment_id": config.experiment_id},
            )
        tokenizer = training_client.get_tokenizer()
        wandb_run = wandb_module.init(
            project=config.project,
            entity=environ.get("WANDB_ENTITY") or None,
            name=config.run_name,
            group=config.suite_id,
            job_type="grpo-training",
            tags=["gsm8k", "grpo", config.experiment_id, f"g{config.group_size}"],
            config=_tracking_config(config, manifest),
        )
        progress(f"started W&B run={getattr(wandb_run, 'url', None)}")

        sampled_prompt_tokens = 0
        sampled_output_tokens = 0
        optimized_input_tokens = 0
        sampled_training_group_count = 0
        checkpoints: list[CheckpointRecord] = []
        baseline_monitor: Optional[MonitorReport] = None
        best_monitor: Optional[MonitorReport] = None
        regression_streak = 0
        early_stop_triggered = False
        completed_steps = 0
        if monitor_rows:
            progress(f"monitoring initialization policy prompts={len(monitor_rows)}")
            baseline_monitor = await _monitor_policy(
                service_client,
                config.parent_sampler_path if config.init_source == "sft" else None,
                monitor_rows,
                config,
                tinker_module,
                "step=0",
                progress,
            )
            sampled_prompt_tokens += baseline_monitor.prompt_tokens
            sampled_output_tokens += baseline_monitor.output_tokens
            baseline_metrics = {
                **_monitor_metrics(baseline_monitor),
                "rl_monitor/is_initialization": 1.0,
                "run_stats/estimated_cumulative_usd": _sampling_cost(
                    sampled_prompt_tokens, sampled_output_tokens, config
                ),
            }
            if hasattr(wandb_module, "Table"):
                baseline_metrics["tables/rl_monitor_rollouts"] = wandb_module.Table(
                    columns=list(MONITOR_COLUMNS),
                    data=list(baseline_monitor.table_rows),
                )
            wandb_run.log(baseline_metrics, step=0)
            best_monitor = baseline_monitor
            progress(
                f"monitor step=0/{config.steps} pass_at_1="
                f"{baseline_monitor.metrics['eval/pass_at_1']:.4f} pass_at_4="
                f"{baseline_monitor.metrics['eval/pass_at_4']:.4f}"
            )
            checkpoints.append(
                CheckpointRecord(
                    step=0,
                    state_path=(
                        config.parent_state_path
                        if config.init_source == "sft"
                        else None
                    ),
                    sampler_path=(
                        config.parent_sampler_path
                        if config.init_source == "sft"
                        else None
                    ),
                    monitor_pass_at_1=baseline_monitor.metrics["eval/pass_at_1"],
                    monitor_pass_at_4=baseline_monitor.metrics["eval/pass_at_4"],
                )
            )

        started_at = clock()
        checkpoint_steps = set(config.checkpoint_steps())
        train_cursor = 0
        for step in range(1, config.steps + 1):
            step_started_at = clock()
            sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            groups: list[RolloutGroup] = []
            resample_rounds = 0
            for resample_round in range(config.max_resample_rounds + 1):
                batch = tuple(
                    train_rows[(train_cursor + offset) % len(train_rows)]
                    for offset in range(config.batch_size)
                )
                train_cursor = (train_cursor + config.batch_size) % len(train_rows)
                round_groups = await _sample_groups(
                    sampling_client,
                    batch,
                    tokenizer,
                    tinker_module,
                    config,
                    seed_offset=(
                        (
                            (step - 1) * (config.max_resample_rounds + 1)
                            + resample_round
                            + 1
                        )
                        * config.batch_size
                    ),
                    require_logprobs=True,
                    label=(
                        f"rollout step={step}/{config.steps} "
                        f"round={resample_round + 1}"
                    ),
                )
                groups.extend(round_groups)
                if (
                    not config.min_effective_groups
                    or _group_metrics(groups, config)["train/effective_group_count"]
                    >= config.min_effective_groups
                ):
                    break
                resample_rounds = resample_round + 1
            sampled_prompt_tokens += sum(
                len(group.prompt_tokens) * config.group_size for group in groups
            )
            sampled_output_tokens += sum(
                rollout.scored.output_tokens
                for group in groups
                for rollout in group.rollouts
            )
            sampled_training_group_count += len(groups)
            metrics = _group_metrics(groups, config)
            data, batch_input_tokens = _materialize_rl_datums(groups, tinker_module)
            optimized_input_tokens += batch_input_tokens
            loss_sum = 0.0
            if data:
                forward_backward = await training_client.forward_backward_async(
                    data=data, loss_fn="importance_sampling"
                )
                forward_backward_result = await forward_backward.result_async()
                loss_sum = _loss_sum(forward_backward_result)
                optimizer = await training_client.optim_step_async(
                    tinker_module.types.AdamParams(learning_rate=config.learning_rate)
                )
                await optimizer.result_async()
            elapsed_seconds = clock() - started_at
            step_seconds = max(clock() - step_started_at, 1e-9)
            eta_seconds = (config.steps - step) * elapsed_seconds / step
            estimated_cost = _sampling_cost(
                sampled_prompt_tokens, sampled_output_tokens, config
            ) + _training_cost(optimized_input_tokens, config)
            metrics.update(
                {
                    "train/importance_sampling_loss_sum": loss_sum,
                    "train/update_applied": float(bool(data)),
                    "train/datums": float(len(data)),
                    "train/candidate_group_count": float(len(groups)),
                    "train/resample_rounds": float(resample_rounds),
                    "train/target_effective_groups": float(
                        config.min_effective_groups
                    ),
                    "train/target_effective_groups_reached": float(
                        not config.min_effective_groups
                        or metrics["train/effective_group_count"]
                        >= config.min_effective_groups
                    ),
                    "train/learning_rate": config.learning_rate,
                    "timing/step_seconds": step_seconds,
                    "timing/elapsed_seconds": elapsed_seconds,
                    "timing/eta_seconds": eta_seconds,
                    "timing/rollout_tokens_per_second": sum(
                        rollout.scored.output_tokens
                        for group in groups
                        for rollout in group.rollouts
                    )
                    / step_seconds,
                    "run_stats/cumulative_sample_prompt_tokens": float(
                        sampled_prompt_tokens
                    ),
                    "run_stats/cumulative_sample_output_tokens": float(
                        sampled_output_tokens
                    ),
                    "run_stats/cumulative_optimized_input_tokens": float(
                        optimized_input_tokens
                    ),
                    "run_stats/cumulative_training_rollouts": float(
                        sampled_training_group_count * config.group_size
                    ),
                    "run_stats/estimated_cumulative_usd": estimated_cost,
                }
            )
            completed_steps = step
            wandb_run.log(metrics, step=step)
            if step == 1 or step % config.progress_every == 0 or step == config.steps:
                progress(
                    f"step={step}/{config.steps} groups={len(groups)} "
                    f"resamples={resample_rounds} "
                    f"reward={metrics['train/reward_mean']:.4f} "
                    f"mixed={metrics['train/group_mixed_frac']:.3f} "
                    f"degenerate={metrics['train/degenerate_group_frac']:.3f} "
                    f"datums={len(data)} throughput="
                    f"{metrics['timing/rollout_tokens_per_second']:.1f}tok/s "
                    f"elapsed={elapsed_seconds:.1f}s eta={eta_seconds:.1f}s "
                    f"estimated_cost=${estimated_cost:.4f}"
                )
            if estimated_cost > config.hard_cap_usd:
                raise GRPOTrainingError(
                    "observed token cost exceeded the configured hard cap"
                )
            if step not in checkpoint_steps:
                continue

            checkpoint_name = f"{config.run_name}-step{step}"
            progress(f"saving checkpoint step={step}/{config.steps}")
            state_future = await training_client.save_state_async(
                checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
            )
            state_result = await state_future.result_async()
            sampler_future = await training_client.save_weights_for_sampler_async(
                checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds
            )
            sampler_result = await sampler_future.result_async()
            monitor: Optional[MonitorReport] = None
            if monitor_rows:
                progress(
                    f"monitoring checkpoint step={step}/{config.steps} prompts={len(monitor_rows)}"
                )
                monitor = await _monitor_policy(
                    service_client,
                    str(sampler_result.path),
                    monitor_rows,
                    config,
                    tinker_module,
                    f"step={step}",
                    progress,
                )
                sampled_prompt_tokens += monitor.prompt_tokens
                sampled_output_tokens += monitor.output_tokens
            record = CheckpointRecord(
                step=step,
                state_path=str(state_result.path),
                sampler_path=str(sampler_result.path),
                monitor_pass_at_1=(
                    monitor.metrics["eval/pass_at_1"] if monitor else None
                ),
                monitor_pass_at_4=(
                    monitor.metrics["eval/pass_at_4"] if monitor else None
                ),
            )
            checkpoints.append(record)
            checkpoint_metrics: Dict[str, Any] = {
                "checkpoint/is_initialization": 0.0,
                "checkpoint/state_path": record.state_path,
                "checkpoint/sampler_path": record.sampler_path,
                "run_stats/estimated_cumulative_usd": _sampling_cost(
                    sampled_prompt_tokens, sampled_output_tokens, config
                )
                + _training_cost(optimized_input_tokens, config),
            }
            if monitor is not None:
                checkpoint_metrics.update(_monitor_metrics(monitor))
                checkpoint_metrics["rl_monitor/is_initialization"] = 0.0
                if baseline_monitor is not None:
                    checkpoint_metrics.update(
                        {
                            "rl_monitor/pass_at_1_delta_from_initialization": (
                                monitor.metrics["eval/pass_at_1"]
                                - baseline_monitor.metrics["eval/pass_at_1"]
                            ),
                            "rl_monitor/pass_at_4_delta_from_initialization": (
                                monitor.metrics["eval/pass_at_4"]
                                - baseline_monitor.metrics["eval/pass_at_4"]
                            ),
                        }
                    )
                if best_monitor is None:
                    best_monitor = monitor
                    is_material_regression = False
                elif _monitor_score(monitor) > _monitor_score(best_monitor):
                    best_monitor = monitor
                    regression_streak = 0
                    is_material_regression = False
                else:
                    is_material_regression = _is_material_monitor_regression(
                        monitor,
                        best_monitor,
                        config.early_stopping_max_regression,
                    )
                if is_material_regression:
                    regression_streak += 1
                elif _monitor_score(monitor) <= _monitor_score(best_monitor):
                    regression_streak = 0
                checkpoint_metrics.update(
                    {
                        "early_stopping/best_pass_at_1": best_monitor.metrics[
                            "eval/pass_at_1"
                        ],
                        "early_stopping/best_pass_at_4": best_monitor.metrics[
                            "eval/pass_at_4"
                        ],
                        "early_stopping/regression_streak": float(regression_streak),
                        "early_stopping/is_material_regression": float(
                            is_material_regression
                        ),
                    }
                )
                early_stop_triggered = bool(
                    config.early_stopping_patience is not None
                    and regression_streak >= config.early_stopping_patience
                )
                checkpoint_metrics["early_stopping/triggered"] = float(
                    early_stop_triggered
                )
                if hasattr(wandb_module, "Table"):
                    checkpoint_metrics["tables/rl_monitor_rollouts"] = (
                        wandb_module.Table(
                            columns=list(MONITOR_COLUMNS), data=list(monitor.table_rows)
                        )
                    )
            wandb_run.log(checkpoint_metrics, step=step)
            if monitor is not None:
                progress(
                    f"monitor step={step}/{config.steps} pass_at_1="
                    f"{monitor.metrics['eval/pass_at_1']:.4f} pass_at_4="
                    f"{monitor.metrics['eval/pass_at_4']:.4f}"
                )
            if early_stop_triggered:
                progress(
                    f"early stop step={step}/{config.steps} streak={regression_streak} "
                    f"tolerance={config.early_stopping_max_regression:.6f}"
                )
                break

        selected = _select_checkpoint(checkpoints, bool(monitor_rows))
        total_estimated_cost = _sampling_cost(
            sampled_prompt_tokens, sampled_output_tokens, config
        ) + _training_cost(optimized_input_tokens, config)
        report = {
            "mode": "remote-grpo-training",
            "network_called": True,
            "run_name": config.run_name,
            "model_id": config.model_id,
            "manifest_hash": manifest.manifest_hash,
            "initialization_source": config.init_source,
            "initialization_label": config.initialization_label
            or ("base" if config.init_source == "base" else "e2s250"),
            "parent_checkpoint": config.parent_checkpoint
            if config.init_source == "sft"
            else None,
            "parent_state_path": config.parent_state_path
            if config.init_source == "sft"
            else None,
            "parent_sampler_path": (
                config.parent_sampler_path if config.init_source == "sft" else None
            ),
            "rl_train_examples": len(train_rows),
            "rl_monitor_examples": len(monitor_rows),
            "training_steps": config.steps,
            "completed_training_steps": completed_steps,
            "training_rollouts": sampled_training_group_count * config.group_size,
            "max_training_rollouts": (
                config.steps * config.max_candidate_groups_per_step * config.group_size
            ),
            "early_stopping_triggered": early_stop_triggered,
            "early_stopping_regression_streak": regression_streak,
            "baseline_monitor": asdict(baseline_monitor) if baseline_monitor else None,
            "selected_checkpoint": asdict(selected),
            "checkpoints": [asdict(record) for record in checkpoints],
            "estimated_token_cost_usd": total_estimated_cost,
            "hard_cap_usd": config.hard_cap_usd,
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _report_path(config.experiment_id, str(getattr(wandb_run, "id", "run"))).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        wandb_run.summary.update(
            {
                "checkpoint/selection_metric": "rl_monitor/pass_at_4_then_pass_at_1",
                "checkpoint/selected_step": selected.step,
                "checkpoint/selected_state_path": selected.state_path,
                "checkpoint/selected_sampler_path": selected.sampler_path,
                "checkpoint/selected_monitor_pass_at_1": selected.monitor_pass_at_1,
                "checkpoint/selected_monitor_pass_at_4": selected.monitor_pass_at_4,
                "early_stopping/triggered": early_stop_triggered,
                "early_stopping/regression_streak": regression_streak,
                "run_stats/completed_training_steps": completed_steps,
                "run_stats/estimated_total_usd": total_estimated_cost,
            }
        )
        progress(
            f"complete selected_step={selected.step} completed_steps={completed_steps} "
            f"estimated_cost=${total_estimated_cost:.4f}"
        )
        return report
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if owned_http_client is not None:
            await owned_http_client.aclose()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight or run frozen GSM8K GRPO.")
    parser.add_argument("--run", action="store_true", help="Start the paid GRPO run.")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--experiment-id", choices=GRPO_EXPERIMENT_IDS, default=EXPERIMENT_ID)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--init-source", choices=("sft", "base"), default="sft")
    parser.add_argument("--init-label")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--monitor-examples", type=int, default=DEFAULT_MONITOR_EXAMPLES
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY
    )
    parser.add_argument("--min-effective-groups", type=int, default=0)
    parser.add_argument("--max-resample-rounds", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-max-regression", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--hard-cap-usd", type=float, default=DEFAULT_HARD_CAP_USD)
    parser.add_argument("--parent-state-path", default=PARENT_STATE_PATH)
    parser.add_argument("--parent-sampler-path", default=PARENT_SAMPLER_PATH)
    parser.add_argument("--parent-checkpoint", default=PARENT_CHECKPOINT)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> GRPOConfig:
    return GRPOConfig(
        model_id=args.model_id,
        experiment_id=args.experiment_id,
        attempt=args.attempt,
        init_source=args.init_source,
        initialization_label=args.init_label,
        parent_checkpoint=args.parent_checkpoint,
        parent_state_path=args.parent_state_path,
        parent_sampler_path=args.parent_sampler_path,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_prompt_tokens=args.max_prompt_tokens,
        max_output_tokens=args.max_output_tokens,
        monitor_examples=args.monitor_examples,
        checkpoint_every=args.checkpoint_every,
        min_effective_groups=args.min_effective_groups,
        max_resample_rounds=args.max_resample_rounds,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_max_regression=args.early_stopping_max_regression,
        progress_every=args.progress_every,
        hard_cap_usd=args.hard_cap_usd,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.run:
            report = asyncio.run(run_grpo_training(config, allow_paid=args.allow_paid))
        else:
            if args.allow_paid:
                raise GRPOTrainingError("--allow-paid requires --run")
            report = build_doctor_report(config)
    except (GRPOTrainingError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
