"""Configurable, real-data SFT pilot for Tinker.

The default command validates the checked-in configuration without network
access. ``--prepare-data`` streams and validates the pinned Hugging Face
dataset without using Tinker. The paid training path requires both ``--run``
and ``--allow-paid``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from dotenv import load_dotenv

from modeling.llm_post_training.tinker.mvp import MIN_PYTHON

try:
    import tomllib
except ImportError:  # pragma: no cover - the Tinker SDK already requires 3.11+
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "sft_deepmath.toml"
TINKER_SDK_PACKAGE = "tinker"
WANDB_PACKAGE = "wandb"

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner. Show the key steps and finish "
    "with a clearly stated final answer."
)


class SFTExperimentError(RuntimeError):
    """Raised when configuration, data, safety, or remote results are invalid."""


def _print_progress(message: str) -> None:
    print(f"[tinker-sft] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class SFTConfig:
    """Fully resolved configuration for one real-data SFT pilot."""

    experiment_name: str
    output_dir: str
    seed: int
    model_id: str
    lora_rank: int
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    dataset_license: str
    solution_field: str
    streaming: bool
    shuffle_buffer: int
    candidate_examples: int
    train_examples: int
    eval_examples: int
    steps: int
    batch_size: int
    learning_rate: float
    max_sequence_tokens: int
    max_eval_prompt_tokens: int
    max_eval_output_tokens: int
    min_eval_completion_rate: float
    checkpoint_prefix: str
    checkpoint_ttl_seconds: int
    wandb_project: str
    prefill_usd_per_million: float
    sample_usd_per_million: float
    train_usd_per_million: float
    hard_cap_usd: float

    def validate(self) -> None:
        positive_ints = {
            "seed": self.seed,
            "lora_rank": self.lora_rank,
            "shuffle_buffer": self.shuffle_buffer,
            "candidate_examples": self.candidate_examples,
            "train_examples": self.train_examples,
            "eval_examples": self.eval_examples,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "max_sequence_tokens": self.max_sequence_tokens,
            "max_eval_prompt_tokens": self.max_eval_prompt_tokens,
            "max_eval_output_tokens": self.max_eval_output_tokens,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0 and name != "seed":
                raise SFTExperimentError(f"{name} must be positive")
            if name == "seed" and value < 0:
                raise SFTExperimentError("seed must be non-negative")
        if self.train_examples < self.batch_size:
            raise SFTExperimentError("train_examples must be at least batch_size")
        if self.candidate_examples < self.train_examples + self.eval_examples:
            raise SFTExperimentError(
                "candidate_examples must cover train_examples plus eval_examples"
            )
        if self.learning_rate <= 0:
            raise SFTExperimentError("learning_rate must be positive")
        if not 0 < self.min_eval_completion_rate <= 1:
            raise SFTExperimentError("evaluation.min_completion_rate must be in (0, 1]")
        for name, value in (
            ("prefill_usd_per_million", self.prefill_usd_per_million),
            ("sample_usd_per_million", self.sample_usd_per_million),
            ("train_usd_per_million", self.train_usd_per_million),
            ("hard_cap_usd", self.hard_cap_usd),
        ):
            if value <= 0:
                raise SFTExperimentError(f"{name} must be positive")
        for name, value in (
            ("experiment.name", self.experiment_name),
            ("model.id", self.model_id),
            ("dataset.id", self.dataset_id),
            ("dataset.revision", self.dataset_revision),
            ("dataset.split", self.dataset_split),
            ("dataset.license", self.dataset_license),
            ("dataset.solution_field", self.solution_field),
            ("checkpoint.prefix", self.checkpoint_prefix),
            ("tracking.wandb_project", self.wandb_project),
        ):
            if not value.strip():
                raise SFTExperimentError(f"{name} must not be empty")


@dataclass(frozen=True)
class MathExample:
    """One source example with a stable, content-derived identity."""

    example_id: str
    question: str
    solution: str
    final_answer: str
    topic: str
    difficulty: Optional[float]


@dataclass(frozen=True)
class TokenizedTrainingExample:
    """One SFT example after applying the model chat template."""

    source: MathExample
    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    weights: tuple[float, ...]
    supervised_tokens: int


@dataclass(frozen=True)
class PreparedDataset:
    """Deterministic, disjoint training and evaluation subsets."""

    train: tuple[TokenizedTrainingExample, ...]
    evaluation: tuple[MathExample, ...]
    skipped_too_long: int


@dataclass(frozen=True)
class EvaluationObservation:
    example_id: str
    expected_answer: str
    parsed_answer: str
    correct: bool
    truncated: bool
    prompt_tokens: int
    output_tokens: int
    response_text: str


@dataclass(frozen=True)
class EvaluationSummary:
    accuracy: float
    score_completed: float
    parse_rate: float
    completion_rate: float
    truncation_rate: float
    prompt_tokens: int
    output_tokens: int
    observations: tuple[EvaluationObservation, ...]


@dataclass(frozen=True)
class SFTDoctorReport:
    mode: str
    network_called: bool
    python_supported: bool
    tinker_sdk_available: bool
    tinker_sdk_version: Optional[str]
    wandb_available: bool
    wandb_version: Optional[str]
    tinker_api_key_configured: bool
    wandb_api_key_configured: bool
    hf_token_configured: bool
    model_id: str
    dataset_id: str
    dataset_revision: str
    steps: int
    batch_size: int
    train_examples: int
    eval_examples: int
    estimated_max_token_cost_usd: float
    hard_cap_usd: float
    ready_for_paid_run: bool


@dataclass(frozen=True)
class DatasetPreparationReport:
    mode: str
    network_called: bool
    dataset_id: str
    dataset_revision: str
    candidates_loaded: int
    train_examples: int
    eval_examples: int
    skipped_too_long: int
    manifest_path: str


@dataclass(frozen=True)
class SFTTrainingReport:
    mode: str
    network_called: bool
    model_id: str
    dataset_id: str
    dataset_revision: str
    steps_completed: int
    train_examples: int
    eval_examples: int
    train_tokens: int
    baseline_accuracy: float
    final_accuracy: float
    accuracy_gain: float
    quality_comparison_valid: bool
    baseline_truncation_rate: float
    final_truncation_rate: float
    checkpoint_path: str
    sampler_path: str
    estimated_token_cost_usd: float
    hard_cap_usd: float
    wandb_project: str
    wandb_run_url: Optional[str]
    manifest_path: str
    report_path: str


_CONFIG_KEYS = {
    "experiment": {"name", "output_dir", "seed"},
    "model": {"id", "lora_rank"},
    "dataset": {
        "id",
        "revision",
        "split",
        "license",
        "solution_field",
        "streaming",
        "shuffle_buffer",
        "candidate_examples",
        "train_examples",
        "eval_examples",
    },
    "training": {"steps", "batch_size", "learning_rate", "max_sequence_tokens"},
    "evaluation": {
        "max_prompt_tokens",
        "max_output_tokens",
        "min_completion_rate",
    },
    "checkpoint": {"prefix", "ttl_seconds"},
    "tracking": {"wandb_project"},
    "pricing": {
        "prefill_usd_per_million",
        "sample_usd_per_million",
        "train_usd_per_million",
        "hard_cap_usd",
    },
}


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> SFTConfig:
    """Load a strict TOML config so misspelled parameters fail loudly."""
    if tomllib is None:
        raise SFTExperimentError("Python 3.11+ is required to read the TOML config")
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SFTExperimentError(f"could not load config {config_path}: {exc}") from exc

    unknown_sections = set(raw) - set(_CONFIG_KEYS)
    if unknown_sections:
        raise SFTExperimentError(f"unknown config sections: {sorted(unknown_sections)}")
    for section, expected_keys in _CONFIG_KEYS.items():
        if section not in raw or not isinstance(raw[section], dict):
            raise SFTExperimentError(f"missing config section [{section}]")
        unknown_keys = set(raw[section]) - expected_keys
        missing_keys = expected_keys - set(raw[section])
        if unknown_keys:
            raise SFTExperimentError(
                f"unknown keys in [{section}]: {sorted(unknown_keys)}"
            )
        if missing_keys:
            raise SFTExperimentError(
                f"missing keys in [{section}]: {sorted(missing_keys)}"
            )

    experiment = raw["experiment"]
    model = raw["model"]
    dataset = raw["dataset"]
    training = raw["training"]
    evaluation = raw["evaluation"]
    checkpoint = raw["checkpoint"]
    tracking = raw["tracking"]
    pricing = raw["pricing"]
    config = SFTConfig(
        experiment_name=str(experiment["name"]),
        output_dir=str(experiment["output_dir"]),
        seed=int(experiment["seed"]),
        model_id=str(model["id"]),
        lora_rank=int(model["lora_rank"]),
        dataset_id=str(dataset["id"]),
        dataset_revision=str(dataset["revision"]),
        dataset_split=str(dataset["split"]),
        dataset_license=str(dataset["license"]),
        solution_field=str(dataset["solution_field"]),
        streaming=bool(dataset["streaming"]),
        shuffle_buffer=int(dataset["shuffle_buffer"]),
        candidate_examples=int(dataset["candidate_examples"]),
        train_examples=int(dataset["train_examples"]),
        eval_examples=int(dataset["eval_examples"]),
        steps=int(training["steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        max_sequence_tokens=int(training["max_sequence_tokens"]),
        max_eval_prompt_tokens=int(evaluation["max_prompt_tokens"]),
        max_eval_output_tokens=int(evaluation["max_output_tokens"]),
        min_eval_completion_rate=float(evaluation["min_completion_rate"]),
        checkpoint_prefix=str(checkpoint["prefix"]),
        checkpoint_ttl_seconds=int(checkpoint["ttl_seconds"]),
        wandb_project=str(tracking["wandb_project"]),
        prefill_usd_per_million=float(pricing["prefill_usd_per_million"]),
        sample_usd_per_million=float(pricing["sample_usd_per_million"]),
        train_usd_per_million=float(pricing["train_usd_per_million"]),
        hard_cap_usd=float(pricing["hard_cap_usd"]),
    )
    config.validate()
    return config


def override_steps(config: SFTConfig, steps: Optional[int]) -> SFTConfig:
    if steps is None:
        return config
    updated = replace(config, steps=steps)
    updated.validate()
    return updated


def load_local_env() -> None:
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _package_version(package: str) -> Optional[str]:
    if importlib.util.find_spec(package) is None:
        return None
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def estimate_max_token_cost_usd(config: SFTConfig) -> float:
    """Conservative maximum for training plus matched baseline/final eval."""
    train_tokens = config.steps * config.batch_size * config.max_sequence_tokens
    eval_prompt_tokens = 2 * config.eval_examples * config.max_eval_prompt_tokens
    eval_output_tokens = 2 * config.eval_examples * config.max_eval_output_tokens
    return (
        train_tokens * config.train_usd_per_million
        + eval_prompt_tokens * config.prefill_usd_per_million
        + eval_output_tokens * config.sample_usd_per_million
    ) / 1_000_000


def estimate_actual_token_cost_usd(
    config: SFTConfig,
    train_tokens: int,
    eval_prompt_tokens: int,
    eval_output_tokens: int,
) -> float:
    if min(train_tokens, eval_prompt_tokens, eval_output_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    return (
        train_tokens * config.train_usd_per_million
        + eval_prompt_tokens * config.prefill_usd_per_million
        + eval_output_tokens * config.sample_usd_per_million
    ) / 1_000_000


def build_doctor_report(
    config: SFTConfig,
    environ: Mapping[str, str] = os.environ,
    tinker_version: Optional[str] = None,
    wandb_version: Optional[str] = None,
) -> SFTDoctorReport:
    detected_tinker = (
        _package_version(TINKER_SDK_PACKAGE)
        if tinker_version is None
        else tinker_version
    )
    detected_wandb = (
        _package_version(WANDB_PACKAGE) if wandb_version is None else wandb_version
    )
    estimated_cost = estimate_max_token_cost_usd(config)
    python_supported = sys.version_info[:2] >= MIN_PYTHON
    tinker_key = bool(environ.get("TINKER_API_KEY"))
    wandb_key = bool(environ.get("WANDB_API_KEY"))
    return SFTDoctorReport(
        mode="local-real-data-sft-preflight",
        network_called=False,
        python_supported=python_supported,
        tinker_sdk_available=detected_tinker is not None,
        tinker_sdk_version=detected_tinker,
        wandb_available=detected_wandb is not None,
        wandb_version=detected_wandb,
        tinker_api_key_configured=tinker_key,
        wandb_api_key_configured=wandb_key,
        hf_token_configured=bool(environ.get("HF_TOKEN")),
        model_id=config.model_id,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        steps=config.steps,
        batch_size=config.batch_size,
        train_examples=config.train_examples,
        eval_examples=config.eval_examples,
        estimated_max_token_cost_usd=estimated_cost,
        hard_cap_usd=config.hard_cap_usd,
        ready_for_paid_run=(
            python_supported
            and detected_tinker is not None
            and detected_wandb is not None
            and tinker_key
            and wandb_key
            and estimated_cost <= config.hard_cap_usd
        ),
    )


def _require_paid_authorization(
    config: SFTConfig,
    allow_paid: bool,
    environ: Mapping[str, str],
) -> None:
    if not allow_paid:
        raise SFTExperimentError(
            "training is blocked; pass --allow-paid only after explicit approval"
        )
    if not environ.get("TINKER_API_KEY"):
        raise SFTExperimentError("TINKER_API_KEY is not configured")
    if not environ.get("WANDB_API_KEY"):
        raise SFTExperimentError("WANDB_API_KEY is not configured")
    if estimate_max_token_cost_usd(config) > config.hard_cap_usd:
        raise SFTExperimentError(
            "estimated maximum token cost exceeds the configured hard cap"
        )


def _normalized_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def _example_id(question: str) -> str:
    return hashlib.sha256(_normalized_question(question).encode("utf-8")).hexdigest()


def load_dataset_candidates(
    config: SFTConfig,
    load_dataset_fn: Optional[Callable[..., Any]] = None,
) -> list[MathExample]:
    """Stream a pinned, shuffled candidate pool without materializing DeepMath."""
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as load_dataset_fn
        except ImportError as exc:
            raise SFTExperimentError("the datasets package is unavailable") from exc

    dataset = load_dataset_fn(
        config.dataset_id,
        split=config.dataset_split,
        revision=config.dataset_revision,
        streaming=config.streaming,
    )
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=config.shuffle_buffer,
        )

    candidates: list[MathExample] = []
    seen_ids: set[str] = set()
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        question = str(row.get("question") or "").strip()
        solution = str(row.get(config.solution_field) or "").strip()
        final_answer = str(row.get("final_answer") or "").strip()
        if not question or not solution or not final_answer:
            continue
        example_id = _example_id(question)
        if example_id in seen_ids:
            continue
        difficulty_raw = row.get("difficulty")
        try:
            difficulty = float(difficulty_raw) if difficulty_raw is not None else None
        except (TypeError, ValueError):
            difficulty = None
        seen_ids.add(example_id)
        candidates.append(
            MathExample(
                example_id=example_id,
                question=question,
                solution=solution,
                final_answer=final_answer,
                topic=str(row.get("topic") or ""),
                difficulty=difficulty,
            )
        )
        if len(candidates) >= config.candidate_examples:
            break

    if len(candidates) < config.candidate_examples:
        raise SFTExperimentError(
            f"dataset yielded only {len(candidates)} valid candidates; "
            f"expected {config.candidate_examples}"
        )
    return candidates


def _chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise SFTExperimentError("the model tokenizer has no chat template")
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        tokens = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        tokens = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(tokens, Mapping):
        if "input_ids" not in tokens:
            raise SFTExperimentError(
                "chat template returned a mapping without input_ids"
            )
        tokens = tokens["input_ids"]
    return [int(token) for token in tokens]


def _user_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def tokenize_training_example(
    example: MathExample,
    tokenizer: Any,
    max_sequence_tokens: int,
) -> Optional[TokenizedTrainingExample]:
    prompt_messages = _user_messages(example.question)
    prompt_tokens = _chat_tokens(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    try:
        completion_tokens = list(
            tokenizer.encode(example.solution, add_special_tokens=False)
        )
    except TypeError:
        completion_tokens = list(tokenizer.encode(example.solution))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_tokens.append(int(eos_token_id))
    full_tokens = prompt_tokens + [int(token) for token in completion_tokens]
    if len(prompt_tokens) >= len(full_tokens):
        raise SFTExperimentError("training example has no supervised assistant tokens")
    if len(full_tokens) - 1 > max_sequence_tokens:
        return None

    input_tokens = tuple(full_tokens[:-1])
    target_tokens = tuple(full_tokens[1:])
    weights = tuple(
        [0.0] * (len(prompt_tokens) - 1)
        + [1.0] * (len(full_tokens) - len(prompt_tokens))
    )
    if not (len(input_tokens) == len(target_tokens) == len(weights)):
        raise SFTExperimentError("shifted SFT tensors have inconsistent lengths")
    supervised_tokens = int(sum(weights))
    if supervised_tokens <= 0:
        raise SFTExperimentError("training example has no supervised tokens")
    return TokenizedTrainingExample(
        source=example,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=weights,
        supervised_tokens=supervised_tokens,
    )


def _eval_prompt_tokens(example: MathExample, tokenizer: Any) -> list[int]:
    return _chat_tokens(
        tokenizer,
        _user_messages(example.question),
        add_generation_prompt=True,
    )


def prepare_dataset(
    candidates: Iterable[MathExample],
    tokenizer: Any,
    config: SFTConfig,
) -> PreparedDataset:
    """Create fixed non-overlapping eval and train subsets after token checks."""
    evaluation: list[MathExample] = []
    training: list[TokenizedTrainingExample] = []
    skipped_too_long = 0
    for example in candidates:
        if len(evaluation) < config.eval_examples:
            prompt_tokens = _eval_prompt_tokens(example, tokenizer)
            if len(prompt_tokens) <= config.max_eval_prompt_tokens:
                evaluation.append(example)
            else:
                skipped_too_long += 1
            continue
        if len(training) >= config.train_examples:
            break
        tokenized = tokenize_training_example(
            example,
            tokenizer,
            config.max_sequence_tokens,
        )
        if tokenized is None:
            skipped_too_long += 1
            continue
        training.append(tokenized)

    if len(evaluation) != config.eval_examples:
        raise SFTExperimentError(
            f"prepared {len(evaluation)} eval examples; expected {config.eval_examples}"
        )
    if len(training) != config.train_examples:
        raise SFTExperimentError(
            f"prepared {len(training)} train examples; "
            f"expected {config.train_examples}; "
            "increase dataset.candidate_examples or max_sequence_tokens"
        )
    train_ids = {example.source.example_id for example in training}
    eval_ids = {example.example_id for example in evaluation}
    if train_ids & eval_ids:
        raise SFTExperimentError("train/eval overlap detected")
    return PreparedDataset(
        train=tuple(training),
        evaluation=tuple(evaluation),
        skipped_too_long=skipped_too_long,
    )


def _manifest_payload(config: SFTConfig, prepared: PreparedDataset) -> dict[str, Any]:
    def source_record(example: MathExample) -> dict[str, Any]:
        return {
            "example_id": example.example_id,
            "difficulty": example.difficulty,
            "topic": example.topic,
        }

    return {
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "dataset_split": config.dataset_split,
        "dataset_license": config.dataset_license,
        "solution_field": config.solution_field,
        "seed": config.seed,
        "train": [source_record(example.source) for example in prepared.train],
        "evaluation": [source_record(example) for example in prepared.evaluation],
        "skipped_too_long": prepared.skipped_too_long,
    }


def _output_path(config: SFTConfig, filename: str) -> Path:
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(config: SFTConfig, prepared: PreparedDataset) -> Path:
    path = _output_path(config, "dataset_manifest.json")
    _write_json(path, _manifest_payload(config, prepared))
    return path


def _last_boxed_value(text: str) -> str:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""
    index = start + len(marker)
    depth = 1
    for cursor in range(index, len(text)):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[index:cursor]
    return ""


def extract_final_answer(text: str) -> str:
    boxed = _last_boxed_value(text)
    if boxed:
        return boxed.strip()
    matches = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer)\s*[:=]\s*(.+?)\s*$",
        text,
    )
    if matches:
        return matches[-1].strip()
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else ""


def normalize_answer(answer: str) -> str:
    value = answer.strip()
    boxed = _last_boxed_value(value)
    if boxed:
        value = boxed
    value = value.replace(r"\dfrac", r"\frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "")
    value = value.replace("$", "")
    value = re.sub(r"\s+", "", value)
    return value.strip(".。;,，").casefold()


async def evaluate_sampling_client(
    sampling_client: Any,
    tokenizer: Any,
    tinker_module: Any,
    examples: Sequence[MathExample],
    config: SFTConfig,
) -> EvaluationSummary:
    observations: list[EvaluationObservation] = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    for index, example in enumerate(examples):
        prompt_tokens = _eval_prompt_tokens(example, tokenizer)
        if len(prompt_tokens) > config.max_eval_prompt_tokens:
            raise SFTExperimentError("an evaluation prompt exceeds its token cap")
        result = await sampling_client.sample_async(
            prompt=tinker_module.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=tinker_module.SamplingParams(
                max_tokens=config.max_eval_output_tokens,
                temperature=0.0,
                seed=config.seed + index,
            ),
        )
        if not getattr(result, "sequences", None):
            raise SFTExperimentError("Tinker returned no evaluation sequence")
        output_tokens = list(result.sequences[0].tokens)
        if len(output_tokens) > config.max_eval_output_tokens:
            raise SFTExperimentError("Tinker returned more tokens than the eval cap")
        response_text = tokenizer.decode(output_tokens)
        parsed_answer = extract_final_answer(response_text)
        truncated = len(output_tokens) >= config.max_eval_output_tokens
        correct = bool(parsed_answer) and normalize_answer(
            parsed_answer
        ) == normalize_answer(example.final_answer)
        observations.append(
            EvaluationObservation(
                example_id=example.example_id,
                expected_answer=example.final_answer,
                parsed_answer=parsed_answer,
                correct=correct,
                truncated=truncated,
                prompt_tokens=len(prompt_tokens),
                output_tokens=len(output_tokens),
                response_text=response_text,
            )
        )
        total_prompt_tokens += len(prompt_tokens)
        total_output_tokens += len(output_tokens)

    count = len(observations)
    if count == 0:
        raise SFTExperimentError("evaluation set is empty")
    completed = [item for item in observations if not item.truncated]
    accuracy = sum(item.correct for item in observations) / count
    score_completed = (
        sum(item.correct for item in completed) / len(completed) if completed else 0.0
    )
    return EvaluationSummary(
        accuracy=accuracy,
        score_completed=score_completed,
        parse_rate=sum(bool(item.parsed_answer) for item in observations) / count,
        completion_rate=len(completed) / count,
        truncation_rate=sum(item.truncated for item in observations) / count,
        prompt_tokens=total_prompt_tokens,
        output_tokens=total_output_tokens,
        observations=tuple(observations),
    )


def _materialize_batch(
    examples: Sequence[TokenizedTrainingExample],
    tinker_module: Any,
) -> tuple[list[Any], int, int]:
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


def _mean_loss(result: Any, supervised_tokens: int) -> float:
    metrics = getattr(result, "metrics", {})
    if "loss:sum" in metrics:
        total_loss = float(metrics["loss:sum"])
    elif getattr(result, "loss", None) is not None:
        total_loss = float(result.loss)
    else:
        raise SFTExperimentError("Tinker did not return a readable loss diagnostic")
    if supervised_tokens <= 0:
        raise SFTExperimentError("training batch has no supervised tokens")
    return total_loss / supervised_tokens


def _batch_for_step(
    train: Sequence[TokenizedTrainingExample],
    step: int,
    batch_size: int,
) -> list[TokenizedTrainingExample]:
    start = ((step - 1) * batch_size) % len(train)
    return [train[(start + offset) % len(train)] for offset in range(batch_size)]


def _evaluation_metrics(prefix: str, summary: EvaluationSummary) -> dict[str, float]:
    return {
        f"{prefix}/accuracy": summary.accuracy,
        f"{prefix}/score_completed": summary.score_completed,
        f"{prefix}/parse_rate": summary.parse_rate,
        f"{prefix}/completion_rate": summary.completion_rate,
        f"{prefix}/truncation_rate": summary.truncation_rate,
    }


def quality_comparison_is_valid(
    baseline: EvaluationSummary,
    final: EvaluationSummary,
    min_completion_rate: float,
) -> bool:
    """Require both sides of a comparison to clear the completion-rate floor."""
    return (
        baseline.completion_rate >= min_completion_rate
        and final.completion_rate >= min_completion_rate
    )


async def run_sft_experiment(
    config: SFTConfig,
    allow_paid: bool,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
    candidates: Optional[Sequence[MathExample]] = None,
    load_dataset_fn: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = _print_progress,
) -> SFTTrainingReport:
    """Run matched baseline/final eval around configurable real-data SFT."""
    _require_paid_authorization(config, allow_paid, environ)
    progress(
        f"authorized model={config.model_id} steps={config.steps} "
        f"max_token_cost_usd={estimate_max_token_cost_usd(config):.6f}"
    )
    if candidates is None:
        progress(f"streaming dataset={config.dataset_id}@{config.dataset_revision[:8]}")
        candidates = load_dataset_candidates(config, load_dataset_fn=load_dataset_fn)

    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise SFTExperimentError(
                "Tinker SDK is unavailable; run with `uv run --extra tinker`"
            ) from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise SFTExperimentError("Weights & Biases is unavailable") from exc

    owned_http_client = None
    if service_client is None:
        import httpx

        owned_http_client = httpx.AsyncClient(follow_redirects=True)
        service_client = tinker_module.ServiceClient(
            user_metadata={"experiment": config.experiment_name},
            http_client=owned_http_client,
        )

    wandb_run = None
    try:
        base_sampling_client = await service_client.create_sampling_client_async(
            base_model=config.model_id
        )
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_id,
            rank=config.lora_rank,
            seed=config.seed,
            user_metadata={"experiment": config.experiment_name},
        )
        tokenizer = training_client.get_tokenizer()
        prepared = prepare_dataset(candidates, tokenizer, config)
        manifest_path = write_manifest(config, prepared)
        progress(
            f"data ready train={len(prepared.train)} eval={len(prepared.evaluation)} "
            f"skipped_too_long={prepared.skipped_too_long}"
        )

        project = environ.get("WANDB_PROJECT", config.wandb_project)
        entity = environ.get("WANDB_ENTITY") or None
        wandb_run = wandb_module.init(
            project=project,
            entity=entity,
            job_type="tinker-real-data-sft",
            tags=["tinker", "sft", "deepmath", "mvp"],
            config=asdict(config),
        )
        progress(f"W&B run={getattr(wandb_run, 'url', None)}")

        progress("evaluating unadapted baseline")
        baseline = await evaluate_sampling_client(
            base_sampling_client,
            tokenizer,
            tinker_module,
            prepared.evaluation,
            config,
        )
        wandb_run.log(_evaluation_metrics("eval/baseline", baseline), step=0)
        progress(
            f"baseline accuracy={baseline.accuracy:.4f} "
            f"truncation_rate={baseline.truncation_rate:.4f}"
        )

        cumulative_train_tokens = 0
        for step in range(1, config.steps + 1):
            selected = _batch_for_step(prepared.train, step, config.batch_size)
            data, batch_tokens, supervised_tokens = _materialize_batch(
                selected, tinker_module
            )
            started_at = clock()
            fwdbwd_future = await training_client.forward_backward_async(
                data=data,
                loss_fn="cross_entropy",
            )
            fwdbwd_result = await fwdbwd_future.result_async()
            optim_future = await training_client.optim_step_async(
                tinker_module.types.AdamParams(learning_rate=config.learning_rate)
            )
            await optim_future.result_async()
            cumulative_train_tokens += batch_tokens
            mean_loss = _mean_loss(fwdbwd_result, supervised_tokens)
            step_seconds = clock() - started_at
            estimated_train_cost = (
                cumulative_train_tokens * config.train_usd_per_million / 1_000_000
            )
            wandb_run.log(
                {
                    "train/loss": mean_loss,
                    "train/examples": config.batch_size,
                    "tokens/train_step": batch_tokens,
                    "tokens/cumulative_train": cumulative_train_tokens,
                    "cost/estimated_cumulative_train_usd": estimated_train_cost,
                    "timing/step_seconds": step_seconds,
                },
                step=step,
            )
            progress(
                f"step={step}/{config.steps} loss={mean_loss:.6g} "
                f"train_tokens={cumulative_train_tokens} "
                f"estimated_train_cost_usd={estimated_train_cost:.6f}"
            )

        run_id = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "-",
            str(getattr(wandb_run, "id", "") or int(time.time())),
        )
        checkpoint_name = f"{config.checkpoint_prefix}-{run_id}-state"
        sampler_name = f"{config.checkpoint_prefix}-{run_id}-sampler"
        progress(f"saving checkpoint={checkpoint_name}")
        state_future = await training_client.save_state_async(
            checkpoint_name,
            ttl_seconds=config.checkpoint_ttl_seconds,
        )
        state_result = await state_future.result_async()
        sampler_future = await training_client.save_weights_for_sampler_async(
            sampler_name,
            ttl_seconds=config.checkpoint_ttl_seconds,
        )
        sampler_result = await sampler_future.result_async()
        trained_sampling_client = await service_client.create_sampling_client_async(
            model_path=sampler_result.path
        )

        progress("evaluating trained checkpoint")
        final = await evaluate_sampling_client(
            trained_sampling_client,
            tokenizer,
            tinker_module,
            prepared.evaluation,
            config,
        )
        wandb_run.log(
            _evaluation_metrics("eval/final", final),
            step=config.steps + 1,
        )
        quality_comparison_valid = quality_comparison_is_valid(
            baseline,
            final,
            config.min_eval_completion_rate,
        )

        eval_prompt_tokens = baseline.prompt_tokens + final.prompt_tokens
        eval_output_tokens = baseline.output_tokens + final.output_tokens
        estimated_cost = estimate_actual_token_cost_usd(
            config,
            cumulative_train_tokens,
            eval_prompt_tokens,
            eval_output_tokens,
        )
        report_path = _output_path(config, f"run_report_{run_id}.json")
        report = SFTTrainingReport(
            mode="remote-real-data-sft",
            network_called=True,
            model_id=config.model_id,
            dataset_id=config.dataset_id,
            dataset_revision=config.dataset_revision,
            steps_completed=config.steps,
            train_examples=len(prepared.train),
            eval_examples=len(prepared.evaluation),
            train_tokens=cumulative_train_tokens,
            baseline_accuracy=baseline.accuracy,
            final_accuracy=final.accuracy,
            accuracy_gain=final.accuracy - baseline.accuracy,
            quality_comparison_valid=quality_comparison_valid,
            baseline_truncation_rate=baseline.truncation_rate,
            final_truncation_rate=final.truncation_rate,
            checkpoint_path=str(state_result.path),
            sampler_path=str(sampler_result.path),
            estimated_token_cost_usd=estimated_cost,
            hard_cap_usd=config.hard_cap_usd,
            wandb_project=project,
            wandb_run_url=getattr(wandb_run, "url", None),
            manifest_path=str(manifest_path),
            report_path=str(report_path),
        )
        _write_json(
            report_path,
            {
                "summary": asdict(report),
                "config": asdict(config),
                "baseline": asdict(baseline),
                "final": asdict(final),
            },
        )
        wandb_run.summary.update(
            {
                "eval/baseline_accuracy": baseline.accuracy,
                "eval/final_accuracy": final.accuracy,
                "eval/accuracy_gain": final.accuracy - baseline.accuracy,
                "eval/quality_comparison_valid": quality_comparison_valid,
                "checkpoint/path": str(state_result.path),
                "checkpoint/sampler_path": str(sampler_result.path),
                "cost/estimated_total_token_usd": estimated_cost,
            }
        )
        progress(
            f"complete baseline={baseline.accuracy:.4f} "
            f"final={final.accuracy:.4f} gain={report.accuracy_gain:.4f} "
            f"quality_valid={quality_comparison_valid} "
            f"estimated_total_token_cost_usd={estimated_cost:.6f}"
        )
        return report
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if owned_http_client is not None:
            await owned_http_client.aclose()


def prepare_data_locally(
    config: SFTConfig,
    load_dataset_fn: Optional[Callable[..., Any]] = None,
    tokenizer: Any = None,
) -> DatasetPreparationReport:
    """Download/stream only public HF assets and write a content-hash manifest."""
    candidates = load_dataset_candidates(config, load_dataset_fn=load_dataset_fn)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SFTExperimentError("transformers is unavailable") from exc
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    prepared = prepare_dataset(candidates, tokenizer, config)
    manifest_path = write_manifest(config, prepared)
    return DatasetPreparationReport(
        mode="hugging-face-data-preparation",
        network_called=True,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        candidates_loaded=len(candidates),
        train_examples=len(prepared.train),
        eval_examples=len(prepared.evaluation),
        skipped_too_long=prepared.skipped_too_long,
        manifest_path=str(manifest_path),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run a configurable real-data Tinker SFT pilot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML experiment config (defaults to the checked-in DeepMath pilot).",
    )
    parser.add_argument(
        "--steps",
        "--iterations",
        dest="steps",
        type=int,
        help="Override training.steps; e.g. --steps 2 for a paid smoke run.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-data",
        action="store_true",
        help="Stream HF data, tokenize locally, and write a manifest; no Tinker call.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run baseline eval, SFT, checkpointing, and final eval on Tinker.",
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Acknowledge explicit approval for the paid Tinker requests.",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    config = override_steps(load_config(args.config), args.steps)
    if args.run:
        report: Any = await run_sft_experiment(
            config,
            allow_paid=args.allow_paid,
        )
    elif args.prepare_data:
        if args.allow_paid:
            raise SFTExperimentError("--allow-paid requires --run")
        report = prepare_data_locally(config)
    else:
        if args.allow_paid:
            raise SFTExperimentError("--allow-paid requires --run")
        report = build_doctor_report(config)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except (SFTExperimentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
