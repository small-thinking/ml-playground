"""One-epoch, cost-gated GSM8K SFT on frozen train and validation splits."""

from __future__ import annotations

import argparse
import asyncio
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
    MODEL_ID,
    PROMPT_VERSION,
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
from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import normalize_number


EXPERIMENT_ID = "e1"
LORA_RANK = 32
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
MAX_SEQUENCE_TOKENS = 1024
VALIDATION_EVERY = 250
PROGRESS_EVERY = 25
CHECKPOINT_TTL_SECONDS = 30 * 24 * 60 * 60
HARD_CAP_USD = 12.0
TRAIN_USD_PER_MILLION = 1.463
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
OUTPUT_DIR = Path(__file__).parent / "outputs"


class SFTTrainingError(RuntimeError):
    """Raised when an SFT run would be invalid, unsafe, or incomplete."""


@dataclass(frozen=True)
class SFTConfig:
    """Fixed conditions for the first GSM8K SFT experiment."""

    model_id: str = MODEL_ID
    project: str = WANDB_PROJECT
    suite_id: str = SUITE_ID
    experiment_id: str = EXPERIMENT_ID
    attempt: int = 1
    lora_rank: int = LORA_RANK
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS
    validation_every: int = VALIDATION_EVERY
    progress_every: int = PROGRESS_EVERY
    checkpoint_ttl_seconds: int = CHECKPOINT_TTL_SECONDS
    hard_cap_usd: float = HARD_CAP_USD
    train_usd_per_million: float = TRAIN_USD_PER_MILLION

    def validate(self, manifest: Optional[SplitManifest] = None) -> None:
        if not self.model_id or not self.project or not self.suite_id:
            raise SFTTrainingError("model, project, and suite identifiers are required")
        if self.experiment_id != EXPERIMENT_ID:
            raise SFTTrainingError("the first SFT run must use experiment ID e1")
        positive_ints = {
            "attempt": self.attempt,
            "lora_rank": self.lora_rank,
            "batch_size": self.batch_size,
            "max_sequence_tokens": self.max_sequence_tokens,
            "validation_every": self.validation_every,
            "progress_every": self.progress_every,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise SFTTrainingError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise SFTTrainingError("learning_rate must be positive")
        if self.hard_cap_usd <= 0 or self.train_usd_per_million <= 0:
            raise SFTTrainingError("cost settings must be positive")
        if manifest is not None:
            manifest.validate()
            if len(manifest.sft_train_ids) < self.batch_size:
                raise SFTTrainingError("sft_train must contain at least one batch")

    @property
    def run_name(self) -> str:
        model_slug = self.model_id.lower().replace("/", "-").replace(".", "-")
        return (
            f"{self.experiment_id}-sft-{model_slug}-r{self.lora_rank}"
            f"-b{self.batch_size}-a{self.attempt:02d}"
        )

    def training_steps(self, manifest: SplitManifest) -> int:
        self.validate(manifest)
        return math.ceil(len(manifest.sft_train_ids) / self.batch_size)

    def validation_steps(self, manifest: SplitManifest) -> Tuple[int, ...]:
        steps = self.training_steps(manifest)
        scheduled = tuple(range(self.validation_every, steps + 1, self.validation_every))
        return scheduled if scheduled and scheduled[-1] == steps else scheduled + (steps,)


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
    """Bound one training epoch plus baseline and checkpoint validations."""
    config.validate(manifest)
    train_tokens = (
        config.training_steps(manifest)
        * config.batch_size
        * config.max_sequence_tokens
    )
    validation_tokens = (
        (1 + len(config.validation_steps(manifest)))
        * len(manifest.sft_validation_ids)
        * config.max_sequence_tokens
    )
    return (train_tokens + validation_tokens) * config.train_usd_per_million / 1_000_000


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
    tinker_sdk = _package_version("tinker") if tinker_version is None else tinker_version
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
        raise SFTTrainingError("WANDB_MODE=offline cannot produce the required dashboard")
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
    return f"{reasoning}\n\\boxed{{{final_answer}}}" if reasoning else f"\\boxed{{{final_answer}}}"


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
        "max_sequence_tokens": config.max_sequence_tokens,
        "training_steps": config.training_steps(manifest),
        "validation_steps": list(config.validation_steps(manifest)),
        "train_usd_per_million": config.train_usd_per_million,
        "hard_cap_usd": config.hard_cap_usd,
        "git_sha": _git_sha(),
        "hypothesis": "One clean SFT epoch improves answer format and GSM8K accuracy.",
        "expected_failure": "Validation NLL improves without formal generation gain.",
    }


def _estimated_cost(processed_tokens: int, config: SFTConfig) -> float:
    return processed_tokens * config.train_usd_per_million / 1_000_000


def _report_path(run_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_id)
    return OUTPUT_DIR / f"e1_sft_report_{safe_id}.json"


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
    """Run one frozen SFT epoch and select its lowest-validation-NLL checkpoint."""
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
            seed=20260901,
            user_metadata={"experiment_id": config.experiment_id},
        )
        tokenizer = training_client.get_tokenizer()
        train = prepare_sft_examples(
            train_rows, tokenizer, config.max_sequence_tokens
        )
        validation = prepare_sft_examples(
            validation_rows, tokenizer, config.max_sequence_tokens
        )
        progress(f"data ready train={len(train)} validation={len(validation)}")
        wandb_run = wandb_module.init(
            project=config.project,
            entity=environ.get("WANDB_ENTITY") or None,
            name=config.run_name,
            group=config.suite_id,
            job_type="sft-training",
            tags=["gsm8k", "sft", "e1", f"rank-{config.lora_rank}"],
            config=_tracking_config(config, manifest),
        )
        progress(f"started W&B run={getattr(wandb_run, 'url', None)}")

        processed_tokens = 0
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
                "run_stats/estimated_cumulative_usd": _estimated_cost(
                    processed_tokens, config
                ),
            },
            step=0,
        )
        progress(
            f"validation step=0/{config.training_steps(manifest)} "
            f"nll={baseline_nll:.5f} perplexity={_perplexity(baseline_nll):.3f}"
        )

        checkpoints: list[CheckpointRecord] = []
        validation_steps = set(config.validation_steps(manifest))
        train_batches = _batches(train, config.batch_size)
        total_steps = len(train_batches)
        for step, batch in enumerate(train_batches, start=1):
            data, batch_tokens, supervised_tokens = _materialize_batch(
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
            processed_tokens += batch_tokens
            nll = _loss_sum(forward_backward_result) / supervised_tokens
            step_seconds = max(clock() - step_started_at, 1e-9)
            elapsed_seconds = clock() - elapsed_started_at
            steps_per_second = step / max(elapsed_seconds, 1e-9)
            eta_seconds = (total_steps - step) / steps_per_second
            metrics = {
                "train/nll": nll,
                "train/perplexity": _perplexity(nll),
                "train/learning_rate": config.learning_rate,
                "train/supervised_tokens": float(supervised_tokens),
                "train/tokens_per_second": batch_tokens / step_seconds,
                "timing/step_seconds": step_seconds,
                "timing/elapsed_seconds": elapsed_seconds,
                "timing/eta_seconds": eta_seconds,
                "run_stats/cumulative_processed_tokens": float(processed_tokens),
                "run_stats/estimated_cumulative_usd": _estimated_cost(
                    processed_tokens, config
                ),
            }
            wandb_run.log(metrics, step=step)
            if (
                step == 1
                or step % config.progress_every == 0
                or step == total_steps
            ):
                progress(
                    f"step={step}/{total_steps} nll={nll:.5f} "
                    f"perplexity={_perplexity(nll):.3f} lr={config.learning_rate:.2g} "
                    f"throughput={batch_tokens / step_seconds:.1f}tok/s "
                    f"elapsed={elapsed_seconds:.1f}s eta={eta_seconds:.1f}s "
                    f"estimated_cost=${_estimated_cost(processed_tokens, config):.4f}"
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
            record = CheckpointRecord(
                step=step,
                nll=validation_nll,
                perplexity=validation_ppl,
                state_path=str(state_result.path),
                sampler_path=str(sampler_result.path),
            )
            checkpoints.append(record)
            wandb_run.log(
                {
                    "sft_validation/nll": validation_nll,
                    "sft_validation/perplexity": validation_ppl,
                    "sft_validation/is_baseline": 0.0,
                    "sft_validation/nll_delta_from_base": validation_nll - baseline_nll,
                    "run_stats/cumulative_processed_tokens": float(processed_tokens),
                    "run_stats/estimated_cumulative_usd": _estimated_cost(
                        processed_tokens, config
                    ),
                },
                step=step,
            )
            progress(
                f"validation step={step}/{total_steps} nll={validation_nll:.5f} "
                f"perplexity={validation_ppl:.3f} "
                f"estimated_cost=${_estimated_cost(processed_tokens, config):.4f}"
            )

        if not checkpoints:
            raise SFTTrainingError("training completed without a validation checkpoint")
        selected = min(checkpoints, key=lambda record: record.nll)
        total_estimated_cost = _estimated_cost(processed_tokens, config)
        if total_estimated_cost > config.hard_cap_usd:
            raise SFTTrainingError("observed token cost exceeded the configured hard cap")
        report = {
            "mode": "remote-sft-training",
            "network_called": True,
            "run_name": config.run_name,
            "model_id": config.model_id,
            "manifest_hash": manifest.manifest_hash,
            "sft_train_examples": len(train),
            "sft_validation_examples": len(validation),
            "training_steps": total_steps,
            "baseline_validation_nll": baseline_nll,
            "baseline_validation_perplexity": _perplexity(baseline_nll),
            "selected_checkpoint": asdict(selected),
            "validation_checkpoints": [asdict(record) for record in checkpoints],
            "estimated_token_cost_usd": total_estimated_cost,
            "hard_cap_usd": config.hard_cap_usd,
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _report_path(str(getattr(wandb_run, "id", "run")))
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        wandb_run.summary.update(
            {
                "sft_validation/baseline_nll": baseline_nll,
                "sft_validation/best_nll": selected.nll,
                "sft_validation/best_perplexity": selected.perplexity,
                "checkpoint/selected_step": selected.step,
                "checkpoint/selected_state_path": selected.state_path,
                "checkpoint/selected_sampler_path": selected.sampler_path,
                "run_stats/estimated_total_usd": total_estimated_cost,
            }
        )
        progress(
            f"complete selected_step={selected.step} best_validation_nll={selected.nll:.5f} "
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
        description="Preflight or run the frozen GSM8K E1 SFT experiment."
    )
    parser.add_argument("--run", action="store_true", help="Start the paid SFT run.")
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Acknowledge approval for the cost-gated Tinker request.",
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--hard-cap-usd", type=float, default=HARD_CAP_USD)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> Dict[str, Any]:
    config = replace(
        SFTConfig(),
        attempt=args.attempt,
        hard_cap_usd=args.hard_cap_usd,
        progress_every=args.progress_every,
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
