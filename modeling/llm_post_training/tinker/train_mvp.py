"""Three-step, cost-gated Tinker SFT smoke test with W&B telemetry.

The default command is a local-only preflight. The remote path requires both
``--run`` and ``--allow-paid`` so importing this module or running it without
the explicit gate cannot spend Tinker credit.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from dotenv import load_dotenv

from modeling.llm_post_training.tinker.mvp import MIN_PYTHON, MODEL_ID, SMOKE_PROMPT


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
TINKER_SDK_PACKAGE = "tinker"
WANDB_PACKAGE = "wandb"
WANDB_PROJECT = "ml-playground-tinker"
TRAINING_STEPS = 3
LORA_RANK = 16
LEARNING_RATE = 1e-4
MAX_SEQUENCE_TOKENS = 128
MAX_SAMPLE_PROMPT_TOKENS = 128
MAX_OUTPUT_TOKENS = 32
HARD_CAP_USD = 0.01
PREFILL_USD_PER_MILLION = 0.33
SAMPLE_USD_PER_MILLION = 1.005
TRAIN_USD_PER_MILLION = 0.737

# These examples exist only to exercise the training path. They are not a
# benchmark or a proposed training dataset.
SFT_SMOKE_EXAMPLES = (
    ("Question: What is 8 + 7?\nAnswer:", " 15"),
    ("Question: What is 9 * 6?\nAnswer:", " 54"),
)


class TrainingMVPError(RuntimeError):
    """Raised when an MVP safety, readiness, or response check fails."""


def _print_progress(message: str) -> None:
    """Write human-readable live progress without mixing it into JSON stdout."""
    print(f"[tinker-mvp] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen bounds for the disposable three-step SFT run."""

    model_id: str = MODEL_ID
    training_steps: int = TRAINING_STEPS
    lora_rank: int = LORA_RANK
    learning_rate: float = LEARNING_RATE
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS
    max_sample_prompt_tokens: int = MAX_SAMPLE_PROMPT_TOKENS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    hard_cap_usd: float = HARD_CAP_USD
    prefill_usd_per_million: float = PREFILL_USD_PER_MILLION
    sample_usd_per_million: float = SAMPLE_USD_PER_MILLION
    train_usd_per_million: float = TRAIN_USD_PER_MILLION


@dataclass(frozen=True)
class TrainingDoctorReport:
    """Machine-readable local readiness report that never contains key values."""

    mode: str
    network_called: bool
    python_supported: bool
    tinker_sdk_available: bool
    tinker_sdk_version: Optional[str]
    wandb_available: bool
    wandb_version: Optional[str]
    tinker_api_key_configured: bool
    wandb_api_key_configured: bool
    model_id: str
    training_steps: int
    examples_per_step: int
    estimated_max_token_cost_usd: float
    hard_cap_usd: float
    ready_for_paid_run: bool


@dataclass(frozen=True)
class PreparedBatch:
    """Tokenized training data plus counts used for cost and metric logging."""

    data: list[Any]
    input_tokens: int
    supervised_tokens: int


@dataclass(frozen=True)
class SampleObservation:
    """One sample and its token counts."""

    response_text: str
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TrainingMVPReport:
    """Result of the remote, three-step SFT smoke test."""

    mode: str
    network_called: bool
    model_id: str
    steps_completed: int
    examples_per_step: int
    train_tokens: int
    before_output_tokens: int
    after_output_tokens: int
    estimated_token_cost_usd: float
    hard_cap_usd: float
    wandb_project: str
    wandb_run_url: Optional[str]
    before_response_text: str
    after_response_text: str


def load_local_env() -> None:
    """Load the ignored repository-root .env without overriding shell values."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _package_version(package: str) -> Optional[str]:
    if importlib.util.find_spec(package) is None:
        return None
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def estimate_max_token_cost_usd(config: TrainingConfig) -> float:
    """Conservative preflight estimate using only frozen local bounds."""
    train_tokens = (
        config.training_steps * len(SFT_SMOKE_EXAMPLES) * config.max_sequence_tokens
    )
    sample_prefill_tokens = 2 * config.max_sample_prompt_tokens
    sample_output_tokens = 2 * config.max_output_tokens
    return (
        train_tokens * config.train_usd_per_million
        + sample_prefill_tokens * config.prefill_usd_per_million
        + sample_output_tokens * config.sample_usd_per_million
    ) / 1_000_000


def estimate_actual_token_cost_usd(
    config: TrainingConfig,
    train_tokens: int,
    sample_prompt_tokens: int,
    sample_output_tokens: int,
) -> float:
    """Estimate token charges from observed counts and frozen public rates."""
    if min(train_tokens, sample_prompt_tokens, sample_output_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    return (
        train_tokens * config.train_usd_per_million
        + sample_prompt_tokens * config.prefill_usd_per_million
        + sample_output_tokens * config.sample_usd_per_million
    ) / 1_000_000


def build_doctor_report(
    config: TrainingConfig,
    environ: Mapping[str, str] = os.environ,
    tinker_version: Optional[str] = None,
    wandb_version: Optional[str] = None,
) -> TrainingDoctorReport:
    """Build a no-network readiness report without constructing either client."""
    detected_tinker = (
        _package_version(TINKER_SDK_PACKAGE)
        if tinker_version is None
        else tinker_version
    )
    detected_wandb = (
        _package_version(WANDB_PACKAGE) if wandb_version is None else wandb_version
    )
    python_supported = sys.version_info[:2] >= MIN_PYTHON
    tinker_key_configured = bool(environ.get("TINKER_API_KEY"))
    wandb_key_configured = bool(environ.get("WANDB_API_KEY"))
    estimated_max_cost = estimate_max_token_cost_usd(config)

    return TrainingDoctorReport(
        mode="local-training-preflight",
        network_called=False,
        python_supported=python_supported,
        tinker_sdk_available=detected_tinker is not None,
        tinker_sdk_version=detected_tinker,
        wandb_available=detected_wandb is not None,
        wandb_version=detected_wandb,
        tinker_api_key_configured=tinker_key_configured,
        wandb_api_key_configured=wandb_key_configured,
        model_id=config.model_id,
        training_steps=config.training_steps,
        examples_per_step=len(SFT_SMOKE_EXAMPLES),
        estimated_max_token_cost_usd=estimated_max_cost,
        hard_cap_usd=config.hard_cap_usd,
        ready_for_paid_run=(
            python_supported
            and detected_tinker is not None
            and detected_wandb is not None
            and tinker_key_configured
            and wandb_key_configured
            and estimated_max_cost <= config.hard_cap_usd
        ),
    )


def _require_paid_authorization(
    config: TrainingConfig,
    allow_paid: bool,
    environ: Mapping[str, str],
) -> None:
    if not allow_paid:
        raise TrainingMVPError(
            "training is blocked; pass --allow-paid only after explicit approval"
        )
    if not environ.get("TINKER_API_KEY"):
        raise TrainingMVPError("TINKER_API_KEY is not configured")
    if not environ.get("WANDB_API_KEY"):
        raise TrainingMVPError("WANDB_API_KEY is not configured")
    if estimate_max_token_cost_usd(config) > config.hard_cap_usd:
        raise TrainingMVPError(
            "estimated maximum token cost exceeds the configured hard cap"
        )


def prepare_training_batch(
    tokenizer: Any,
    tinker_module: Any,
    config: TrainingConfig,
) -> PreparedBatch:
    """Build two tiny next-token SFT examples with prompt-masked loss weights."""
    data = []
    total_input_tokens = 0
    total_supervised_tokens = 0

    for prompt_text, completion_text in SFT_SMOKE_EXAMPLES:
        prompt_tokens = list(tokenizer.encode(prompt_text))
        completion_tokens = list(tokenizer.encode(completion_text))
        if not prompt_tokens or not completion_tokens:
            raise TrainingMVPError("the smoke tokenizer produced an empty sequence")

        full_tokens = prompt_tokens + completion_tokens
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
        if len(input_tokens) > config.max_sequence_tokens:
            raise TrainingMVPError(
                "an SFT smoke example exceeds the sequence-token cap"
            )
        if not (len(input_tokens) == len(target_tokens) == len(weights)):
            raise TrainingMVPError("shifted SFT tensors have inconsistent lengths")

        datum = tinker_module.types.Datum(
            model_input=tinker_module.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "weights": weights,
                "target_tokens": target_tokens,
            },
        )
        data.append(datum)
        total_input_tokens += len(input_tokens)
        total_supervised_tokens += len(completion_tokens)

    return PreparedBatch(
        data=data,
        input_tokens=total_input_tokens,
        supervised_tokens=total_supervised_tokens,
    )


def _mean_cross_entropy_loss(result: Any, batch: PreparedBatch) -> float:
    """Read the documented loss diagnostic, with a logprob fallback."""
    metrics = getattr(result, "metrics", {})
    if "loss:sum" in metrics:
        total_loss = float(metrics["loss:sum"])
    elif getattr(result, "loss", None) is not None:
        total_loss = float(result.loss)
    else:
        total_loss = 0.0
        outputs = getattr(result, "loss_fn_outputs", [])
        if len(outputs) != len(batch.data):
            raise TrainingMVPError("Tinker did not return a readable loss diagnostic")
        for output, datum in zip(outputs, batch.data):
            logprobs = output["logprobs"].data
            weights = datum.loss_fn_inputs["weights"].data
            if len(logprobs) != len(weights):
                raise TrainingMVPError("loss output and weights have different lengths")
            total_loss += sum(
                -float(logprob) * float(weight)
                for logprob, weight in zip(logprobs, weights)
            )

    if batch.supervised_tokens <= 0:
        raise TrainingMVPError("the SFT smoke batch has no supervised tokens")
    return total_loss / batch.supervised_tokens


async def _sample_once(
    sampling_client: Any,
    tokenizer: Any,
    tinker_module: Any,
    config: TrainingConfig,
) -> SampleObservation:
    prompt_tokens = list(tokenizer.encode(SMOKE_PROMPT))
    if len(prompt_tokens) > config.max_sample_prompt_tokens:
        raise TrainingMVPError("the sample prompt exceeds the prompt-token cap")

    result = await sampling_client.sample_async(
        prompt=tinker_module.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=1,
        sampling_params=tinker_module.SamplingParams(
            max_tokens=config.max_output_tokens,
            temperature=0.0,
            seed=0,
        ),
    )
    output_tokens = list(result.sequences[0].tokens)
    if len(output_tokens) > config.max_output_tokens:
        raise TrainingMVPError("Tinker returned more tokens than the output-token cap")
    response_text = tokenizer.decode(output_tokens)
    if not response_text.strip():
        raise TrainingMVPError("Tinker returned an empty sample")
    return SampleObservation(
        response_text=response_text,
        prompt_tokens=len(prompt_tokens),
        output_tokens=len(output_tokens),
    )


async def run_training_mvp(
    config: TrainingConfig,
    allow_paid: bool,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = _print_progress,
) -> TrainingMVPReport:
    """Run baseline sample, three SFT updates, W&B logging, and trained sample."""
    _require_paid_authorization(config, allow_paid, environ)
    progress(
        f"authorized model={config.model_id} steps={config.training_steps} "
        f"max_token_cost_usd={estimate_max_token_cost_usd(config):.9f}"
    )

    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise TrainingMVPError(
                "Tinker SDK is unavailable; run with `uv run --extra tinker`"
            ) from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise TrainingMVPError("Weights & Biases is unavailable") from exc

    if service_client is None:
        import httpx

        # Tinker's default pyqwest transport can miss macOS trust-store issuers.
        # Standard HTTPX keeps TLS verification enabled and is supported by the
        # SDK's ServiceClient ``http_client`` escape hatch.
        progress("connecting to Tinker with verified HTTPX transport")
        service_client = tinker_module.ServiceClient(
            user_metadata={"experiment": "ml-playground-tinker-sft-wandb-mvp"},
            http_client=httpx.AsyncClient(follow_redirects=True),
        )

    base_sampling_client = await service_client.create_sampling_client_async(
        base_model=config.model_id
    )
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_id,
        rank=config.lora_rank,
        seed=0,
        user_metadata={"experiment": "three-step-sft-wandb-mvp"},
    )
    tokenizer = training_client.get_tokenizer()
    batch = prepare_training_batch(tokenizer, tinker_module, config)
    train_tokens_per_step = batch.input_tokens
    progress(
        f"clients ready examples_per_step={len(batch.data)} "
        f"train_tokens_per_step={train_tokens_per_step}"
    )
    project = environ.get("WANDB_PROJECT", WANDB_PROJECT)
    entity = environ.get("WANDB_ENTITY") or None

    wandb_run = None
    try:
        progress(f"initializing W&B project={project}")
        wandb_run = wandb_module.init(
            project=project,
            entity=entity,
            job_type="tinker-sft-mvp",
            tags=["tinker", "sft", "mvp"],
            config={
                "model_id": config.model_id,
                "training_steps": config.training_steps,
                "examples_per_step": len(batch.data),
                "lora_rank": config.lora_rank,
                "learning_rate": config.learning_rate,
                "max_sequence_tokens": config.max_sequence_tokens,
                "max_output_tokens": config.max_output_tokens,
                "hard_cap_usd": config.hard_cap_usd,
            },
        )
        if getattr(wandb_run, "url", None):
            progress(f"W&B run={wandb_run.url}")

        progress("sampling unadapted model")
        before = await _sample_once(
            base_sampling_client, tokenizer, tinker_module, config
        )
        progress(
            f"baseline sample complete prompt_tokens={before.prompt_tokens} "
            f"output_tokens={before.output_tokens}"
        )
        cumulative_train_tokens = 0
        for step in range(1, config.training_steps + 1):
            started_at = clock()
            fwdbwd_future = await training_client.forward_backward_async(
                data=batch.data,
                loss_fn="cross_entropy",
            )
            fwdbwd_result = await fwdbwd_future.result_async()
            optim_future = await training_client.optim_step_async(
                tinker_module.types.AdamParams(learning_rate=config.learning_rate)
            )
            await optim_future.result_async()
            step_seconds = clock() - started_at
            cumulative_train_tokens += train_tokens_per_step
            mean_loss = _mean_cross_entropy_loss(fwdbwd_result, batch)
            estimated_cumulative_train_cost = (
                cumulative_train_tokens * config.train_usd_per_million / 1_000_000
            )
            wandb_run.log(
                {
                    "train/step": step,
                    "train/loss": mean_loss,
                    "train/examples": len(batch.data),
                    "tokens/train_step": train_tokens_per_step,
                    "tokens/cumulative_train": cumulative_train_tokens,
                    "cost/estimated_step_usd": (
                        train_tokens_per_step * config.train_usd_per_million / 1_000_000
                    ),
                    "cost/estimated_cumulative_train_usd": (
                        estimated_cumulative_train_cost
                    ),
                    "timing/step_seconds": step_seconds,
                },
                step=step,
            )
            progress(
                f"step={step}/{config.training_steps} loss={mean_loss:.10g} "
                f"cumulative_train_tokens={cumulative_train_tokens} "
                f"step_seconds={step_seconds:.3f} "
                f"estimated_train_cost_usd={estimated_cumulative_train_cost:.8f}"
            )

        progress("sampling trained ephemeral checkpoint")
        trained_sampling_client = training_client.save_weights_and_get_sampling_client()
        after = await _sample_once(
            trained_sampling_client, tokenizer, tinker_module, config
        )
        estimated_cost = estimate_actual_token_cost_usd(
            config,
            train_tokens=cumulative_train_tokens,
            sample_prompt_tokens=before.prompt_tokens + after.prompt_tokens,
            sample_output_tokens=before.output_tokens + after.output_tokens,
        )
        progress(
            f"trained sample complete prompt_tokens={after.prompt_tokens} "
            f"output_tokens={after.output_tokens}"
        )
        progress(f"complete estimated_total_token_cost_usd={estimated_cost:.8f}")
        wandb_run.summary.update(
            {
                "sample/before_text": before.response_text,
                "sample/after_text": after.response_text,
                "sample/before_output_tokens": before.output_tokens,
                "sample/after_output_tokens": after.output_tokens,
                "cost/estimated_total_token_usd": estimated_cost,
            }
        )

        return TrainingMVPReport(
            mode="remote-sft-wandb-mvp",
            network_called=True,
            model_id=config.model_id,
            steps_completed=config.training_steps,
            examples_per_step=len(batch.data),
            train_tokens=cumulative_train_tokens,
            before_output_tokens=before.output_tokens,
            after_output_tokens=after.output_tokens,
            estimated_token_cost_usd=estimated_cost,
            hard_cap_usd=config.hard_cap_usd,
            wandb_project=project,
            wandb_run_url=getattr(wandb_run, "url", None),
            before_response_text=before.response_text,
            after_response_text=after.response_text,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local training preflight or gated three-step SFT MVP."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the remote three-step Tinker SFT and W&B smoke test.",
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Acknowledge explicit approval for the remote paid requests.",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    config = TrainingConfig()
    if args.run:
        report = await run_training_mvp(config, allow_paid=args.allow_paid)
    else:
        if args.allow_paid:
            raise TrainingMVPError("--allow-paid requires --run")
        report = build_doctor_report(config)

    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except TrainingMVPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
