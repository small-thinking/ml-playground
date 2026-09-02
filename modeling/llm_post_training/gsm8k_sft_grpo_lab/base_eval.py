"""Cost-gated Base-model calibration on the frozen GSM8K evaluation split."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv

from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    SplitManifest,
    content_id,
    load_official_eval_rows,
    read_manifest,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import (
    Completion,
    evaluate_groups,
)


MODEL_ID = "Qwen/Qwen3.5-9B-Base"
WANDB_PROJECT = "mini-posttraining-lab"
SUITE_ID = "gsm8k-sft-grpo-v1"
EXPERIMENT_ID = "e0a"
PROMPT_VERSION = "gsm8k-raw-completion-v1"
PARSER_VERSION = "numeric-boxed-v1"
EVALUATION_VERSION = "gsm8k-eval-v1"
CALIBRATION_EXAMPLES = 32
GROUP_SIZE = 4
TEMPERATURE = 1.0
MAX_PROMPT_TOKENS = 512
MAX_OUTPUT_TOKENS = 512
SEED = 20260901
HARD_CAP_USD = 0.25
PREFILL_USD_PER_MILLION = 0.66
SAMPLE_USD_PER_MILLION = 1.995
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"

PROMPT_TEMPLATE = """Solve the following grade-school math problem.
Show concise reasoning, then put the final numeric answer in \\boxed{{...}}.

Question: {question}

Answer:"""

METRIC_KEYS = (
    "eval/exact_match",
    "eval/pass_at_4",
    "eval/format_accuracy",
    "eval/avg_output_tokens",
    "eval/truncation_rate",
    "eval/group_all_correct_frac",
    "eval/group_all_wrong_frac",
    "eval/group_mixed_frac",
    "eval/group_reward_std_mean",
    "eval/process_check_coverage",
    "eval/process_validity_rate",
    "eval/final_correct_process_invalid",
    "eval/final_correct_process_valid",
)
TABLE_COLUMNS = (
    "example_id",
    "group_id",
    "rollout_id",
    "question",
    "ground_truth",
    "model_id",
    "checkpoint",
    "experiment_id",
    "generated_response",
    "parsed_answer",
    "correct",
    "output_tokens",
    "format_valid",
    "truncated",
    "process_checked_steps",
    "process_valid_steps",
    "process_invalid_steps",
    "reward",
    "group_mean_reward",
    "advantage",
)


class BaseEvalError(RuntimeError):
    """Raised when the calibration cannot run safely."""


@dataclass(frozen=True)
class BaseEvalConfig:
    """Conditions that must remain fixed for a comparable evaluation."""

    model_id: str = MODEL_ID
    project: str = WANDB_PROJECT
    suite_id: str = SUITE_ID
    experiment_id: str = EXPERIMENT_ID
    attempt: int = 1
    eval_examples: int = CALIBRATION_EXAMPLES
    group_size: int = GROUP_SIZE
    temperature: float = TEMPERATURE
    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    seed: int = SEED
    hard_cap_usd: float = HARD_CAP_USD
    prefill_usd_per_million: float = PREFILL_USD_PER_MILLION
    sample_usd_per_million: float = SAMPLE_USD_PER_MILLION

    def validate(self, manifest: Optional[SplitManifest] = None) -> None:
        if not self.model_id or not self.project or not self.suite_id:
            raise BaseEvalError("model, project, and suite identifiers are required")
        if self.attempt <= 0 or self.eval_examples <= 0 or self.group_size != 4:
            raise BaseEvalError("evaluation requires a positive example count and G=4")
        if min(self.temperature, self.max_prompt_tokens, self.max_output_tokens) <= 0:
            raise BaseEvalError("decoding limits and temperature must be positive")
        if self.hard_cap_usd <= 0:
            raise BaseEvalError("hard cap must be positive")
        if manifest is not None and self.eval_examples > len(manifest.eval_ids):
            raise BaseEvalError("evaluation exceeds the frozen held-out split")

    @property
    def run_name(self) -> str:
        model_slug = self.model_id.lower().replace("/", "-").replace(".", "-")
        return (
            f"{self.experiment_id}-base-calibration-{model_slug}"
            f"-g{self.group_size}-a{self.attempt:02d}"
        )


def load_local_env() -> None:
    """Load ignored repository credentials without replacing shell values."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def estimate_token_cost_usd(
    prompt_tokens: int, output_tokens: int, config: BaseEvalConfig
) -> float:
    """Estimate public, uncached token charges."""
    if min(prompt_tokens, output_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    return (
        prompt_tokens * config.prefill_usd_per_million
        + output_tokens * config.sample_usd_per_million
    ) / 1_000_000


def estimate_max_token_cost_usd(config: BaseEvalConfig) -> float:
    """Bound all requested rollouts before constructing a remote client."""
    rollouts = config.eval_examples * config.group_size
    return estimate_token_cost_usd(
        rollouts * config.max_prompt_tokens,
        rollouts * config.max_output_tokens,
        config,
    )


def build_doctor_report(
    config: BaseEvalConfig,
    environ: Mapping[str, str] = os.environ,
    tinker_version: Optional[str] = None,
    wandb_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Check local prerequisites without contacting any remote service."""
    config.validate(read_manifest())
    tinker_sdk = (
        _package_version("tinker") if tinker_version is None else tinker_version
    )
    wandb_sdk = _package_version("wandb") if wandb_version is None else wandb_version
    estimated_cost = estimate_max_token_cost_usd(config)
    hf_token = bool(environ.get("HF_TOKEN"))
    tinker_key = bool(environ.get("TINKER_API_KEY"))
    wandb_key = bool(environ.get("WANDB_API_KEY"))
    return {
        "mode": "local-base-eval-preflight",
        "network_called": False,
        "model_id": config.model_id,
        "tinker_sdk_version": tinker_sdk,
        "wandb_version": wandb_sdk,
        "hf_token_configured": hf_token,
        "tinker_api_key_configured": tinker_key,
        "wandb_api_key_configured": wandb_key,
        "estimated_max_token_cost_usd": estimated_cost,
        "hard_cap_usd": config.hard_cap_usd,
        "ready_for_paid_run": (
            sys.version_info[:2] >= (3, 11)
            and tinker_sdk is not None
            and wandb_sdk is not None
            and tinker_key
            and wandb_key
            and environ.get("WANDB_MODE", "").lower() != "offline"
            and estimated_cost <= config.hard_cap_usd
        ),
    }


def _authorize(
    config: BaseEvalConfig, allow_paid: bool, environ: Mapping[str, str]
) -> None:
    config.validate()
    if not allow_paid:
        raise BaseEvalError("evaluation is blocked; pass --allow-paid after approval")
    if not environ.get("TINKER_API_KEY") or not environ.get("WANDB_API_KEY"):
        raise BaseEvalError("TINKER_API_KEY and WANDB_API_KEY are required")
    if environ.get("WANDB_MODE", "").lower() == "offline":
        raise BaseEvalError("WANDB_MODE=offline cannot produce the required dashboard")
    if estimate_max_token_cost_usd(config) > config.hard_cap_usd:
        raise BaseEvalError("estimated maximum token cost exceeds the hard cap")


def build_prompt(question: str) -> str:
    """Render the raw-completion prompt shared by comparable stages."""
    return PROMPT_TEMPLATE.format(question=question.strip())


def evaluation_protocol_id(config: BaseEvalConfig, manifest: SplitManifest) -> str:
    """Hash every condition that changes which metric values are comparable."""
    config.validate(manifest)
    payload = {
        "dataset_revision": manifest.dataset_revision,
        "evaluated_ids_hash": hashlib.sha256(
            "\n".join(manifest.eval_ids[: config.eval_examples]).encode()
        ).hexdigest(),
        "eval_examples": config.eval_examples,
        "group_size": config.group_size,
        "manifest_hash": manifest.manifest_hash,
        "max_output_tokens": config.max_output_tokens,
        "max_prompt_tokens": config.max_prompt_tokens,
        "metric_keys": METRIC_KEYS,
        "parser_version": PARSER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "temperature": config.temperature,
        "version": EVALUATION_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{EVALUATION_VERSION}-{hashlib.sha256(encoded).hexdigest()[:12]}"


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


def _tracking_config(config: BaseEvalConfig, manifest: SplitManifest) -> Dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "attempt": config.attempt,
        "suite_id": config.suite_id,
        "checkpoint": "base",
        "parent_checkpoint": None,
        "model_id": config.model_id,
        "model_revision": "not exposed by Tinker sampling API",
        "dataset_id": manifest.dataset_id,
        "dataset_revision": manifest.dataset_revision,
        "manifest_hash": manifest.manifest_hash,
        "evaluation_protocol_id": evaluation_protocol_id(config, manifest),
        "eval_examples": config.eval_examples,
        "group_size": config.group_size,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "temperature": config.temperature,
        "max_prompt_tokens": config.max_prompt_tokens,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "prefill_usd_per_million": config.prefill_usd_per_million,
        "sample_usd_per_million": config.sample_usd_per_million,
        "hard_cap_usd": config.hard_cap_usd,
        "git_sha": _git_sha(),
        "hypothesis": "Raw Base calibration establishes a post-training baseline.",
        "expected_failure": "Low format accuracy or too few mixed rollout groups.",
    }


async def _sample_group(
    row: Mapping[str, object],
    index: int,
    client: Any,
    tokenizer: Any,
    tinker: Any,
    config: BaseEvalConfig,
) -> Dict[str, Any]:
    prompt_tokens = list(tokenizer.encode(build_prompt(str(row["question"]))))
    if len(prompt_tokens) > config.max_prompt_tokens:
        raise BaseEvalError("a prompt exceeds the configured prompt-token limit")
    result = await client.sample_async(
        prompt=tinker.ModelInput.from_ints(tokens=prompt_tokens),
        num_samples=config.group_size,
        sampling_params=tinker.SamplingParams(
            max_tokens=config.max_output_tokens,
            temperature=config.temperature,
            seed=config.seed + index,
        ),
    )
    if len(result.sequences) != config.group_size:
        raise BaseEvalError("Tinker returned the wrong number of rollout samples")
    responses = tuple(
        (tokenizer.decode(sequence.tokens), len(sequence.tokens))
        for sequence in result.sequences
    )
    if any(tokens > config.max_output_tokens for _, tokens in responses):
        raise BaseEvalError("Tinker exceeded the configured output-token limit")
    return {
        "example_id": content_id(row),
        "question": str(row["question"]),
        "ground_truth": str(row["answer"]),
        "prompt_tokens": len(prompt_tokens),
        "responses": responses,
    }


def _prediction_rows(
    scored_rows: Sequence[Any],
    samples: Sequence[Mapping[str, Any]],
    config: BaseEvalConfig,
) -> Tuple[Tuple[Any, ...], ...]:
    sample_by_id = {sample["example_id"]: sample for sample in samples}
    rewards: Dict[str, list[float]] = {}
    for row in scored_rows:
        rewards.setdefault(row.group_id, []).append(float(row.correct))
    group_means = {
        group_id: sum(values) / len(values) for group_id, values in rewards.items()
    }
    return tuple(
        (
            row.example_id,
            row.group_id,
            rollout_id % config.group_size,
            sample_by_id[row.example_id]["question"],
            sample_by_id[row.example_id]["ground_truth"],
            config.model_id,
            "base",
            config.experiment_id,
            row.response,
            row.parsed_answer,
            row.correct,
            row.output_tokens,
            row.format_valid,
            row.truncated,
            row.process.checked_steps,
            row.process.valid_steps,
            row.process.invalid_steps,
            float(row.correct),
            group_means[row.group_id],
            float(row.correct) - group_means[row.group_id],
        )
        for rollout_id, row in enumerate(scored_rows)
    )


async def run_remote_evaluation(
    config: BaseEvalConfig,
    manifest: SplitManifest,
    rows: Sequence[Mapping[str, object]],
    allow_paid: bool,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
) -> Dict[str, Any]:
    """Sample `G=4` rollouts, score them, and write one W&B run."""
    _authorize(config, allow_paid, environ)
    config.validate(manifest)
    if len(rows) < config.eval_examples:
        raise BaseEvalError("loaded rows do not cover the requested evaluation")
    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise BaseEvalError(
                "Tinker SDK is unavailable; run with `--extra tinker`"
            ) from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise BaseEvalError("Weights & Biases is unavailable") from exc
    if service_client is None:
        import httpx

        service_client = tinker_module.ServiceClient(
            user_metadata={
                "experiment_id": config.experiment_id,
                "suite_id": config.suite_id,
            },
            http_client=httpx.AsyncClient(follow_redirects=True),
        )

    wandb_run = None
    try:
        wandb_run = wandb_module.init(
            project=config.project,
            entity=environ.get("WANDB_ENTITY") or None,
            name=config.run_name,
            group=config.suite_id,
            job_type="evaluation",
            tags=["gsm8k", "base", "calibration", "g4"],
            config=_tracking_config(config, manifest),
        )
        client = await service_client.create_sampling_client_async(
            base_model=config.model_id
        )
        tokenizer = client.get_tokenizer()
        samples = await asyncio.gather(
            *(
                _sample_group(row, index, client, tokenizer, tinker_module, config)
                for index, row in enumerate(rows[: config.eval_examples])
            )
        )
        groups = {
            sample["example_id"]: tuple(
                Completion(
                    example_id=sample["example_id"],
                    response=response,
                    ground_truth=sample["ground_truth"],
                    output_tokens=output_tokens,
                    max_output_tokens=config.max_output_tokens,
                )
                for response, output_tokens in sample["responses"]
            )
            for sample in samples
        }
        scored = evaluate_groups(groups, pass_k=config.group_size)
        prompt_total = sum(
            sample["prompt_tokens"] * config.group_size for sample in samples
        )
        output_total = sum(
            tokens for sample in samples for _, tokens in sample["responses"]
        )
        actual_cost = estimate_token_cost_usd(prompt_total, output_total, config)
        if actual_cost > config.hard_cap_usd:
            raise BaseEvalError("observed token cost exceeded the configured hard cap")
        metrics = dict(scored.metrics)
        metrics.update(
            {
                "eval/examples": float(len(samples)),
                "eval/rollouts": float(len(scored.rows)),
                "tokens/prompt": float(prompt_total),
                "tokens/output": float(output_total),
                "cost/estimated_max_token_usd": estimate_max_token_cost_usd(config),
                "cost/estimated_actual_token_usd": actual_cost,
            }
        )
        wandb_run.log(metrics)
        wandb_run.log(
            {
                "eval/rollouts_table": wandb_module.Table(
                    columns=TABLE_COLUMNS,
                    data=list(_prediction_rows(scored.rows, samples, config)),
                )
            }
        )
        wandb_run.summary.update(metrics)
        return {
            "mode": "remote-base-eval",
            "network_called": True,
            "run_name": config.run_name,
            "model_id": config.model_id,
            "evaluation_protocol_id": evaluation_protocol_id(config, manifest),
            "evaluated_examples": len(samples),
            "generated_rollouts": len(scored.rows),
            "estimated_token_cost_usd": actual_cost,
            "hard_cap_usd": config.hard_cap_usd,
            "wandb_run_url": getattr(wandb_run, "url", None),
            "metrics": scored.metrics,
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()


async def run_e0a(config: BaseEvalConfig, allow_paid: bool) -> Dict[str, Any]:
    """Load pinned rows only after passing the paid-run safety gate."""
    _authorize(config, allow_paid, os.environ)
    manifest = read_manifest()
    config.validate(manifest)
    return await run_remote_evaluation(
        config,
        manifest,
        load_official_eval_rows(manifest),
        allow_paid=allow_paid,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GSM8K Base-model calibration."
    )
    parser.add_argument(
        "--run", action="store_true", help="Run the remote E0a calibration."
    )
    parser.add_argument(
        "--allow-paid", action="store_true", help="Acknowledge paid use."
    )
    parser.add_argument(
        "--attempt", type=int, default=1, help="Record a retry explicitly."
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=CALIBRATION_EXAMPLES,
        help="Use a prefix of frozen held-out IDs; 32 is the E0a calibration.",
    )
    parser.add_argument(
        "--hard-cap-usd",
        type=float,
        default=HARD_CAP_USD,
        help="Block the run when its worst-case token estimate exceeds this USD cap.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        load_local_env()
        args = parse_args(argv)
        config = BaseEvalConfig(
            eval_examples=args.eval_examples,
            attempt=args.attempt,
            hard_cap_usd=args.hard_cap_usd,
        )
        if args.run:
            report = asyncio.run(run_e0a(config, allow_paid=args.allow_paid))
        else:
            if args.allow_paid:
                raise BaseEvalError("--allow-paid requires --run")
            report = build_doctor_report(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except BaseEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
