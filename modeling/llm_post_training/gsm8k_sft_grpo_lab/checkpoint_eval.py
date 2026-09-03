"""Formal GSM8K evaluation for a selected post-training sampler checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval import (
    FORMAL_EXAMPLES,
    PROGRESS_EVERY,
    BaseEvalConfig,
    BaseEvalError,
    build_doctor_report,
    load_local_env,
    run_remote_evaluation,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    SplitManifest,
    load_official_test_rows,
    read_manifest,
)


@dataclass(frozen=True)
class CheckpointFormalEvalConfig:
    """Provenance for a formal evaluation of an SFT, GRPO, or KD sampler."""

    sampler_path: str
    source_training_run_url: str
    experiment_id: str
    evaluation_stage: str
    parent_checkpoint: Optional[str] = None
    attempt: int = 1
    hard_cap_usd: float = 7.0
    progress_every: int = PROGRESS_EVERY

    def validate(self) -> None:
        if not self.experiment_id:
            raise BaseEvalError("experiment_id is required")
        if self.evaluation_stage not in {"sft", "grpo", "kd"}:
            raise BaseEvalError("evaluation_stage must be sft, grpo, or kd")
        if not self.sampler_path.startswith("tinker://"):
            raise BaseEvalError("sampler_path must be a Tinker URI")
        if "/sampler_weights/" not in self.sampler_path:
            raise BaseEvalError("sampler_path must reference sampler_weights")
        if not self.source_training_run_url.startswith("https://wandb.ai/"):
            raise BaseEvalError("source_training_run_url must be a W&B URL")
        if self.attempt <= 0 or self.hard_cap_usd <= 0 or self.progress_every <= 0:
            raise BaseEvalError(
                "attempt, hard cap, and progress interval must be positive"
            )

    @property
    def checkpoint_label(self) -> str:
        return self.sampler_path.rsplit("/", 1)[-1]

    def base_config(self) -> BaseEvalConfig:
        self.validate()
        return BaseEvalConfig(
            experiment_id=self.experiment_id,
            evaluation_stage=self.evaluation_stage,
            model_path=self.sampler_path,
            checkpoint=self.checkpoint_label,
            parent_checkpoint=self.parent_checkpoint,
            source_training_run_url=self.source_training_run_url,
            evaluation_split="formal",
            eval_examples=FORMAL_EXAMPLES,
            attempt=self.attempt,
            hard_cap_usd=self.hard_cap_usd,
            progress_every=self.progress_every,
        )


def build_checkpoint_doctor_report(
    config: CheckpointFormalEvalConfig,
) -> Dict[str, Any]:
    """Preflight a checkpoint without calling Tinker or W&B."""
    report = build_doctor_report(config.base_config())
    report["source_training_run_url"] = config.source_training_run_url
    return report


async def run_checkpoint_formal(
    config: CheckpointFormalEvalConfig, allow_paid: bool
) -> Dict[str, Any]:
    """Evaluate a selected sampler on every frozen formal GSM8K prompt."""
    base_config = config.base_config()
    manifest: SplitManifest = read_manifest()
    report = await run_remote_evaluation(
        base_config,
        manifest,
        load_official_test_rows(manifest, "formal"),
        allow_paid=allow_paid,
    )
    report["source_training_run_url"] = config.source_training_run_url
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or formally evaluate a selected post-training sampler."
    )
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--source-training-run-url", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--evaluation-stage", choices=("sft", "grpo", "kd"), required=True
    )
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--hard-cap-usd", type=float, default=7.0)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> CheckpointFormalEvalConfig:
    return CheckpointFormalEvalConfig(
        sampler_path=args.sampler_path,
        source_training_run_url=args.source_training_run_url,
        experiment_id=args.experiment_id,
        evaluation_stage=args.evaluation_stage,
        parent_checkpoint=args.parent_checkpoint,
        attempt=args.attempt,
        hard_cap_usd=args.hard_cap_usd,
        progress_every=args.progress_every,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env()
    args = parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.run:
            report = asyncio.run(run_checkpoint_formal(config, allow_paid=args.allow_paid))
        else:
            if args.allow_paid:
                raise BaseEvalError("--allow-paid requires --run")
            report = build_checkpoint_doctor_report(config)
    except (BaseEvalError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
