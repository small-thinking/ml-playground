"""Formal E1 evaluation for a selected GSM8K SFT sampler checkpoint."""

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


EXPERIMENT_ID = "e1"


@dataclass(frozen=True)
class SFTFormalEvalConfig:
    """The SFT provenance added to the shared formal evaluation protocol."""

    sampler_path: str
    source_training_run_url: str
    attempt: int = 1
    hard_cap_usd: float = 7.0
    progress_every: int = PROGRESS_EVERY

    def validate(self) -> None:
        if not self.sampler_path.startswith("tinker://"):
            raise BaseEvalError("sampler_path must be a Tinker URI")
        if "/sampler_weights/" not in self.sampler_path:
            raise BaseEvalError("sampler_path must reference sampler_weights")
        if not self.source_training_run_url.startswith("https://wandb.ai/"):
            raise BaseEvalError("source_training_run_url must be a W&B URL")
        if self.attempt <= 0 or self.hard_cap_usd <= 0 or self.progress_every <= 0:
            raise BaseEvalError("attempt, hard cap, and progress interval must be positive")

    @property
    def checkpoint_label(self) -> str:
        return self.sampler_path.rsplit("/", 1)[-1]

    def base_config(self) -> BaseEvalConfig:
        self.validate()
        return BaseEvalConfig(
            experiment_id=EXPERIMENT_ID,
            evaluation_stage="sft",
            model_path=self.sampler_path,
            checkpoint=self.checkpoint_label,
            parent_checkpoint="base",
            source_training_run_url=self.source_training_run_url,
            evaluation_split="formal",
            eval_examples=FORMAL_EXAMPLES,
            attempt=self.attempt,
            hard_cap_usd=self.hard_cap_usd,
            progress_every=self.progress_every,
        )


def build_sft_doctor_report(config: SFTFormalEvalConfig) -> Dict[str, Any]:
    """Preflight the selected sampler without calling Tinker or W&B."""
    report = build_doctor_report(config.base_config())
    report["source_training_run_url"] = config.source_training_run_url
    return report


async def run_e1_formal(
    config: SFTFormalEvalConfig, allow_paid: bool
) -> Dict[str, Any]:
    """Evaluate the selected SFT sampler on every frozen formal test prompt."""
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
        description="Preflight or run the formal GSM8K evaluation of one SFT sampler."
    )
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--source-training-run-url", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--hard-cap-usd", type=float, default=7.0)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> SFTFormalEvalConfig:
    return SFTFormalEvalConfig(
        sampler_path=args.sampler_path,
        source_training_run_url=args.source_training_run_url,
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
            report = asyncio.run(run_e1_formal(config, allow_paid=args.allow_paid))
        else:
            if args.allow_paid:
                raise BaseEvalError("--allow-paid requires --run")
            report = build_sft_doctor_report(config)
    except (BaseEvalError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
