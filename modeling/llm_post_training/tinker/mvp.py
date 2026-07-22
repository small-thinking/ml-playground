"""Minimal, cost-gated Tinker connectivity smoke test.

The default command is local-only. A remote sampling request requires both
``--remote-sample`` and ``--allow-paid`` so an accidental invocation cannot
spend Tinker credit.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Optional, Sequence


MODEL_ID = "Qwen/Qwen3.5-4B"
SDK_PACKAGE = "tinker"
MIN_PYTHON = (3, 11)
MAX_PROMPT_TOKENS = 512
MAX_OUTPUT_TOKENS = 64
HARD_CAP_USD = 0.01
PREFILL_USD_PER_MILLION = 0.33
SAMPLE_USD_PER_MILLION = 1.005
SMOKE_PROMPT = (
    "Solve the following problem. Return only the final integer answer.\n\n"
    "What is 17 * 23?"
)


class SmokeTestError(RuntimeError):
    """Raised when a smoke-test safety or readiness check fails."""


@dataclass(frozen=True)
class SmokeConfig:
    """Frozen limits for the single-request sampling smoke test."""

    model_id: str = MODEL_ID
    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    hard_cap_usd: float = HARD_CAP_USD
    prefill_usd_per_million: float = PREFILL_USD_PER_MILLION
    sample_usd_per_million: float = SAMPLE_USD_PER_MILLION


@dataclass(frozen=True)
class DoctorReport:
    """Machine-readable local readiness report."""

    mode: str
    network_called: bool
    python_version: str
    python_supported: bool
    sdk_available: bool
    sdk_version: Optional[str]
    api_key_configured: bool
    model_id: str
    max_prompt_tokens: int
    max_output_tokens: int
    estimated_max_cost_usd: float
    hard_cap_usd: float
    ready_for_remote_sample: bool


@dataclass(frozen=True)
class SampleReport:
    """Result of the one-request remote sampling smoke test."""

    mode: str
    network_called: bool
    model_id: str
    prompt_tokens: int
    output_tokens: int
    estimated_token_cost_usd: float
    hard_cap_usd: float
    response_text: str


def estimate_token_cost_usd(
    prompt_tokens: int,
    output_tokens: int,
    config: SmokeConfig,
) -> float:
    """Estimate token charges using the rates frozen in ``config``."""
    if prompt_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")

    prefill_cost = prompt_tokens * config.prefill_usd_per_million / 1_000_000
    sample_cost = output_tokens * config.sample_usd_per_million / 1_000_000
    return prefill_cost + sample_cost


def _sdk_version() -> Optional[str]:
    if importlib.util.find_spec(SDK_PACKAGE) is None:
        return None
    try:
        return version(SDK_PACKAGE)
    except PackageNotFoundError:
        return None


def build_doctor_report(
    config: SmokeConfig,
    environ: Mapping[str, str] = os.environ,
    sdk_version: Optional[str] = None,
) -> DoctorReport:
    """Build a readiness report without constructing a Tinker client."""
    detected_sdk_version = _sdk_version() if sdk_version is None else sdk_version
    python_supported = sys.version_info[:2] >= MIN_PYTHON
    api_key_configured = bool(environ.get("TINKER_API_KEY"))
    estimated_max_cost = estimate_token_cost_usd(
        config.max_prompt_tokens,
        config.max_output_tokens,
        config,
    )
    budget_valid = estimated_max_cost <= config.hard_cap_usd

    return DoctorReport(
        mode="local-doctor",
        network_called=False,
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        python_supported=python_supported,
        sdk_available=detected_sdk_version is not None,
        sdk_version=detected_sdk_version,
        api_key_configured=api_key_configured,
        model_id=config.model_id,
        max_prompt_tokens=config.max_prompt_tokens,
        max_output_tokens=config.max_output_tokens,
        estimated_max_cost_usd=estimated_max_cost,
        hard_cap_usd=config.hard_cap_usd,
        ready_for_remote_sample=(
            python_supported
            and detected_sdk_version is not None
            and api_key_configured
            and budget_valid
        ),
    )


def _require_remote_authorization(
    config: SmokeConfig,
    allow_paid: bool,
    environ: Mapping[str, str],
) -> None:
    if not allow_paid:
        raise SmokeTestError(
            "remote sampling is blocked; pass --allow-paid only after explicit approval"
        )
    if not environ.get("TINKER_API_KEY"):
        raise SmokeTestError("TINKER_API_KEY is not configured")

    estimated_max_cost = estimate_token_cost_usd(
        config.max_prompt_tokens,
        config.max_output_tokens,
        config,
    )
    if estimated_max_cost > config.hard_cap_usd:
        raise SmokeTestError(
            "estimated maximum token cost exceeds the configured hard cap"
        )


async def run_remote_sample(
    config: SmokeConfig,
    allow_paid: bool,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    service_client: Any = None,
) -> SampleReport:
    """Run exactly one bounded sampling request after all safety checks pass."""
    _require_remote_authorization(config, allow_paid, environ)

    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise SmokeTestError(
                "Tinker SDK is unavailable; run with `uv run --extra tinker`"
            ) from exc

    if service_client is None:
        service_client = tinker_module.ServiceClient(
            user_metadata={"experiment": "ml-playground-tinker-mvp"}
        )

    sampling_client = service_client.create_sampling_client(base_model=config.model_id)
    tokenizer = sampling_client.get_tokenizer()
    prompt_tokens = tokenizer.encode(SMOKE_PROMPT)
    if len(prompt_tokens) > config.max_prompt_tokens:
        raise SmokeTestError("smoke prompt exceeds the prompt-token cap")

    prompt = tinker_module.ModelInput.from_ints(tokens=prompt_tokens)
    sampling_params = tinker_module.SamplingParams(
        max_tokens=config.max_output_tokens,
        temperature=0.0,
        seed=0,
    )
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )
    sequence = result.sequences[0]
    output_tokens = list(sequence.tokens)
    response_text = tokenizer.decode(output_tokens)
    if not response_text.strip():
        raise SmokeTestError("Tinker returned an empty sample")
    estimated_cost = estimate_token_cost_usd(
        len(prompt_tokens),
        len(output_tokens),
        config,
    )

    return SampleReport(
        mode="remote-sample",
        network_called=True,
        model_id=config.model_id,
        prompt_tokens=len(prompt_tokens),
        output_tokens=len(output_tokens),
        estimated_token_cost_usd=estimated_cost,
        hard_cap_usd=config.hard_cap_usd,
        response_text=response_text,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Tinker doctor or one cost-gated sample."
    )
    parser.add_argument(
        "--remote-sample",
        action="store_true",
        help="Run one bounded Tinker sampling request.",
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Acknowledge explicit approval for the remote paid request.",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    config = SmokeConfig()
    if args.remote_sample:
        report = await run_remote_sample(config, allow_paid=args.allow_paid)
    else:
        if args.allow_paid:
            raise SmokeTestError("--allow-paid requires --remote-sample")
        report = build_doctor_report(config)

    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except SmokeTestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
