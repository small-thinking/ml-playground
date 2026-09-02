"""Score fixed GSM8K rollouts without depending on a training backend."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Optional, Sequence, Tuple


class EvaluationError(ValueError):
    """Raised when rollout groups cannot support the requested metrics."""


@dataclass(frozen=True)
class Completion:
    example_id: str
    response: str
    ground_truth: str
    output_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class ProcessDiagnostic:
    checked_steps: int
    valid_steps: int
    invalid_steps: int


@dataclass(frozen=True)
class ScoredCompletion:
    example_id: str
    group_id: str
    response: str
    parsed_answer: Optional[str]
    correct: bool
    format_valid: bool
    output_tokens: int
    truncated: bool
    process: ProcessDiagnostic


@dataclass(frozen=True)
class EvaluationReport:
    rows: Tuple[ScoredCompletion, ...]
    metrics: Mapping[str, float]


_NUMBER = r"-?(?:\d+(?:\.\d+)?|\d+/\d+)"
_EQUATION = re.compile(
    rf"(?<![\w.])({_NUMBER})\s*([+*/-])\s*({_NUMBER})\s*=\s*({_NUMBER})(?!\w)"
)
_LABELED_ANSWER = re.compile(
    r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE
)


def normalize_number(value: str) -> Optional[str]:
    """Normalize ordinary numeric GSM8K answers for exact comparison."""
    text = value.strip().replace(",", "").replace("$", "").replace("%", "")
    fraction_match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if fraction_match:
        text = f"{fraction_match.group(1)}/{fraction_match.group(2)}"
    if not re.fullmatch(_NUMBER, text):
        return None
    try:
        return str(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def _last_boxed_value(text: str) -> Optional[str]:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    index = start + len(r"\boxed{")
    depth = 1
    for end in range(index, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[index:end]
    return None


def extract_numeric_answer(text: str) -> Tuple[Optional[str], bool]:
    """Return the final numeric answer and whether it used the boxed format."""
    boxed = _last_boxed_value(text)
    if boxed is not None:
        normalized = normalize_number(boxed)
        return normalized, normalized is not None

    if "####" in text:
        candidate = text.rsplit("####", 1)[1].splitlines()[0]
        return normalize_number(candidate), False

    matches = list(_LABELED_ANSWER.finditer(text))
    if matches:
        return normalize_number(matches[-1].group(1).strip().rstrip(".")), False
    return None, False


def process_diagnostic(text: str) -> ProcessDiagnostic:
    """Check only explicit arithmetic equations; it is not a reasoning judge."""
    valid_steps = 0
    invalid_steps = 0
    for left, operator, right, stated in _EQUATION.findall(text):
        try:
            left_value, right_value, stated_value = map(Fraction, (left, right, stated))
            expected = {
                "+": left_value + right_value,
                "-": left_value - right_value,
                "*": left_value * right_value,
                "/": left_value / right_value,
            }[operator]
        except ZeroDivisionError:
            invalid_steps += 1
            continue
        if expected == stated_value:
            valid_steps += 1
        else:
            invalid_steps += 1
    return ProcessDiagnostic(
        checked_steps=valid_steps + invalid_steps,
        valid_steps=valid_steps,
        invalid_steps=invalid_steps,
    )


def score_completion(completion: Completion, group_id: str) -> ScoredCompletion:
    """Score one response against a GSM8K final answer."""
    parsed_answer, format_valid = extract_numeric_answer(completion.response)
    expected_answer, _ = extract_numeric_answer(completion.ground_truth)
    if expected_answer is None:
        raise EvaluationError(
            f"ground truth for {completion.example_id} is not numeric"
        )
    return ScoredCompletion(
        example_id=completion.example_id,
        group_id=group_id,
        response=completion.response,
        parsed_answer=parsed_answer,
        correct=parsed_answer == expected_answer,
        format_valid=format_valid,
        output_tokens=completion.output_tokens,
        truncated=(
            completion.max_output_tokens > 0
            and completion.output_tokens >= completion.max_output_tokens
        ),
        process=process_diagnostic(completion.response),
    )


def evaluate_groups(
    groups: Mapping[str, Sequence[Completion]], pass_k: int = 4
) -> EvaluationReport:
    """Aggregate outcome, group-signal, and process metrics over rollouts."""
    if pass_k <= 0 or not groups:
        raise EvaluationError("pass_k and at least one rollout group are required")

    scored_groups = []
    for group_id, completions in groups.items():
        if len(completions) < pass_k:
            raise EvaluationError(f"group {group_id} has fewer than {pass_k} rollouts")
        scored_groups.append(
            tuple(score_completion(item, group_id) for item in completions)
        )

    rows = tuple(row for group in scored_groups for row in group)
    correct = [row.correct for row in rows]
    format_valid = [row.format_valid for row in rows]
    process_rows = [row for row in rows if row.process.checked_steps]
    final_correct = [row for row in rows if row.correct]
    group_rewards = [[int(row.correct) for row in group] for group in scored_groups]

    metrics = {
        "eval/exact_match": sum(correct) / len(rows),
        "eval/pass_at_4": sum(
            any(reward for reward in group[:pass_k]) for group in group_rewards
        )
        / len(group_rewards),
        "eval/format_accuracy": sum(format_valid) / len(rows),
        "eval/avg_output_tokens": sum(row.output_tokens for row in rows) / len(rows),
        "eval/truncation_rate": sum(row.truncated for row in rows) / len(rows),
        "eval/group_all_correct_frac": sum(all(group) for group in group_rewards)
        / len(group_rewards),
        "eval/group_all_wrong_frac": sum(not any(group) for group in group_rewards)
        / len(group_rewards),
        "eval/group_mixed_frac": sum(
            any(group) and not all(group) for group in group_rewards
        )
        / len(group_rewards),
        "eval/group_reward_std_mean": sum(
            statistics.pstdev(group) for group in group_rewards
        )
        / len(group_rewards),
        "eval/process_check_coverage": len(process_rows) / len(rows),
        "eval/process_validity_rate": (
            sum(row.process.valid_steps for row in process_rows)
            / sum(row.process.checked_steps for row in process_rows)
            if process_rows
            else 0.0
        ),
        "eval/final_correct_process_invalid": (
            sum(row.process.invalid_steps > 0 for row in final_correct)
            / len(final_correct)
            if final_correct
            else 0.0
        ),
        "eval/final_correct_process_valid": (
            sum(
                row.process.checked_steps > 0 and row.process.invalid_steps == 0
                for row in final_correct
            )
            / len(final_correct)
            if final_correct
            else 0.0
        ),
    }
    return EvaluationReport(rows=rows, metrics=metrics)
