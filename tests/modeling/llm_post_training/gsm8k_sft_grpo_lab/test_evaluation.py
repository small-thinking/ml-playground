import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import (
    Completion,
    EvaluationError,
    evaluate_groups,
    extract_numeric_answer,
    process_diagnostic,
    score_completion,
)


def _completion(example_id, response, answer="#### 4", tokens=12, max_tokens=32):
    return Completion(example_id, response, answer, tokens, max_tokens)


def test_extract_numeric_answer_distinguishes_boxed_format_from_fallbacks():
    assert extract_numeric_answer(r"The result is \boxed{\frac{1}{2}}") == ("1/2", True)
    assert extract_numeric_answer("Reasoning\n#### 4") == ("4", False)
    assert extract_numeric_answer("Final answer: 4.") == ("4", False)


def test_process_diagnostic_checks_only_explicit_equations():
    diagnostic = process_diagnostic(r"2 + 2 = 4, then 3 * 4 = 11. \boxed{4}")

    assert diagnostic.checked_steps == 2
    assert diagnostic.valid_steps == 1
    assert diagnostic.invalid_steps == 1


def test_group_evaluation_reports_outcomes_and_learning_signal():
    groups = {
        "easy": [_completion("easy", r"2 + 2 = 4. \boxed{4}") for _ in range(4)],
        "mixed": [
            _completion("mixed", r"\boxed{4}"),
            _completion("mixed", r"\boxed{5}"),
            _completion("mixed", r"\boxed{5}"),
            _completion("mixed", "Final answer: 5"),
        ],
        "hard": [_completion("hard", r"\boxed{5}", tokens=32) for _ in range(4)],
    }

    report = evaluate_groups(groups)

    assert report.metrics["eval/exact_match"] == pytest.approx(5 / 12)
    assert report.metrics["eval/pass_at_4"] == pytest.approx(2 / 3)
    assert report.metrics["eval/format_accuracy"] == pytest.approx(11 / 12)
    assert report.metrics["eval/truncation_rate"] == pytest.approx(1 / 3)
    assert report.metrics["eval/group_all_correct_frac"] == pytest.approx(1 / 3)
    assert report.metrics["eval/group_all_wrong_frac"] == pytest.approx(1 / 3)
    assert report.metrics["eval/group_mixed_frac"] == pytest.approx(1 / 3)
    assert report.metrics["eval/group_unique_response_frac"] == pytest.approx(
        (1 / 4 + 3 / 4 + 1 / 4) / 3
    )
    assert report.metrics["eval/process_check_coverage"] == pytest.approx(1 / 3)


def test_final_correct_with_invalid_process_is_preserved():
    scored = score_completion(_completion("example", r"2 + 2 = 5. \boxed{4}"), "group")

    assert scored.correct is True
    assert scored.process.invalid_steps == 1


def test_evaluation_requires_enough_rollouts_for_pass_at_4():
    with pytest.raises(EvaluationError, match="fewer than 4"):
        evaluate_groups({"short": [_completion("short", r"\boxed{4}")]})
