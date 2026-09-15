import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.teacher_probe import (
    feedback_summary,
)


def test_identical_prefix_feedback_distinguishes_direction_and_magnitude():
    result = feedback_summary([-1, -1], [-0.5, -2], [-2, -0.5])
    assert result["old_sampled_reverse_kl"] == pytest.approx(0.25)
    assert result["new_sampled_reverse_kl"] == pytest.approx(0.25)
    assert result["informative_sign_disagreement"] == 1
    assert result["mean_abs_teacher_logprob_difference"] == 1.5


def test_uninformative_feedback_has_no_sign_denominator():
    result = feedback_summary([-1], [-1], [-1])
    assert result["informative_tokens"] == 0
    assert result["informative_sign_disagreement"] is None


@pytest.mark.parametrize("new", [[float("nan")], [0.1], [-1, -2], []])
def test_invalid_feedback_rejected(new):
    with pytest.raises(ValueError, match="aligned finite"):
        feedback_summary([-1], [-1], new)


@pytest.mark.parametrize("budget", ["nan", "inf", "0", "-1"])
def test_invalid_budget_rejected_before_loading_data(monkeypatch, budget):
    import sys
    from modeling.llm_post_training.vlm_table_extraction_lab.teacher_probe import main

    argv = ["teacher-probe", "--max-estimated-usd", budget]
    for flag in [
        "source-run",
        "manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
    ]:
        argv.extend([f"--{flag}", "does-not-exist"])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="Budget must be positive and finite"):
        main()
