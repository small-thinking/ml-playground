from modeling.llm_post_training.gsm8k_sft_grpo_lab.distillation_schema import (
    DISTILLATION_SIGNAL_KINDS,
    HARD_RESPONSE,
    ON_POLICY_TOPK,
    TEACHER_JUDGE,
    configure_wandb_metrics,
    metric_dictionary_markdown,
    metric_schema_dict,
    metric_specs,
    method_spec,
    validate_logged_metric_keys,
)


class _FakeRun:
    def __init__(self):
        self.calls = []

    def define_metric(self, name, **kwargs):
        self.calls.append((name, kwargs))


def test_every_registered_kd_method_has_the_same_core_experiment_ledger():
    required = {
        "train/optimized_input_tokens",
        "cost/cumulative_usd",
        "dev/pass_at_1",
        "dev/pass_at_4",
        "dev/optimized_input_tokens",
        "dev/generated_rollouts",
        "dev/group_unique_response_frac",
        "dev/is_initialization_policy",
        "selection/selected_checkpoint_step",
        "selection/selected_dev_pass_at_1",
        "selection/selected_dev_pass_at_4",
        "selection/selected_is_initialization",
    }

    for signal_kind in DISTILLATION_SIGNAL_KINDS:
        assert required <= {spec.key for spec in metric_specs(signal_kind)}


def test_method_extensions_preserve_their_own_learning_semantics():
    hard_keys = {spec.key for spec in metric_specs(HARD_RESPONSE)}
    judge = method_spec(TEACHER_JUDGE)
    on_policy_keys = {spec.key for spec in metric_specs(ON_POLICY_TOPK)}

    assert "train/hard_kd_nll" in hard_keys
    assert judge.on_policy is True
    assert "reward-derived" in judge.student_target
    assert "data/on_policy_prefix_tokens" in on_policy_keys


def test_schema_marks_only_dev_behavior_as_a_checkpoint_selector():
    schema = metric_schema_dict(HARD_RESPONSE)

    assert schema["selection_policy"]["primary"] == "dev/pass_at_4"
    assert "train/hard_kd_nll" in schema["selection_policy"]["prohibited"]
    assert (
        "algorithm-independent frozen formal inference"
        in schema["selection_policy"]["formal_rule"]
    )
    assert "Development unique-response fraction" in metric_dictionary_markdown(
        HARD_RESPONSE
    )


def test_wandb_configuration_uses_shared_axes_and_method_specific_summaries():
    hard_run = _FakeRun()
    judge_run = _FakeRun()

    configure_wandb_metrics(hard_run, HARD_RESPONSE)
    configure_wandb_metrics(judge_run, TEACHER_JUDGE)

    assert ("train/*", {"step_metric": "train/optimizer_step"}) in hard_run.calls
    assert ("dev/*", {"step_metric": "dev/optimized_input_tokens"}) in hard_run.calls
    assert ("train/hard_kd_nll", {"summary": "min"}) in hard_run.calls
    assert ("train/hard_kd_nll", {"summary": "min"}) not in judge_run.calls


def test_schema_validation_rejects_unknown_dashboard_keys():
    assert (
        validate_logged_metric_keys(
            HARD_RESPONSE,
            ["train/hard_kd_nll", "checkpoint/state_path"],
        )
        == ()
    )
    assert validate_logged_metric_keys(HARD_RESPONSE, ["teacher/old_name"]) == (
        "teacher/old_name",
    )
