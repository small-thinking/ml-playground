import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval import BaseEvalError
from modeling.llm_post_training.gsm8k_sft_grpo_lab.checkpoint_eval import (
    CheckpointFormalEvalConfig,
    parse_args,
)


SAMPLER_PATH = "tinker://run:train:0/sampler_weights/e4-grpo-step75"
TRAINING_RUN_URL = "https://wandb.ai/example/project/runs/e4"


def test_grpo_checkpoint_uses_the_common_formal_protocol():
    config = CheckpointFormalEvalConfig(
        SAMPLER_PATH,
        TRAINING_RUN_URL,
        experiment_id="e4",
        evaluation_stage="grpo",
        parent_checkpoint="e2-sft-step250",
    ).base_config()

    assert config.evaluation_stage == "grpo"
    assert config.evaluation_split == "formal"
    assert config.eval_examples == 1287
    assert config.group_size == 4
    assert config.checkpoint == "e4-grpo-step75"
    assert config.parent_checkpoint == "e2-sft-step250"


def test_checkpoint_eval_rejects_a_training_state_for_sampling():
    with pytest.raises(BaseEvalError, match="sampler_weights"):
        CheckpointFormalEvalConfig(
            "tinker://run:train:0/weights/e4-grpo-step75",
            TRAINING_RUN_URL,
            experiment_id="e4",
            evaluation_stage="grpo",
        ).base_config()


def test_cli_requires_stage_and_provenance():
    args = parse_args(
        [
            "--sampler-path",
            SAMPLER_PATH,
            "--source-training-run-url",
            TRAINING_RUN_URL,
            "--experiment-id",
            "e4",
            "--evaluation-stage",
            "grpo",
        ]
    )

    assert args.experiment_id == "e4"
    assert args.evaluation_stage == "grpo"
    assert args.run is False


def test_kd_checkpoint_uses_the_same_formal_protocol():
    config = CheckpointFormalEvalConfig(
        SAMPLER_PATH,
        TRAINING_RUN_URL,
        experiment_id="e9",
        evaluation_stage="kd",
        parent_checkpoint="base-fresh-lora",
    ).base_config()

    assert config.evaluation_stage == "kd"
    assert config.parent_checkpoint == "base-fresh-lora"
