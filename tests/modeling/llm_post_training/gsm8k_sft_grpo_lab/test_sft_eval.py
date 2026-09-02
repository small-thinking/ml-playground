import pytest

from modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval import BaseEvalError
from modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_eval import (
    SFTFormalEvalConfig,
    parse_args,
)


SAMPLER_PATH = "tinker://run:train:0/sampler_weights/e1-sft-step625"
TRAINING_RUN_URL = "https://wandb.ai/example/project/runs/e1"


def test_sft_formal_config_uses_the_common_formal_protocol():
    config = SFTFormalEvalConfig(SAMPLER_PATH, TRAINING_RUN_URL).base_config()

    assert config.evaluation_stage == "sft"
    assert config.evaluation_split == "formal"
    assert config.eval_examples == 1287
    assert config.group_size == 4
    assert config.checkpoint == "e1-sft-step625"
    assert config.parent_checkpoint == "base"


def test_sft_formal_config_rejects_training_state_for_sampling():
    with pytest.raises(BaseEvalError, match="sampler_weights"):
        SFTFormalEvalConfig(
            "tinker://run:train:0/weights/e1-sft-step625", TRAINING_RUN_URL
        ).base_config()


def test_cli_requires_explicit_sft_provenance():
    args = parse_args(
        [
            "--sampler-path",
            SAMPLER_PATH,
            "--source-training-run-url",
            TRAINING_RUN_URL,
            "--hard-cap-usd",
            "7",
        ]
    )

    assert args.sampler_path == SAMPLER_PATH
    assert args.source_training_run_url == TRAINING_RUN_URL
    assert args.run is False
