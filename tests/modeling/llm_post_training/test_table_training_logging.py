import json
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.training_logging import (
    evaluation_metrics,
    TrainingLogger,
    training_config,
    training_role,
)


@pytest.fixture
def report():
    report = dict.fromkeys(
        [
            "model",
            "initialization",
            "hosted_weight_revision",
            "processor_revision",
            "cookbook_revision",
            "official_revision",
            "metrics_version",
            "tinker_version",
            "train_manifest_sha256",
            "dev_manifest_sha256",
            "train_examples",
            "dev_examples",
            "optimizer_steps",
            "lora",
            "optimizer",
            "loss_reduction",
            "learning_rate_schedule",
            "warmup_steps",
            "rates_usd_per_million",
            "implementation_sha256",
        ],
        "public",
    )
    report.update(
        {
            "config": dict.fromkeys(
                [
                    "epochs",
                    "batch_size",
                    "seed",
                    "warmup_ratio",
                    "eval_every",
                    "generate_every",
                    "generate_train",
                    "train_nll_endpoints_only",
                    "max_sequence_tokens",
                    "max_new_tokens",
                    "max_pixels",
                    "dataset_label",
                    "inference_concurrency",
                ],
                1,
            ),
            "prompt": "fixed instruction",
            "cost": {"training_tokens": 1000, "estimated_usd_bound": 4.0},
            "sampler_paths": {"after": "PRIVATE_MARKER"},
            "training_info": {"private": "PRIVATE_MARKER"},
        }
    )
    report["config"]["train_manifest"] = "/PRIVATE_MARKER/train.jsonl"
    report["lora"] = {
        "rank": 8,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }
    report["optimizer"] = {"learning_rate": 5e-5, "beta1": 0.9, "beta2": 0.95}
    return report


def test_public_training_configuration_keeps_specs_and_omits_private_context(report):
    config = training_config(report)
    assert "PRIVATE_MARKER" not in json.dumps(config)
    assert config["lora"]["rank"] == 8
    assert config["optimizer"]["learning_rate"] == 5e-5
    assert config["lora_alpha"] is None
    assert config["lora_dropout"] is None
    assert config["planned_training_tokens"] == 1000


@pytest.mark.parametrize(
    "algorithm, objective, gold_weight",
    [
        ("on_policy_reverse_kl_opd", "sampled_reverse_kl", 0.0),
        ("on_policy_topk_distillation", "topk_forward_kl", 0.0),
        ("on_policy_reverse_kl_gold", "hybrid_gold", 0.25),
    ],
)
def test_opd_logging_keeps_objective_and_role_without_private_rollouts(
    report, monkeypatch, algorithm, objective, gold_weight
):
    report.update(
        algorithm=algorithm,
        teacher_model="Qwen/Qwen3.5-397B-A17B",
        teacher_processor_revision="teacher-processor-hash",
        teacher_hosted_weight_revision="unavailable",
        tokenizer_sha256="tokenizer-hash",
        loss_temperature=1.0,
        rollout_temperature=1.0,
        opd_objective=objective,
        gold_weight=gold_weight,
        advantage_discount=0.0,
        created_at="2026-09-14",
        rollouts=[{"text": "PRIVATE_MARKER", "token_ids": [1, 2]}],
        teacher_logprobs="PRIVATE_MARKER",
        cache_sha256="PRIVATE_MARKER",
        top_k=64,
        retained_mass="PRIVATE_MARKER",
        intermediate_sampler_paths={"step_0025": "PRIVATE_MARKER"},
        diagnostics={"target_tokens": [1, 2], "text": "PRIVATE_MARKER"},
        selected_checkpoint_path="PRIVATE_MARKER",
    )
    report["config"].update(
        generate_dev=True,
        rollouts_per_example=1,
        updates_per_rollout=1,
        wandb_mode="online",
        wandb_project="fixtures",
        objective=objective,
        diagnostic_every=5,
        train_probe_examples=32,
        generate_dev_every=25,
        output_dir="/PRIVATE_MARKER/output",
        data_root="/PRIVATE_MARKER/data",
        env_file="/PRIVATE_MARKER/env",
    )
    process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO('{"ready": true, "run_id": "opd-fixture"}\n'),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    TrainingLogger(report)
    payload = json.loads(process.stdin.getvalue())
    config = payload["config"]
    assert payload["job_type"] == "opd"
    assert payload["group"] == "table-opd-training"
    assert "opd" in payload["tags"]
    assert config["algorithm"] == algorithm
    assert config["opd_objective"] == config["objective"] == objective
    assert config["gold_weight"] == gold_weight
    assert config["diagnostic_every"] == 5
    assert config["train_probe_examples"] == 32
    assert config["generate_dev_every"] == 25
    assert config["train_likelihood_scope"] == "fixed_gold_training_probe"
    assert config["advantage_discount"] == 0.0
    assert config["loss_temperature"] == config["rollout_temperature"] == 1.0
    assert config["rollouts_per_example"] == config["updates_per_rollout"] == 1
    assert config["teacher_model"] == report["teacher_model"]
    assert config["teacher_processor_revision"] == "teacher-processor-hash"
    assert config["teacher_hosted_weight_revision"] == "unavailable"
    assert config["tokenizer_sha256"] == "tokenizer-hash"
    assert config["generate_dev"] is True
    assert config["checkpoint_selection"] == "final_step"
    assert config["planned_training_tokens"] == 1000
    assert config["estimated_training_run_usd_bound"] == 4.0
    assert not {"cache_sha256", "top_k", "retained_mass"} & config.keys()
    assert "PRIVATE_MARKER" not in json.dumps(payload)
    assert (
        not {"target_tokens", "diagnostics", "selected_checkpoint_path"} & config.keys()
    )

    # The final publisher reads selection metadata from the chosen source report.
    report.update(
        checkpoint_selection="dev_cell_f1_then_nll", selected_checkpoint_step=25
    )
    selected = training_config(report)
    assert selected["checkpoint_selection"] == "dev_cell_f1_then_nll"
    assert selected["selected_checkpoint_step"] == 25
    assert selected["opd_objective"] == objective
    assert "PRIVATE_MARKER" not in json.dumps(selected)
    report["config"]["train_probe_examples"] = 0
    assert training_config(report)["train_likelihood_scope"] == "not_measured"


@pytest.mark.parametrize(
    "algorithm, expected",
    [
        (None, "sft"),
        ("off_policy_topk_kd", "kd"),
        ("on_policy_reverse_kl_opd", "opd"),
        ("on_policy_topk_distillation", "opd"),
        ("on_policy_reverse_kl_gold", "opd"),
    ],
)
def test_training_role_classification(algorithm, expected):
    assert training_role({"algorithm": algorithm}) == expected


def test_dev_only_check_does_not_report_stale_train_or_gap():
    values = evaluation_metrics(
        {"step": 25, "dev": {"assistant_nll": 0.2, "assistant_perplexity": 1.22}}
    )
    assert values == {
        "training/optimizer_step": 25,
        "dev/nll": 0.2,
        "dev/perplexity": 1.22,
    }


def test_complete_dev_quality_metrics_remain_aggregate_only():
    values = evaluation_metrics(
        {
            "step": 25,
            "dev": {
                "eval/table_exact": 0.3,
                "eval/structure_exact": 0.6,
                "eval/official_rd_similarity": 0.7,
                "eval/official_rd_similarity_raw": 0.8,
                "per_example": [{"target_tokens": [1, 2], "html": "PRIVATE_MARKER"}],
                "predictions_path": "/PRIVATE_MARKER/predictions.jsonl",
            },
        }
    )
    assert values == {
        "training/optimizer_step": 25,
        "dev/table_exact": 0.3,
        "dev/structure_exact": 0.6,
        "dev/rd_similarity_format_gated": 0.7,
        "dev/rd_similarity": 0.8,
    }
    assert "PRIVATE_MARKER" not in json.dumps(values)


def test_real_offline_stream_records_curves_without_private_context(tmp_path):
    marker = "PRIVATE_STREAM_MARKER_a31f"
    payload = {
        "mode": "offline",
        "project": "table-eval-software-fixtures",
        "job_type": "opd",
        "config": {
            "rank": 8,
            "epochs": 1,
            "objective": "hybrid_gold",
            "gold_weight": 0.25,
        },
        "metrics": {},
    }
    events = [
        {
            "training/optimizer_step": i,
            "dev/nll": 1 / (i + 1),
            "opd_diagnostics/advantage_std": 0.2,
            "optimizer/unclipped_gradient_norm": 0.3,
        }
        for i in (0, 25, 50)
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modeling.llm_post_training.vlm_table_extraction_lab.telemetry",
            "--stream",
            marker,
        ],
        input="\n".join(json.dumps(r) for r in [payload, *events]) + "\n",
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "WANDB_NOTES": marker,
            "WANDB_CONFIG_PATHS": str(tmp_path / marker),
        },
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    ready, receipt = map(json.loads, result.stdout.splitlines())
    assert ready["ready"] and ready["run_id"] == receipt["run_id"]
    root = Path(receipt["offline_dir"])
    try:
        blobs = [p.read_bytes() for p in root.rglob("*") if p.is_file()]
        assert all(marker.encode() not in b for b in blobs)
        assert any(b"training/optimizer_step" in b and b"dev/nll" in b for b in blobs)
        assert any(b"opd_diagnostics/advantage_std" in b for b in blobs)
        assert any(b"optimizer/unclipped_gradient_norm" in b for b in blobs)
        assert not list(root.rglob("wandb-metadata.json"))
    finally:
        shutil.rmtree(root)
