import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from modeling.llm_post_training.vlm_table_extraction_lab.training_logging import (
    evaluation_metrics,
    training_config,
)


def test_public_training_configuration_keeps_specs_and_omits_private_context():
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
    config = training_config(report)
    assert "PRIVATE_MARKER" not in json.dumps(config)
    assert config["lora"]["rank"] == 8
    assert config["optimizer"]["learning_rate"] == 5e-5
    assert config["lora_alpha"] is None
    assert config["lora_dropout"] is None
    assert config["planned_training_tokens"] == 1000


def test_dev_only_check_does_not_report_stale_train_or_gap():
    values = evaluation_metrics(
        {"step": 25, "dev": {"assistant_nll": 0.2, "assistant_perplexity": 1.22}}
    )
    assert values == {
        "training/optimizer_step": 25,
        "dev/nll": 0.2,
        "dev/perplexity": 1.22,
    }


def test_real_offline_stream_records_curves_without_private_context(tmp_path):
    marker = "PRIVATE_STREAM_MARKER_a31f"
    payload = {
        "mode": "offline",
        "project": "table-eval-software-fixtures",
        "job_type": "sft",
        "config": {"rank": 8, "epochs": 1},
        "metrics": {},
    }
    events = [
        {"training/optimizer_step": i, "dev/nll": 1 / (i + 1)} for i in (0, 25, 50)
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
        assert not list(root.rglob("wandb-metadata.json"))
    finally:
        shutil.rmtree(root)
