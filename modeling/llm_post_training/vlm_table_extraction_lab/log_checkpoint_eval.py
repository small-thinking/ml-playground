"""Publish verified checkpoint aggregates to W&B without repeating inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from .evaluate import digest, evaluate, read_jsonl
from .checkpoint_eval import likelihood_summary
from .inference import PROMPT
from .metrics import VERSION
from .official import OfficialScorer, REVISION
from .reporting import grouped_metrics
from .sft import MODEL
from .tinker_inference import COOKBOOK_REVISION, PROCESSOR_REVISION


def make_payload(
    report,
    metrics,
    stage,
    prediction_hash,
    project,
    baseline_run_id,
    training=None,
    training_run_id=None,
    trained_role="sft",
):
    """Explicit allowlist: never serialize source config or sampler addresses."""
    if trained_role not in {"sft", "kd"}:
        raise ValueError("Unknown trained model role")
    config = {
        "model_label": "Qwen3.5-4B",
        "model_role": "base" if stage == "before" else trained_role,
        "split": report["config"]["split"],
        "manifest_sha256": report["manifest_sha256"],
        "examples": report["examples"],
        "numeric_examples": metrics["eval/numeric_f1_count"],
        "rd_scored_examples": metrics["eval/official_rd_similarity_raw_count"],
        "evaluator_version": VERSION,
        "official_revision": REVISION,
        "processor_revision": report["processor_revision"],
        "cookbook_revision": COOKBOOK_REVISION,
        "sampling_seed": report["seed"],
        "max_new_tokens": report["config"]["max_new_tokens"],
        "max_pixels": report["config"]["max_pixels"],
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "temperature": 0,
        "thinking": False,
        "backend": "tinker-saved-sampler",
        "publication_method": "existing_predictions_rescored",
        "source_run_sha256": report["source_run_sha256"],
        "predictions_sha256": prediction_hash,
        "baseline_run_id": baseline_run_id,
        "metric_view": "checkpoint-v1",
    }
    if training is not None:
        config["training"] = training
        config["training_run_id"] = training_run_id
    # Same data and quality protocol share a group across model iterations.
    protocol = {
        k: config[k]
        for k in (
            "split",
            "manifest_sha256",
            "evaluator_version",
            "official_revision",
            "processor_revision",
            "cookbook_revision",
            "sampling_seed",
            "max_new_tokens",
            "max_pixels",
            "prompt_sha256",
            "temperature",
            "thinking",
        )
    }
    group_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()[:12]
    group = f"rd-{config['split']}{config['examples']}-{group_hash}"
    values = grouped_metrics(metrics)
    values.update(
        {
            "quality/rd_similarity_format_gated": metrics[
                "eval/official_rd_similarity"
            ],
            "likelihood/nll": report[stage]["assistant_nll"],
            "likelihood/perplexity": report[stage]["assistant_perplexity"],
        }
    )
    if not all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in values.values()
    ):
        raise ValueError("Expected finite aggregate metrics")
    identity = {"config": config, "metrics": values}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    return {
        "config": config,
        "metrics": values,
        "mode": "online",
        "project": project,
        "run_id": run_id,
        "name": f"qwen35-4b-{config['model_role']}-{config['split']}{config['examples']}-{run_id[:6]}",
        "group": group,
        "tags": [config["split"], config["model_role"], "fixed-comparison"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-dir", "manifest", "data-root", "official-repo"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--stage", choices=["before", "after"], required=True)
    p.add_argument("--baseline-run-id")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--wandb-project", default="vlm-table-extraction")
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()
    if (args.stage == "after") != bool(args.baseline_run_id):
        raise ValueError("Only SFT runs require --baseline-run-id")
    report = json.loads((args.evaluation_dir / "run.json").read_text())
    source_path = Path(report["config"]["source_run"])
    source = json.loads(source_path.read_text())
    if (
        digest(source_path) != report["source_run_sha256"]
        or source["model"] != MODEL
        or source["cookbook_revision"] != COOKBOOK_REVISION
        or source["prompt"] != PROMPT
        or source["official_revision"] != REVISION
        or source["metrics_version"] != VERSION
    ):
        raise ValueError("Source model or scoring protocol differs from this publisher")
    records = read_jsonl(args.manifest)
    if (
        report["status"] != "completed"
        or report["processor_revision"] != PROCESSOR_REVISION
        or digest(args.manifest) != report["manifest_sha256"]
        or len(records) != report["examples"]
        or any(r["split"] != report["config"]["split"] for r in records)
    ):
        raise ValueError("Incomplete evaluation or mismatched manifest/protocol")
    path = args.evaluation_dir / f"{args.stage}_predictions.jsonl"
    predictions = read_jsonl(path)
    if {r["id"] for r in predictions} != {r["id"] for r in records}:
        raise ValueError("Expected complete predictions for this manifest")
    _, metrics = evaluate(
        records,
        args.data_root,
        {r["id"]: r for r in predictions},
        OfficialScorer(args.official_repo),
    )
    if any(value != report[args.stage].get(key) for key, value in metrics.items()):
        raise ValueError("Re-scored metrics differ from completed report")
    likelihoods = read_jsonl(args.evaluation_dir / f"{args.stage}_likelihoods.jsonl")
    if {r["id"] for r in likelihoods} != {r["id"] for r in records}:
        raise ValueError("Expected complete likelihoods")
    likelihood = likelihood_summary(likelihoods)
    for key in ("assistant_nll", "assistant_perplexity"):
        if not math.isclose(likelihood[key], report[args.stage][key], abs_tol=1e-12):
            raise ValueError("Recomputed likelihood differs from completed report")
    training = None
    if args.stage == "after" and source.get("wandb_run_id"):
        from .training_logging import training_config

        training = training_config(source)
    payload = make_payload(
        report,
        metrics,
        args.stage,
        digest(path),
        args.wandb_project,
        args.baseline_run_id,
        training=training,
        training_run_id=source.get("wandb_run_id"),
        trained_role="kd" if source.get("algorithm") == "off_policy_topk_kd" else "sft",
    )
    # Public payload and receipt are locally inspectable before any network upload.
    payload_path = args.evaluation_dir / f"{args.stage}_wandb_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.upload:
        print(
            json.dumps(
                {
                    "status": "preflight",
                    "run_id": payload["run_id"],
                    "group": payload["group"],
                }
            )
        )
        return
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "modeling.llm_post_training.vlm_table_extraction_lab.telemetry",
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "W&B upload failed; rerun to resume the same deterministic run ID"
        )
    receipt = json.loads(completed.stdout)
    (args.evaluation_dir / f"{args.stage}_wandb_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
