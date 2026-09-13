"""Evaluate local table predictions or a Transformers VLM and log scalar metrics."""

import argparse
import hashlib
import json
from importlib.metadata import version
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .metrics import VERSION, parse_table, score_tables
from .official import OfficialScorer, REVISION


def read_jsonl(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    ids = [r.get("id") for r in rows]
    if (
        not rows
        or any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("JSONL must contain nonempty unique string IDs")
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(records, data_root, predictions, scorer):
    """All manifest rows are scored; missing predictions count as failures."""
    if set(predictions) - {r["id"] for r in records}:
        raise ValueError("Predictions contain IDs outside the evaluation manifest")
    results = []
    for row in records:
        try:
            reference = parse_table((Path(data_root) / row["label"]).read_text())
        except (ValueError, OSError, KeyError) as exc:
            raise ValueError(
                "Invalid reference: fix manifest/label before evaluating"
            ) from exc
        if (
            row.get("label_sha256")
            and digest(Path(data_root) / row["label"]) != row["label_sha256"]
        ):
            raise ValueError("Reference hash mismatch")
        pred = predictions.get(row["id"])
        parsed, parse_error = None, None
        if pred is not None:
            try:
                parsed = parse_table(pred.get("html", ""))
            except ValueError as exc:
                parse_error = str(exc)
        scores = score_tables(reference, parsed)
        scores["prediction_present"] = float(pred is not None)
        scores["empty_output"] = float(
            pred is None
            or not isinstance(pred.get("html"), str)
            or not pred["html"].strip()
        )
        scores["official_rd_similarity"] = 0.0
        scores["official_error"] = 0.0
        if parsed is not None:
            try:
                scores["official_rd_similarity"] = scorer(reference, parsed)
            except (ValueError, ArithmeticError, IndexError):
                # Report excluded upstream failures; do not turn unavailable scores into 0.
                scores["official_rd_similarity"] = None
                scores["official_error"] = 1.0
        for key in ["input_tokens", "output_tokens", "latency_seconds", "cost_usd"]:
            value = pred.get(key) if pred else None
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "Prediction telemetry must be finite nonnegative numbers"
                )
            scores[key] = value
        reason = pred.get("stop_reason") if pred else None
        scores["truncated"] = (
            float(reason in ("length", "max_tokens")) if reason else None
        )
        results.append({"id": row["id"], "metrics": scores, "parse_error": parse_error})
    metrics = {"eval/examples": len(results)}
    for key in results[0]["metrics"]:
        values = [r["metrics"][key] for r in results if r["metrics"][key] is not None]
        metrics[f"eval/{key}_count"] = len(values)
        if values:
            metrics[f"eval/{key}"] = sum(values) / len(values)
            if key in ("input_tokens", "output_tokens", "latency_seconds", "cost_usd"):
                metrics[f"eval/{key}_total"] = sum(values)
    return results, metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--official-repo", type=Path, required=True)
    p.add_argument(
        "--backend", choices=["predictions", "transformers"], default="predictions"
    )
    p.add_argument("--predictions", type=Path)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="HF ID or local model path; never logged",
    )
    p.add_argument(
        "--run-kind", choices=["evaluation", "software_fixture"], default="evaluation"
    )
    p.add_argument("--revision", help="HF commit/revision for model and processor")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument(
        "--env-file", type=Path, help="Optional local credentials file; not uploaded"
    )
    p.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    p.add_argument("--wandb-project", default="vlm-table-extraction")
    return p


def main():
    args = parser().parse_args()
    if args.max_new_tokens <= 0 or args.max_pixels <= 0:
        raise ValueError("Token and pixel limits must be positive")
    if args.backend == "predictions" and args.predictions is None:
        raise ValueError("--predictions is required for the predictions backend")
    if args.backend != "predictions" and args.predictions is not None:
        raise ValueError("--predictions is only accepted by the predictions backend")
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    records = read_jsonl(args.manifest)
    scorer = OfficialScorer(args.official_repo)
    # Validate all references before loading a large model or making predictions.
    evaluate(records, args.data_root, {}, scorer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (args.output_dir / name).exists()
        for name in ["summary.json", "per_sample.jsonl", "predictions.jsonl"]
    ):
        raise ValueError(
            "Use a fresh output directory; existing results will not be overwritten"
        )
    start = perf_counter()
    resolved_revision = None
    if args.backend == "predictions":
        predictions = {r["id"]: r for r in read_jsonl(args.predictions)}
    else:
        from .inference import TransformersPredictor

        # Preflight image integrity before loading the model.
        from PIL import Image

        for row in records:
            path = args.data_root / row["image"]
            if row.get("image_sha256") and digest(path) != row["image_sha256"]:
                raise ValueError("Image hash mismatch")
            with Image.open(path) as image:
                image.verify()
        predictor = TransformersPredictor(
            args.model, args.revision, args.device, args.max_new_tokens, args.max_pixels
        )
        resolved_revision = predictor.revision
        predictions = {}
        with (args.output_dir / "predictions.jsonl").open("x") as stream:
            for row in records:
                pred = {"id": row["id"], **predictor(args.data_root / row["image"])}
                predictions[row["id"]] = pred
                stream.write(json.dumps(pred) + "\n")
                stream.flush()
    results, metrics = evaluate(records, args.data_root, predictions, scorer)
    metrics["eval/wall_seconds"] = perf_counter() - start
    # Deliberately allowlisted; no args, model paths, IDs, HTML or raw errors in W&B.
    from .inference import PROMPT

    config = {
        "evaluator_version": VERSION,
        "run_kind": args.run_kind,
        "wandb_version": version("wandb"),
        "transformers_version": version("transformers"),
        "official_revision": REVISION,
        "backend": args.backend,
        "prompt_sha256": (
            hashlib.sha256(PROMPT.encode()).hexdigest()
            if args.backend == "transformers"
            else None
        ),
        "manifest_sha256": digest(args.manifest),
        "max_new_tokens": (
            args.max_new_tokens if args.backend == "transformers" else None
        ),
        "max_pixels": args.max_pixels if args.backend == "transformers" else None,
        "thinking": False if args.backend == "transformers" else None,
    }
    summary = {"config": config, "metrics": metrics, "wandb_status": "pending"}
    # Local-only provenance can contain user paths. Never sent to telemetry subprocess.
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "model": args.model if args.backend == "transformers" else None,
        "revision": args.revision,
        "resolved_revision": resolved_revision,
        "predictions_sha256": (
            digest(args.predictions)
            if args.predictions
            else digest(args.output_dir / "predictions.jsonl")
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (args.output_dir / "per_sample.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results)
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    payload = {
        "config": config,
        "metrics": metrics,
        "mode": args.wandb_mode,
        "project": args.wandb_project,
    }
    # Safe argv: no dataset paths or credentials passed to the telemetry process.
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
            "W&B logging failed; local results remain saved with pending status"
        )
    report = json.loads(completed.stdout)
    summary["wandb_status"] = args.wandb_mode
    summary["wandb_run_id"] = report.get("run_id")
    if report.get("offline_dir"):
        import shutil

        shutil.move(report["offline_dir"], args.output_dir / "wandb_offline")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
