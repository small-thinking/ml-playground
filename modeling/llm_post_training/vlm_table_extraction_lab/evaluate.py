"""Evaluate Tinker or local VLM predictions and log grouped scalar metrics."""

import argparse
import hashlib
import json
from importlib.metadata import version
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .metrics import VERSION, parse_table, score_tables
from .official import OfficialScorer, REVISION
from .reporting import grouped_metrics


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
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_identity(model_path, resolved_revision):
    """Local content fingerprint, or the resolved immutable hosted revision."""
    path = Path(model_path)
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix
            in {".safetensors", ".bin", ".json", ".model", ".jinja", ".txt"}
        )
        if not files:
            raise ValueError("Local model has no identifiable model files")
        manifest = {str(p.relative_to(path)): digest(p) for p in files}
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if not resolved_revision:
        raise ValueError(
            "Model revision could not be resolved for reproducible inference"
        )
    return resolved_revision


def generate_predictions(records, predictor, data_root, concurrency):
    """Keep at most concurrency requests in flight; persist each completed result."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    def predict(row):
        return {"id": row["id"], **predictor(data_root / row["image"])}

    if concurrency == 1:
        for row in records:
            yield predict(row)
        return
    remaining = iter(records)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = set()
    try:
        for row in remaining:
            pending.add(pool.submit(predict, row))
            if len(pending) == concurrency:
                break
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
            for _ in completed:
                row = next(remaining, None)
                if row is not None:
                    pending.add(pool.submit(predict, row))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


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
        scores["official_rd_similarity_raw"] = None
        if hasattr(scorer, "score_raw"):
            try:
                scores["official_rd_similarity_raw"] = scorer.score_raw(
                    reference.html, pred.get("html", "") if pred else ""
                )
            except (ValueError, ArithmeticError, IndexError):
                pass
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
        "--backend",
        choices=["tinker", "predictions", "transformers", "mlx"],
        default="tinker",
    )
    p.add_argument("--predictions", type=Path)
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="HF ID or local model path; never logged",
    )
    p.add_argument(
        "--model-label",
        help="Optional public display label for W&B; do not pass a private path",
    )
    p.add_argument(
        "--run-kind", choices=["evaluation", "software_fixture"], default="evaluation"
    )
    p.add_argument(
        "--revision", help="HF revision; for Tinker this pins only the processor"
    )
    p.add_argument("--tinker-cookbook-dir", type=Path)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=8192)
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
    if args.max_new_tokens <= 0 or args.max_pixels <= 0 or args.concurrency <= 0:
        raise ValueError("Token, pixel and concurrency limits must be positive")
    if args.backend == "tinker":
        from .tinker_inference import PROCESSOR_REVISION

        if args.tinker_cookbook_dir is None:
            raise ValueError("--tinker-cookbook-dir is required for Tinker inference")
        args.revision = args.revision or PROCESSOR_REVISION
    if args.backend == "predictions" and args.predictions is None:
        raise ValueError("--predictions is required for the predictions backend")
    if args.backend != "predictions" and args.predictions is not None:
        raise ValueError("--predictions is only accepted by the predictions backend")
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    if args.backend in {"tinker", "predictions"}:
        args.device = None
    if args.device == "auto" and args.backend == "mlx":
        args.device = "mps"
    if args.device == "auto":
        import torch

        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    records = read_jsonl(args.manifest)
    scorer = OfficialScorer(args.official_repo)
    # Validate all references before loading a large model or making predictions.
    evaluate(records, args.data_root, {}, scorer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(
        (args.output_dir / name).exists()
        for name in ["summary.json", "per_sample.jsonl", "predictions.jsonl"]
    ):
        raise ValueError(
            "Use a fresh output directory; existing results will not be overwritten"
        )
    start = perf_counter()
    resolved_revision = None
    predictor = None
    if args.backend == "predictions":
        predictions = {r["id"]: r for r in read_jsonl(args.predictions)}
    else:
        from .inference import TransformersPredictor, MLXPredictor

        # Preflight image integrity before loading the model.
        from PIL import Image

        for row in records:
            path = args.data_root / row["image"]
            if row.get("image_sha256") and digest(path) != row["image_sha256"]:
                raise ValueError("Image hash mismatch")
            with Image.open(path) as image:
                image.verify()
        from .inference import PROMPT

        protocol = {
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "backend": args.backend,
            "manifest_sha256": digest(args.manifest),
            "model": args.model,
            "revision": args.revision,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels,
        }
        protocol_path = args.output_dir / "inference_config.json"
        prediction_path = args.output_dir / "predictions.jsonl"
        if (
            args.resume
            and prediction_path.exists()
            and prediction_path.stat().st_size
            and not protocol_path.exists()
        ):
            raise ValueError("Cannot resume predictions without an existing protocol")
        if args.backend == "tinker":
            from .tinker_inference import TinkerPredictor, COOKBOOK_REVISION, SEED

            predictor = TinkerPredictor(
                args.model,
                args.revision,
                args.max_new_tokens,
                args.max_pixels,
                args.tinker_cookbook_dir,
            )
            protocol.update(
                cookbook_revision=COOKBOOK_REVISION,
                processor_revision=predictor.processor_revision,
                seed=SEED,
                tinker_sdk_version=predictor.sdk_version,
                transformers_version=version("transformers"),
            )
            protocol["model_identity"] = "hosted-unpinned:" + args.model
        else:
            predictor_class = (
                MLXPredictor if args.backend == "mlx" else TransformersPredictor
            )
            predictor = predictor_class(
                args.model,
                args.revision,
                args.device,
                args.max_new_tokens,
                args.max_pixels,
            )
            protocol["model_identity"] = model_identity(args.model, predictor.revision)
        resolved_revision = predictor.revision
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("Resume protocol mismatch")
        else:
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        predictions = (
            {r["id"]: r for r in read_jsonl(prediction_path)}
            if args.resume
            and prediction_path.exists()
            and prediction_path.stat().st_size
            else {}
        )
        if set(predictions) - {r["id"] for r in records}:
            raise ValueError("Resume contains unknown IDs")
        with prediction_path.open("a" if args.resume else "x") as stream:
            pending = [row for row in records if row["id"] not in predictions]
            for pred in generate_predictions(
                pending,
                predictor,
                args.data_root,
                args.concurrency if args.backend == "tinker" else 1,
            ):
                predictions[pred["id"]] = pred
                stream.write(json.dumps(pred) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            "completed": len(predictions),
                            "total": len(records),
                            "output_tokens": pred["output_tokens"],
                            "latency_seconds": round(pred["latency_seconds"], 2),
                        }
                    ),
                    flush=True,
                )
    results, metrics = evaluate(records, args.data_root, predictions, scorer)
    metrics["eval/wall_seconds"] = perf_counter() - start
    # Deliberately allowlisted; no args, model paths, IDs, HTML or raw errors in W&B.
    from .inference import PROMPT

    config = {
        "evaluator_version": VERSION,
        "metric_view": "grouped-v2",
        "mlx_vlm_version": version("mlx-vlm") if args.backend == "mlx" else None,
        "tinker_sdk_version": getattr(predictor, "sdk_version", None),
        "processor_revision": getattr(predictor, "processor_revision", None),
        "cookbook_revision": COOKBOOK_REVISION if args.backend == "tinker" else None,
        "sampling_seed": SEED if args.backend == "tinker" else None,
        "inference_concurrency": args.concurrency if args.backend == "tinker" else 1,
        "model_label": args.model_label,
        "examples": len(records),
        "numeric_examples": metrics.get("eval/numeric_f1_count", 0),
        "rd_scored_examples": metrics.get("eval/official_rd_similarity_raw_count", 0),
        "inference_device": (
            "remote"
            if args.backend == "tinker"
            else (
                "metal"
                if args.backend == "mlx"
                else args.device if args.backend != "predictions" else None
            )
        ),
        "resolved_model_revision": resolved_revision,
        "run_kind": args.run_kind,
        "wandb_version": version("wandb"),
        "transformers_version": version("transformers"),
        "official_revision": REVISION,
        "backend": args.backend,
        "prompt_sha256": (
            hashlib.sha256(PROMPT.encode()).hexdigest()
            if args.backend != "predictions"
            else None
        ),
        "manifest_sha256": digest(args.manifest),
        "max_new_tokens": (
            args.max_new_tokens if args.backend != "predictions" else None
        ),
        "max_pixels": args.max_pixels if args.backend != "predictions" else None,
        "thinking": False if args.backend != "predictions" else None,
    }
    summary = {"config": config, "metrics": metrics, "wandb_status": "pending"}
    # Local-only provenance can contain user paths. Never sent to telemetry subprocess.
    provenance = {
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "model": args.model if args.backend != "predictions" else None,
        "revision": args.revision,
        "resolved_revision": resolved_revision,
        "tinker_cookbook_dir": (
            str(args.tinker_cookbook_dir.resolve())
            if args.tinker_cookbook_dir
            else None
        ),
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
        "metrics": grouped_metrics(metrics),
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
