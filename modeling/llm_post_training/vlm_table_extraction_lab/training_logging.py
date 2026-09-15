"""Allowlisted live training metrics/config in an isolated W&B subprocess."""

import hashlib
import json
import subprocess
import sys


def training_role(report):
    return {
        "off_policy_topk_kd": "kd",
        "on_policy_reverse_kl_opd": "opd",
        "on_policy_topk_distillation": "opd",
        "on_policy_reverse_kl_gold": "opd",
    }.get(report.get("algorithm"), "sft")


def training_config(report):
    args = report["config"]
    config = {
        **{
            k: report[k]
            for k in (
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
            )
        },
        **{
            k: args[k]
            for k in (
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
            )
        },
        "prompt_sha256": hashlib.sha256(report["prompt"].encode()).hexdigest(),
        "planned_training_tokens": report["cost"]["training_tokens"],
        "estimated_training_run_usd_bound": report["cost"]["estimated_usd_bound"],
        "checkpoint_selection": report.get("checkpoint_selection", "final_step"),
        "checkpoint_ttl_days": 7,
        "lora_alpha": None,
        "lora_dropout": None,
        "implementation_sha256": report["implementation_sha256"],
    }
    if "selected_checkpoint_step" in report:
        config["selected_checkpoint_step"] = report["selected_checkpoint_step"]
    if training_role(report) in {"kd", "opd"}:
        config["generate_dev"] = args["generate_dev"]
        config.update(
            {
                k: report[k]
                for k in (
                    "algorithm",
                    "teacher_model",
                    "teacher_processor_revision",
                    "loss_temperature",
                    "rollout_temperature",
                    "tokenizer_sha256",
                    "teacher_hosted_weight_revision",
                )
            }
        )
    if training_role(report) == "opd":
        for key in (
            "objective",
            "diagnostic_every",
            "train_probe_examples",
            "generate_dev_every",
        ):
            if key in args:
                config[key] = args[key]
        if "gold_weight" in report:
            config["gold_weight"] = report["gold_weight"]
        config["train_likelihood_scope"] = (
            "fixed_gold_training_probe"
            if args.get("train_probe_examples", 0)
            else "not_measured"
        )
        config.update(
            {k: args[k] for k in ("rollouts_per_example", "updates_per_rollout")}
        )
        config.update({k: report[k] for k in ("opd_objective", "advantage_discount")})
    if training_role(report) == "kd":
        for key in (
            "candidate_examples",
            "rejected_examples",
            "target_filter_policy",
            "teacher_format_invalid_examples",
            "teacher_truncated_examples",
            "teacher_sampling_seed",
        ):
            if key in report:
                config[key] = report[key]
        config.update(
            {
                k: report[k]
                for k in (
                    "cache_sha256",
                    "top_k",
                    "retained_mass",
                )
            }
        )
    return config


def batch_metrics(entry):
    return {
        "training/optimizer_step": entry["step"],
        "training/epoch": entry["epoch"],
        "training/learning_rate": entry["learning_rate"],
        "training/batch_nll": entry["assistant_nll"],
        "training/batch_perplexity": entry["assistant_perplexity"],
        "training/input_tokens": entry["input_tokens"],
        "training/supervised_tokens": entry["supervised_tokens"],
        "training/step_seconds": entry["seconds"],
    }


def evaluation_metrics(results):
    metrics = {"training/optimizer_step": results["step"]}
    if "dev_train_nll_gap" in results:
        metrics["dev/train_nll_gap"] = results["dev_train_nll_gap"]
    for split in ("train", "dev"):
        if split not in results:
            continue
        for local, public in {
            "assistant_nll": "nll",
            "assistant_perplexity": "perplexity",
            "eval/cell_f1": "cell_f1",
            "eval/table_exact": "table_exact",
            "eval/structure_exact": "structure_exact",
            "eval/official_rd_similarity": "rd_similarity_format_gated",
            "eval/numeric_f1": "numeric_f1",
            "eval/official_rd_similarity_raw": "rd_similarity",
            "eval/parse_success": "format_pass_rate",
            "eval/truncated": "truncation_rate",
        }.items():
            if local in results[split]:
                metrics[f"{split}/{public}"] = results[split][local]
    return metrics


class TrainingLogger:
    def __init__(self, report):
        self.process = None
        if report["config"]["wandb_mode"] == "disabled":
            return
        algorithm = training_role(report)
        payload = {
            "config": training_config(report),
            "metrics": {},
            "mode": "online",
            "project": report["config"]["wandb_project"],
            "job_type": algorithm,
            "group": f"table-{algorithm}-training",
            "name": f"qwen35-4b-lora-{report['config']['dataset_label']}-{report['created_at']}",
            "tags": [algorithm, "lora", report["config"]["dataset_label"]],
        }
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "modeling.llm_post_training.vlm_table_extraction_lab.telemetry",
                "--stream",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        ready = self.process.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            self.process.wait(timeout=120)
            raise RuntimeError("W&B training initialization failed before Tinker calls")
        report["wandb_run_id"] = json.loads(ready)["run_id"]

    def log(self, event):
        if self.process:
            self.process.stdin.write(json.dumps(event, allow_nan=False) + "\n")
            self.process.stdin.flush()

    def finish(self):
        if self.process:
            self.process.stdin.close()
            if self.process.wait(timeout=120):
                raise RuntimeError(
                    "W&B training logging failed; local metrics are preserved"
                )
            return json.loads(self.process.stdout.read())
        return {}
