"""Scalar-only W&B subprocess. Receives no raw data or user filesystem paths."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def log_metrics(payload):
    if payload["mode"] == "disabled":
        return {}
    # Isolate auto-detected environment/config files, argv, working directory,
    # host, Git, machine metadata, console, code and artifacts from the run.
    allowed = {"WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_ENTITY"}
    for name in list(os.environ):
        if name.startswith("WANDB_") and name not in allowed:
            del os.environ[name]
    directory = tempfile.mkdtemp(prefix="table-eval-telemetry-", dir="/tmp")
    os.chdir(directory)
    os.environ.update(WANDB_ERROR_REPORTING="false", WANDB_CONFIG_DIR=directory)
    sys.argv = ["table-evaluation"]
    import wandb

    settings = wandb.Settings(
        console="off",
        disable_code=True,
        save_code=False,
        disable_git=True,
        disable_job_creation=True,
        x_disable_meta=True,
        x_disable_stats=True,
        x_disable_machine_info=True,
        host="redacted",
        git_remote_url="",
        git_commit="",
        program="table-evaluation",
        program_abspath="/redacted/table-evaluation",
        program_relpath="table-evaluation",
        root_dir=directory,
        ignore_globs=["*.log", "wandb-metadata.json", "requirements.txt"],
    )
    with wandb.init(
        project=payload["project"],
        mode=payload["mode"],
        dir=directory,
        config=payload["config"],
        settings=settings,
        job_type="evaluation",
    ) as run:
        run.log(payload["metrics"])
        run.summary.update(payload["metrics"])
        run_id = run.id
    report = {"run_id": run_id}
    if payload["mode"] == "offline":
        report["offline_dir"] = directory
    else:
        shutil.rmtree(directory)
    return report


if __name__ == "__main__":
    print(json.dumps(log_metrics(json.loads(sys.stdin.read()))))
