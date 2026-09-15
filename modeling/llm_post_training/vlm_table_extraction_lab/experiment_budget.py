"""Run one sequential campaign operation under a durable shared cost cap.

Commands are explicit JSON argv lists executed without a shell. All child output
stays in a private local log. Uncertain operations require manual reconciliation;
this wrapper never automatically retries a command or infers spend from failure.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from .kd_collect import CollectionBudget, atomic_json, cache_lock


def _number(value, name, *, positive=False):
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"Invalid {name}")
    return value


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _completed_report(path, reservation):
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("status") != "completed":
        raise ValueError("Operation report is not completed")
    amount = _number(report.get("estimated_compute_usd"), "reported compute cost")
    if amount > reservation:
        raise ValueError("Reported compute cost exceeds reservation")
    return amount, _hash(raw)


def run_operation(
    campaign_dir, operation_id, reservation, report, command_json, maximum=9.9
):
    """Execute once or return a hash-verified completed receipt without replay.

    report must be a fresh output path on first launch. An incomplete receipt,
    existing log/report, or pending budget blocks a new launch. The maximum is
    fixed when the campaign journal is first written. Receipts contain hashes
    and aggregate costs/timing, never command arguments or child output.
    """
    if not isinstance(operation_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", operation_id
    ):
        raise ValueError("Expected a safe operation label")
    _number(reservation, "reservation", positive=True)
    _number(maximum, "maximum", positive=True)
    campaign_dir, report, command_json = map(Path, (campaign_dir, report, command_json))
    command_raw = command_json.read_bytes()
    command = json.loads(command_raw)
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
        or not command[0]
    ):
        raise ValueError("Expected a nonempty JSON argv list")
    command_hash = _hash(command_raw)
    receipt_path = campaign_dir / f"{operation_id}.receipt.json"
    log_path = campaign_dir / f"{operation_id}.log"
    usage_path = campaign_dir / "usage.json"
    protected = [
        receipt_path,
        log_path,
        usage_path,
        campaign_dir / ".lock",
        command_json,
    ]
    if report.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Report path overlaps campaign control files")
    with cache_lock(campaign_dir):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            expected = {
                "status": "completed",
                "operation_id": operation_id,
                "command_sha256": command_hash,
                "reservation_usd": reservation,
                "maximum_usd": maximum,
            }
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("Existing operation receipt differs or is incomplete")
            amount, report_hash = _completed_report(report, reservation)
            if (
                receipt.get("report_sha256") != report_hash
                or receipt.get("estimated_compute_usd") != amount
            ):
                raise ValueError("Completed operation report differs from its receipt")
            return receipt

        budget = CollectionBudget(usage_path, maximum)
        _number(budget.state.get("estimated_compute_usd"), "campaign compute cost")
        if type(budget.state.get("calls")) is not int or budget.state["calls"] < 0:
            raise ValueError("Invalid campaign call count")
        if budget.state.get("maximum_usd", maximum) != maximum:
            raise ValueError("Campaign maximum differs from the existing journal")
        if report.exists() or log_path.exists():
            raise ValueError("Fresh operation requires unused report and log paths")
        budget.state["maximum_usd"] = maximum
        budget.reserve(operation_id, reservation)
        started = time.monotonic()
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        elapsed = time.monotonic() - started
        if result.returncode:
            raise RuntimeError(
                "Operation failed; reservation remains pending; inspect local log"
            )
        amount, report_hash = _completed_report(report, reservation)
        receipt = {
            "status": "verified_pending_settlement",
            "operation_id": operation_id,
            "command_sha256": command_hash,
            "report_sha256": report_hash,
            "reservation_usd": reservation,
            "estimated_compute_usd": amount,
            "wall_seconds": elapsed,
            "maximum_usd": maximum,
        }
        # An interruption across these two files never enables command replay:
        # pending budget or an incomplete receipt requires manual reconciliation.
        atomic_json(receipt_path, receipt)
        budget.settle(amount)
        receipt["status"] = "completed"
        atomic_json(receipt_path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--reservation", type=float, required=True)
    parser.add_argument("--maximum", type=float, default=9.9)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run_operation(**vars(args))
    except (OSError, ValueError, RuntimeError):
        parser.exit(
            1,
            "Campaign operation blocked or failed; inspect local journal/log. No automatic retry.\n",
        )
    print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    main()
