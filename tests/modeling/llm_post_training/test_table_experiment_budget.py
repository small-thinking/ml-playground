import json
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import experiment_budget
from modeling.llm_post_training.vlm_table_extraction_lab.kd_collect import cache_lock


@pytest.fixture
def operation(tmp_path, monkeypatch):
    command = tmp_path / "command.json"
    command.write_text(json.dumps(["fixture-command", "PRIVATE_ARGUMENT"]))
    args = {
        "campaign_dir": tmp_path / "campaign",
        "operation_id": "candidate-a",
        "reservation": 2.0,
        "report": tmp_path / "run.json",
        "command_json": command,
    }
    state = {"calls": 0, "amount": 0.5, "returncode": 0, "report_status": "completed"}

    def run(argv, stdout, stderr, check):
        state["calls"] += 1
        journal = json.loads((args["campaign_dir"] / "usage.json").read_text())
        assert journal["pending"]["estimated_usd_bound"] == args["reservation"]
        assert journal["pending"]["key"] == args["operation_id"]
        assert argv == ["fixture-command", "PRIVATE_ARGUMENT"]
        assert not check
        stdout.write(b"PRIVATE_STDOUT\n")
        if state["report_status"] is not None:
            args["report"].write_text(
                json.dumps(
                    {
                        "status": state["report_status"],
                        "estimated_compute_usd": state["amount"],
                    }
                )
            )
        return SimpleNamespace(returncode=state["returncode"])

    monkeypatch.setattr(experiment_budget.subprocess, "run", run)
    return args, state


def usage(args):
    return json.loads((args["campaign_dir"] / "usage.json").read_text())


def test_success_settles_actual_refunds_reservation_and_logs_privately(operation):
    args, state = operation
    receipt = experiment_budget.run_operation(**args)
    assert state["calls"] == 1
    assert usage(args)["estimated_compute_usd"] == 0.5
    assert usage(args)["pending"] is None
    assert receipt["status"] == "completed" and receipt["wall_seconds"] >= 0
    assert receipt["reservation_usd"] == 2 and receipt["estimated_compute_usd"] == 0.5
    assert len(receipt["command_sha256"]) == len(receipt["report_sha256"]) == 64
    assert "PRIVATE" not in json.dumps(receipt)
    log = args["campaign_dir"] / "candidate-a.log"
    assert log.read_text() == "PRIVATE_STDOUT\n"
    assert log.stat().st_mode & 0o777 == 0o600
    args.update(
        operation_id="candidate-b",
        report=args["report"].with_name("run-b.json"),
        reservation=8.4,
    )
    experiment_budget.run_operation(**args)
    assert usage(args)["estimated_compute_usd"] == 1.0
    assert state["calls"] == 2  # (0.5 + 8.4) * 1.1 < 9.9


def test_completed_replay_verifies_hashes_without_relaunch(operation):
    args, state = operation
    receipt = experiment_budget.run_operation(**args)
    assert experiment_budget.run_operation(**args) == receipt
    assert state["calls"] == 1
    args["report"].write_text(
        json.dumps({"status": "completed", "estimated_compute_usd": 0.6})
    )
    with pytest.raises(ValueError, match="differs"):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 1


def test_completed_command_change_cannot_replay(operation):
    args, state = operation
    experiment_budget.run_operation(**args)
    args["command_json"].write_text('["different-command"]')
    with pytest.raises(ValueError, match="differs"):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 1


@pytest.mark.parametrize(
    "failure", ["child", "overcost", "missing", "incomplete", "nan", "negative"]
)
def test_uncertain_operation_stays_pending_without_retry(operation, failure):
    args, state = operation
    state.update(
        {
            "child": {"returncode": 1},
            "overcost": {"amount": 2.1},
            "missing": {"report_status": None},
            "incomplete": {"report_status": "running"},
            "nan": {"amount": float("nan")},
            "negative": {"amount": -0.1},
        }[failure]
    )
    with pytest.raises((OSError, ValueError, RuntimeError)):
        experiment_budget.run_operation(**args)
    assert usage(args)["pending"] is not None
    assert usage(args)["estimated_compute_usd"] == 0
    with pytest.raises(ValueError, match="Uncertain"):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 1


def test_cap_margin_is_checked_before_launch(operation):
    args, state = operation
    args["reservation"] = 9.1
    with pytest.raises(ValueError, match="before next paid request"):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 0


@pytest.mark.parametrize(
    "field,value",
    [("reservation", float("nan")), ("maximum", -1), ("operation_id", "../escape")],
)
def test_invalid_inputs_never_launch(operation, field, value):
    args, state = operation
    args[field] = value
    with pytest.raises(ValueError):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 0


def test_campaign_lock_covers_command_lifetime(operation):
    args, state = operation
    with cache_lock(args["campaign_dir"]):
        with pytest.raises(BlockingIOError):
            experiment_budget.run_operation(**args)
    assert state["calls"] == 0


def test_existing_report_is_not_treated_as_new_command_result(operation):
    args, state = operation
    args["report"].write_text('{"status":"completed","estimated_compute_usd":0}')
    with pytest.raises(ValueError, match="unused report"):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 0


def test_cli_returns_only_receipt_not_child_output(operation, monkeypatch, capsys):
    args, _ = operation
    argv = ["experiment_budget"]
    for name, value in args.items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    monkeypatch.setattr("sys.argv", argv)
    experiment_budget.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "completed"
    assert "PRIVATE" not in captured.out + captured.err


@pytest.mark.parametrize(
    "command", [[], "echo PRIVATE", ["fixture", 1], ["fixture\u0000"]]
)
def test_invalid_command_argv_cannot_launch(operation, command):
    args, state = operation
    args["command_json"].write_text(json.dumps(command))
    with pytest.raises(ValueError, match="JSON argv"):
        experiment_budget.run_operation(**args)
    assert state["calls"] == 0
