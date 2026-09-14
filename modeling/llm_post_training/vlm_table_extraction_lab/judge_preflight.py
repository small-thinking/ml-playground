"""Verify the pinned official judge CLI, parsing, and metrics without API calls."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

REVISION = "5e10ee1af6696a978b990bd441eb335aa92ea3c7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.work_dir.resolve()
    UPSTREAM = args.official_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if commit != REVISION:
        raise ValueError("Unexpected upstream revision; review before using")
    output = ROOT / "outputs/judge_setup"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(UPSTREAM / ".venv/bin/table-judge-run"),
        "--models",
        "gpt-5.4",  # A supported provider name for planning only.
        "--manifest",
        str(ROOT / "data/raw/table-judge-benchmark/manifest.jsonl"),
        "--output",
        str(output / "NOT_EXECUTED.jsonl"),
        "--dry-run",
    ]
    result = subprocess.run(
        command, cwd=UPSTREAM, check=True, capture_output=True, text=True
    )
    plan = json.loads(result.stdout)
    if plan["cases"] != 538 or plan["planned_calls"] != 1076:
        raise ValueError("Unexpected case count or an existing result file")
    (output / "official_dry_run.json").write_text(json.dumps(plan, indent=2) + "\n")
    sys.path.insert(0, str(UPSTREAM / "src"))
    from table_judge_benchmark.benchmark import parse_judgments, RESULT_KEYS
    from table_judge_benchmark.analysis import summarize_results, render_markdown

    # These are hand-authored software fixtures, not outputs from any model.
    records = []
    for policy in ["fixture_oracle", "fixture_accept_all", "fixture_reject_all"]:
        for condition in ["clean", "corrupted"]:
            judgments = {key: True for key in RESULT_KEYS}
            if policy == "fixture_reject_all" or (
                policy == "fixture_oracle" and condition == "corrupted"
            ):
                judgments["content_accuracy"] = False
            response = "".join(
                f"<{key}>{str(value).lower()}</{key}>"
                for key, value in judgments.items()
            )
            assert parse_judgments(response) == judgments
            records.append(
                {
                    "model": policy,
                    "stem": "fixture_numeric",
                    "condition": condition,
                    "status": "success",
                    "judgments": judgments,
                    "error_type": "content",
                    "error_subtype": "wrong_cell_value",
                    "expected_judgment": "content_accuracy",
                }
            )
    for invalid in [
        "",
        "<content_accuracy>maybe</content_accuracy>",
        response + "<content_accuracy>true</content_accuracy>",
    ]:
        try:
            parse_judgments(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed/conflicting output accepted")
    summary = summarize_results(records)
    expected = {
        "fixture_oracle": (0.0, 1.0),
        "fixture_accept_all": (0.0, 0.0),
        "fixture_reject_all": (1.0, 1.0),
    }
    for name, rates in expected.items():
        actual = summary["models"][name]
        assert (actual["fpr"], actual["tpr"]) == rates
        assert actual["coverage"] == 1.0
    (output / "software_fixtures_only.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output / "software_fixtures_only.md").write_text(
        "# Software fixtures only; no model inference\n\n" + render_markdown(summary)
    )
    verification = {
        "revision": commit,
        "official_manifest_hash": plan["manifest_sha256"],
        "official_prompt_hash": plan["prompt_sha256"],
        "cases_validated": 538,
        "dry_run_potential_calls": 1076,
        "actual_api_calls": 0,
        "malformed_parser_checks": "passed",
        "known_fpr_tpr_checks": "passed",
        "qwen_or_tinker_provider_adapter": "not implemented upstream; needed for those backends",
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
