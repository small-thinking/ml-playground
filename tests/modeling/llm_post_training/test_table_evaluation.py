"""Synthetic fixtures only; no private data, credentials or downloaded images."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("lxml")
from modeling.llm_post_training.vlm_table_extraction_lab.evaluate import (
    evaluate,
    read_jsonl,
)
from modeling.llm_post_training.vlm_table_extraction_lab.metrics import (
    parse_table,
    score_tables,
)
from modeling.llm_post_training.vlm_table_extraction_lab.official import OfficialScorer


def table(*rows):
    return (
        "<table>"
        + "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows
        )
        + "</table>"
    )


def score(a, b):
    return score_tables(parse_table(a), parse_table(b))


@pytest.mark.parametrize(
    "left,right", [("-100", "100"), ("(100)", "100"), ("1.2", "12"), ("10%", "10")]
)
def test_numbers_preserve_sign_punctuation_and_units(left, right):
    result = score(table([left]), table([right]))
    assert result["numeric_f1"] == 0
    assert result["cell_f1"] == 0


def test_missing_or_extra_boundary_row_is_penalized():
    for a, b in [
        (table(["a"], ["b"]), table(["a"])),
        (table(["a"]), table(["a"], ["b"])),
    ]:
        result = score(a, b)
        assert result["row_count_exact"] == 0
        assert result["cell_f1"] == pytest.approx(2 / 3)
        assert result["table_exact"] == 0


def test_merged_cells_are_not_equivalent_to_repeated_cells():
    result = score('<table><tr><td colspan="2">x</td></tr></table>', table(["x", "x"]))
    assert result["cell_f1"] == 1
    assert result["span_f1"] == 0
    assert result["structure_exact"] == 0
    assert result["table_exact"] == 0


def test_rowspan_and_unicode_whitespace():
    html = '<table><tr><td rowspan="2">−10</td><td>A  B</td></tr><tr><td>C</td></tr></table>'
    parsed = parse_table(html)
    assert parsed.cells[(1, 0)] == "-10"
    assert parsed.cells[(0, 1)] == "A B"
    assert score(html, html)["table_exact"] == 1


def test_rowspan_beyond_last_row_is_clipped_but_declaration_preserved():
    parsed = parse_table('<table><tr><td rowspan="2">x</td></tr></table>')
    assert parsed.rows == 1 and len(parsed.cells) == 1
    assert (0, 0, 2, 1) in parsed.spans


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<table><tr><td>x",
        "<table></table>",
        table(["x"]) + table(["y"]),
        "Here it is:" + table(["x"]),
        '<table><tr><td colspan="999999">x</td></tr></table>',
    ],
)
def test_invalid_output_is_rejected(html):
    with pytest.raises(ValueError):
        parse_table(html)


def test_no_numbers_has_explicit_undefined_metric():
    assert score(table(["word"]), table(["word"]))["numeric_f1"] is None


def test_swapped_cells_cannot_hide_behind_bag_score():
    result = score(table(["a", "b"]), table(["b", "a"]))
    assert result["cell_bag_f1"] == 1
    assert result["cell_f1"] == 0


def test_missing_predictions_stay_in_denominator_and_unknown_ids_fail(tmp_path):
    (tmp_path / "label.html").write_text(table(["1"]))
    rows = [{"id": "a", "label": "label.html"}, {"id": "b", "label": "label.html"}]
    pred = {"a": {"html": table(["1"]), "stop_reason": "length"}}
    results, metrics = evaluate(rows, tmp_path, pred, lambda a, b: 1.0)
    assert len(results) == 2
    assert metrics["eval/cell_f1"] == 0.5
    assert metrics["eval/prediction_present"] == 0.5
    assert metrics["eval/truncated_count"] == 1
    assert metrics["eval/cost_usd_count"] == 0
    assert "eval/cost_usd" not in metrics
    with pytest.raises(ValueError):
        evaluate(rows, tmp_path, {"extra": {}}, lambda a, b: 1.0)


def test_reference_hash_and_duplicate_ids_rejected(tmp_path):
    (tmp_path / "label.html").write_text(table(["x"]))
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate(
            [{"id": "x", "label": "label.html", "label_sha256": "bad"}],
            tmp_path,
            {},
            lambda a, b: 1,
        )
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id":"same"}\n{"id":"same"}\n')
    with pytest.raises(ValueError, match="unique"):
        read_jsonl(path)


def test_official_source_tampering_rejected_before_execution(tmp_path):
    (tmp_path / "grading.py").write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match="hash mismatch"):
        OfficialScorer(tmp_path)


def test_official_failure_is_visible_not_a_false_zero(tmp_path):
    (tmp_path / "label.html").write_text(table(["x"]))

    def failed(*_):
        raise ValueError("bounded work")

    _, m = evaluate(
        [{"id": "a", "label": "label.html"}],
        tmp_path,
        {"a": {"html": table(["x"])}},
        failed,
    )
    assert m["eval/official_error"] == 1
    assert m["eval/official_rd_similarity_count"] == 0
    assert "eval/official_rd_similarity" not in m


def test_pinned_official_regressions():
    source = os.environ.get("RD_OFFICIAL_REPO")
    if not source:
        pytest.skip("Optional local integration: supply RD_OFFICIAL_REPO")
    scorer = OfficialScorer(source)
    assert scorer(parse_table(table(["-100"])), parse_table(table(["100"]))) == 1
    assert scorer(parse_table(table(["a"], ["b"])), parse_table(table(["a"]))) == 1
    assert scorer(
        parse_table(table(["a"])), parse_table(table(["z"]))
    ) == pytest.approx(5 / 6)


def test_wandb_offline_records_do_not_capture_private_context(tmp_path):
    marker = "PRIVATE_PATH_MARKER_71ab"
    payload = {
        "mode": "offline",
        "project": "table-eval-software-fixtures",
        "config": {"evaluator_version": "fixture"},
        "metrics": {"eval/examples": 2, "eval/cell_f1": 0.5},
    }
    env = dict(
        os.environ, WANDB_CONFIG_PATHS=str(tmp_path / marker), WANDB_NOTES=marker
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "modeling.llm_post_training.vlm_table_extraction_lab.telemetry",
            marker,
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    directory = Path(report["offline_dir"])
    try:
        blobs = [p.read_bytes() for p in directory.rglob("*") if p.is_file()]
        assert blobs
        assert all(marker.encode() not in b for b in blobs)
        assert any(b"eval/cell_f1" in b for b in blobs)
        assert not list(directory.rglob("wandb-metadata.json"))
    finally:
        import shutil

        shutil.rmtree(directory)


def test_inference_only_sends_image_and_prompt_and_trims_input(tmp_path):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from modeling.llm_post_training.vlm_table_extraction_lab.inference import (
        TransformersPredictor,
        PROMPT,
    )

    image_path = tmp_path / "synthetic.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][1]["text"] == PROMPT
            assert kwargs["enable_thinking"] is False
            assert len(messages[0]["content"]) == 2
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [3, 9]
            return table(["x"])

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[9])

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 9]])

    predictor = TransformersPredictor.__new__(TransformersPredictor)
    predictor.torch = torch
    predictor.processor = Processor()
    predictor.model = Model()
    predictor.max_pixels = 100
    predictor.max_new_tokens = 2
    result = predictor(image_path)
    assert result["input_tokens"] == 2 and result["output_tokens"] == 2
    assert result["stop_reason"] == "stop"  # EOS exactly at cap is not truncation.


def test_extra_empty_row_changes_exact_match():
    result = score("<table><tr><td>x</td></tr><tr></tr></table>", table(["x"]))
    assert result["table_exact"] == 0 and result["structure_exact"] == 0


@pytest.mark.parametrize("html", [None, 12, {}])
def test_non_string_prediction_counts_as_invalid(tmp_path, html):
    (tmp_path / "label.html").write_text(table(["1"]))
    _, metrics = evaluate(
        [{"id": "a", "label": "label.html"}],
        tmp_path,
        {"a": {"html": html}},
        lambda a, b: 1,
    )
    assert metrics["eval/parse_success"] == 0 and metrics["eval/cell_f1"] == 0


def test_html_body_wrappers_and_fences_are_accepted():
    html = table(["x"])
    for wrapped in [f"<html><body>{html}</body></html>", f"```html\n{html}\n```"]:
        assert score(wrapped, html)["table_exact"] == 1
