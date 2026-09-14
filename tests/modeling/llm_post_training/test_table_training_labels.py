import hashlib
import json

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.metrics import parse_table
from modeling.llm_post_training.vlm_table_extraction_lab.prepare_training_labels import (
    clean_label,
    prepare_training_labels,
)

VALID = "<table><tr><th>A &amp; B</th><th>Rate</th></tr><tr><td><b>Net</b> -12</td><td>1.5%</td></tr></table>"


def fixture_manifest(root, labels):
    rows = []
    for index, label in enumerate(labels):
        image_path, label_path = root / f"{index}.png", root / f"{index}.html"
        image_path.write_bytes(b"untouched image bytes")
        label_path.write_text(label)
        rows.append(
            {
                "id": f"example-{index}",
                "split": "train",
                "image": image_path.name,
                "label": label_path.name,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                "label_characters": len(label),
                "extra": "preserved",
            }
        )
    manifest = root / "input.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, rows


def test_valid_unchanged_and_outer_text_cleanup_preserve_all_cells():
    assert clean_label(VALID) == VALID
    cleaned = clean_label("<html><body>Outside prefix" + VALID + "</body></html>")
    assert parse_table(cleaned).cells == parse_table(VALID).cells
    assert "Outside prefix" not in cleaned
    # Serialization drops the table tail, even when that tail is only whitespace.
    cleaned = clean_label("<html><body>prefix" + VALID + "  \n</body></html>")
    assert cleaned.endswith("</table>")
    assert parse_table(cleaned).cells == parse_table(VALID).cells
    assert "<b>Net</b> -12" in cleaned
    styled = (
        "<html><head><style>table {color: black}</style></head><body>prefix"
        + VALID
        + "</body></html>"
    )
    assert clean_label(styled) == cleaned


@pytest.mark.parametrize(
    "text",
    [
        "<table><tr><td>Unclosed",
        VALID + VALID,
        "<table><tr><td>" + VALID + "</td></tr></table>",
        "prefix<table><tr><td><script>bad</script></td></tr></table>",
        "prefix<table><tr><td><style>bad</style></td></tr></table>",
        "prefix<table><tr><td><img src='x'></td></tr></table>",
        "<script>bad</script>" + VALID,
        "<img src='x'>" + VALID,
    ],
)
def test_rejects_unsupported_and_other_parse_failures(text):
    with pytest.raises(ValueError):
        clean_label(text)


def test_full_manifest_retains_rows_assets_and_audits_changes(tmp_path):
    manifest, rows = fixture_manifest(
        tmp_path, [VALID, "<style>td{color:black}</style>outer" + VALID]
    )
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    output = tmp_path / "derived"
    audit = prepare_training_labels(manifest, tmp_path, output)
    derived = [
        json.loads(line)
        for line in (output / "rd_train.jsonl").read_text().splitlines()
    ]
    assert len(derived) == audit["output_rows"] == audit["input_rows"] == 2
    assert audit["changed_labels"] == audit["unchanged_labels"] == 1
    assert audit["verified_images"] == audit["verified_labels"] == 2
    assert audit["changes"][0]["removed_outer_style_nodes"] == 1
    assert derived[0] == rows[0]
    for old, new in zip(rows, derived):
        assert new["id"] == old["id"]
        assert new["image"] == old["image"]
        assert new["image_sha256"] == old["image_sha256"]
        assert new["extra"] == old["extra"]
        data = (tmp_path / new["label"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == new["label_sha256"]
        parse_table(data.decode())
    assert (
        audit["original_manifest_sha256"]
        == hashlib.sha256(original[manifest]).hexdigest()
    )
    assert (
        audit["derived_manifest_sha256"]
        == hashlib.sha256((output / "rd_train.jsonl").read_bytes()).hexdigest()
    )
    assert len(list(output.glob("*.html"))) == 1
    assert all(path.read_bytes() == data for path, data in original.items())
    with pytest.raises(ValueError, match="absent or empty"):
        prepare_training_labels(manifest, tmp_path, output)


@pytest.mark.parametrize("failure", ["hash", "dev", "malformed"])
def test_preflight_failure_leaves_no_output(tmp_path, failure):
    manifest, rows = fixture_manifest(tmp_path, ["outer" + VALID, VALID])
    if failure == "hash":
        (tmp_path / rows[1]["image"]).write_bytes(b"changed")
    elif failure == "dev":
        rows[1]["split"] = "dev"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = tmp_path / rows[1]["label"]
        path.write_text("<table><tr><td>unclosed")
        rows[1]["label_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "derived"
    with pytest.raises(ValueError):
        prepare_training_labels(manifest, tmp_path, output)
    assert not output.exists()
