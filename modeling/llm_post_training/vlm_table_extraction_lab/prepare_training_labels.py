"""Derive train-only table labels by removing surrounding text, never dropping rows."""

import argparse
import hashlib
import json
from pathlib import Path

from lxml import etree

from .metrics import parse_table

VERSION = "outside-text-and-document-style-table-only-v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_label(text):
    """Return an unchanged valid label, or extract its sole safe table."""
    valid = True
    try:
        parse_table(text)
    except ValueError as error:
        if str(error) != "outside_text_or_unsupported_content":
            raise
        valid = False
    tree = etree.HTML(text, parser=etree.HTMLParser(no_network=True))
    tables = tree.xpath("//table")
    if len(tables) != 1 or tree.xpath("//script|//img") or tables[0].xpath(".//style"):
        raise ValueError(
            "Cleanup requires one safe table without scripts/images or internal styles"
        )
    if valid:
        return text
    cleaned = etree.tostring(
        tables[0], encoding="unicode", method="html", with_tail=False
    )
    parse_table(cleaned)
    return cleaned


def prepare_training_labels(manifest, data_root, output_dir):
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    output.relative_to(root)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be absent or empty")
    original_manifest = Path(manifest).read_bytes()
    rows = [json.loads(line) for line in original_manifest.splitlines() if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Expected nonempty train-only manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    # Verify every original asset before transforming or writing anything.
    labels = []
    for row in rows:
        for field in ("image", "label"):
            path = root / row[field]
            if Path(row[field]).is_absolute() or not path.resolve().is_relative_to(
                root
            ):
                raise ValueError("Expected asset paths relative to data root")
            data = path.read_bytes()
            if sha256(data) != row.get(f"{field}_sha256"):
                raise ValueError(f"Mismatched {field} hash")
            if field == "label":
                labels.append(data.decode("utf-8"))
    derived, replacements, changes = [], {}, []
    for index, (row, text) in enumerate(zip(rows, labels)):
        cleaned = clean_label(text)
        updated = dict(row)
        if cleaned != text:
            relative = (output / f"label-{index:04d}.html").relative_to(root).as_posix()
            data = cleaned.encode("utf-8")
            replacements[relative] = data
            updated.update(label=relative, label_sha256=sha256(data))
            if "label_characters" in updated:
                updated["label_characters"] = len(cleaned)
            changes.append(
                {
                    "row_index": index,
                    "original_label_sha256": row["label_sha256"],
                    "derived_label_sha256": updated["label_sha256"],
                    "removed_outer_style_nodes": len(
                        etree.HTML(
                            text, parser=etree.HTMLParser(no_network=True)
                        ).xpath("//style[not(ancestor::table)]")
                    ),
                }
            )
        derived.append(updated)
    manifest_data = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in derived
    ).encode("utf-8")
    audit = {
        "transformation_version": VERSION,
        "original_manifest_sha256": sha256(original_manifest),
        "derived_manifest_sha256": sha256(manifest_data),
        "input_rows": len(rows),
        "output_rows": len(derived),
        "changed_labels": len(changes),
        "unchanged_labels": len(rows) - len(changes),
        "verified_images": len(rows),
        "verified_labels": len(rows),
        "transformation": "Serialize the sole table without tail; discard outside text and document styles; preserve table contents and spans",
        "changes": changes,
    }
    output.mkdir(parents=True, exist_ok=True)
    for relative, data in replacements.items():
        (root / relative).write_bytes(data)
    (output / "rd_train.jsonl").write_bytes(manifest_data)
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare_training_labels(args.manifest, args.data_root, args.output_dir)
    print(json.dumps({key: value for key, value in audit.items() if key != "changes"}))


if __name__ == "__main__":
    main()
