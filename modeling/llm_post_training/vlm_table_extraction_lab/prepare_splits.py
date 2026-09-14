"""Freeze RD train/dev/test and nested train subsets without model calls.

Group exact images, original-image IDs, equal cell text, and conservative
perceptual candidates. These heuristics do not prove source-document independence.
"""

import argparse
from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

ROOT = None
SEED = 20260913


class CellText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.current = []

    def handle_data(self, value):
        if self.current is not None:
            self.current.append(value)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.current is not None:
            self.cells.append(" ".join("".join(self.current).split()))
            self.current = None


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--revise-from-v1",
        action="store_true",
        help="Archive the old 800/200 split and replace it with 800/100/100",
    )
    args = parser.parse_args()
    ROOT = args.work_dir.resolve()
    source = ROOT / "data/manifests/rd-tablebench.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    parents = list(range(len(records)))
    links = []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(a, b, reason):
        parents[find(b)] = find(a)
        links.append({"a": records[a]["id"], "b": records[b]["id"], "reason": reason})

    seen = {}
    fingerprints = []
    for i, row in enumerate(records):
        parser = CellText()
        parser.feed((ROOT / row["label"]).read_text())
        text_key = hashlib.sha256(
            json.dumps(parser.cells, ensure_ascii=False).encode()
        ).hexdigest()
        keys = [
            ("pixels", row["pixels_sha256"]),
            ("image_id", row["id"].split(".rf.")[0]),
        ]
        if any(parser.cells):
            keys.append(("cell_text", text_key))
        for key in keys:
            if key in seen:
                join(seen[key], i, key[0])
            else:
                seen[key] = i
        with Image.open(ROOT / row["image"]) as im:
            small = np.array(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        bits = np.packbits((small[:, 1:] > small[:, :-1]).flatten())
        fingerprints.append(int.from_bytes(bits.tobytes(), "big"))
    for i, row in enumerate(records):
        ratio = row["width"] / row["height"]
        for j in range(i):
            other_ratio = records[j]["width"] / records[j]["height"]
            if (
                abs(ratio / other_ratio - 1) <= 0.05
                and bin(fingerprints[i] ^ fingerprints[j]).count("1") <= 4
            ):
                join(i, j, "dhash_le4_aspect_within5pct_candidate")
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row["id"])
    ordered = sorted(
        (sorted(group) for group in groups.values()), key=lambda group: group[0]
    )
    random.Random(SEED).shuffle(ordered)
    heldout, heldout_groups, train_groups = [], [], []
    for group in ordered:
        if len(heldout) + len(group) <= 200:
            heldout.extend(group)
            heldout_groups.append(group)
        else:
            train_groups.append(group)
    # Preserve the existing train pool and its nested subsets. Divide the old
    # development pool into dev/test with whole groups on one side only.
    random.Random(SEED + 1).shuffle(heldout_groups)
    dev, test = [], []
    for group in heldout_groups:
        if len(dev) + len(group) <= 100:
            dev.extend(group)
        else:
            test.extend(group)
    train = [item for group in train_groups for item in group]
    subsets = {}
    for name, size in [
        ("train_smoke", 8),
        ("train_10pct", len(train) // 10),
        ("train_25pct", len(train) // 4),
        ("train_50pct", len(train) // 2),
    ]:
        selected = []
        for group in train_groups:
            if len(selected) >= size:
                break
            selected.extend(group)
        subsets[name] = selected
    split = {
        "version": "rd-train-dev-test-v2",
        "seed": SEED,
        "heldout_partition_seed": SEED + 1,
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_revision": "7748503e2bd5f210d27aa2ef5fdf4b8aa13099bb",
        "purpose": "personal train/dev/test partition; no model training or inference has started",
        "grouping": "exact pixels + original image ID + exact cell text + dHash <=4 with aspect within5%; no source document IDs available",
        "test_policy": "RD test100 is reserved for final frozen-model comparison, not prompt/reward/hyperparameter selection. MLE is unlabeled OOD inspection; Judge is auxiliary evaluation.",
        "train": train,
        "dev": dev,
        "test": test,
        **subsets,
    }
    assert (len(train), len(dev), len(test)) == (800, 100, 100)
    assert len(set(train + dev + test)) == len(records)
    previous = set()
    for ids in [*subsets.values(), train]:
        assert previous <= set(ids) <= set(train)
        previous = set(ids)
    for group in groups.values():
        assert any(set(group) <= set(partition) for partition in [train, dev, test])
    target = ROOT / "manifests/rd_splits.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(split, indent=2) + "\n"
    if target.exists() and target.read_text() != serialized:
        previous_text = target.read_text()
        previous_split = json.loads(previous_text)
        if (
            not args.revise_from_v1
            or previous_split.get("version") != "rd-train-dev-v1"
        ):
            raise ValueError(
                "Frozen split differs; an explicit v1 revision is required"
            )
        for name in ["train", *subsets]:
            assert (
                previous_split[name] == split[name]
            ), f"Changed training subset: {name}"
        assert set(previous_split["dev"]) == set(dev + test)
        archive = target.parent / "archive/rd_splits_v1.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and archive.read_text() != previous_text:
            raise ValueError("Existing v1 archive differs")
        archive.write_text(previous_text)
    target.write_text(serialized)
    by_id = {row["id"]: row for row in records}
    for name in ["train", "dev", "test", *subsets]:
        path = ROOT / f"data/manifests/rd_{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        **by_id[item],
                        "split": name if name in ("dev", "test") else "train",
                        "subset": name,
                    }
                )
                + "\n"
                for item in split[name]
            )
        )
    report = {
        "split_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "counts": {
            name: len(split[name]) for name in ["train", "dev", "test", *subsets]
        },
        "groups": len(groups),
        "multi_example_groups": sum(len(group) > 1 for group in groups.values()),
        "links": links,
        "train_dev_overlap": 0,
        "train_test_overlap": 0,
        "dev_test_overlap": 0,
        "nested_subsets_verified": True,
        "limitation": "dHash candidates are conservatively grouped, not manually confirmed duplicates; no full source-document or cross-dataset crop audit",
    }
    path = ROOT / "data/manifests/split_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "links"}, indent=2))


if __name__ == "__main__":
    main()
