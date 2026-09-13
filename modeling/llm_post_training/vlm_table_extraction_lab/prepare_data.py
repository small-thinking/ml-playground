"""Download pinned public Reducto datasets and audit local image/label pairs.

An optional HF token is read from an explicitly supplied env file or environment.
No inference, training, or external logging is used. Raw corpora and generated
manifests live in the ignored data/ directory.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from huggingface_hub import snapshot_download
from dotenv import dotenv_values
from PIL import Image


ROOT = None
DATA = None
SOURCES = {
    "rd-tablebench": "7748503e2bd5f210d27aa2ef5fdf4b8aa13099bb",
    "mle-interview": "7222a4c04d8eae8ca13cbca3ae34caed4400239c",
    "table-judge-benchmark": "7bf19d636d13c93d7cdf70441ad024d5137db708",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def download_source(name):
    destination = DATA / "raw" / name
    token = os.environ.get("HF_TOKEN") or False
    print(f"Downloading {name} at {SOURCES[name]}", flush=True)
    snapshot_download(
        repo_id=f"reducto/{name}",
        repo_type="dataset",
        revision=SOURCES[name],
        local_dir=destination,
        token=token,
        max_workers=8,
    )
    for archive in destination.glob("*.zip"):
        extracted = DATA / "extracted" / name
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                target = (extracted / item.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise ValueError(f"Unsafe ZIP entry in {name}")
                # ZipFile.extract does not create symlinks. CRC is checked on read.
                bundle.extract(item, extracted)
        print(f"Extracted and CRC-checked {name}", flush=True)
    print(f"Download complete: {name}", flush=True)


def image_record(dataset, path, label=None, **extra):
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        width, height = im.size
    result = {
        "dataset": dataset,
        "id": path.stem,
        "image": str(path.relative_to(ROOT)),
        "image_sha256": sha256(path),
        "pixels_sha256": pixel_hash,
        "width": width,
        "height": height,
        "split": "unassigned",
        **extra,
    }
    if label is not None:
        text = label.read_text(encoding="utf-8")
        if "<table" not in text.lower():
            raise ValueError(f"Missing table element: {label.name}")
        result.update(
            label=str(label.relative_to(ROOT)),
            label_sha256=sha256(label),
            label_characters=len(text),
        )
    return result


def audit():
    rd = DATA / "extracted/rd-tablebench/rd-tablebench"
    mle = DATA / "extracted/mle-interview/order_images_interview"
    judge = DATA / "raw/table-judge-benchmark"
    rd_images = sorted((rd / "_images").glob("*.jpg"))
    rd_labels = sorted((rd / "groundtruth").glob("*.html"))
    rd_pdfs = sorted((rd / "pdfs").glob("*.pdf"))
    rd_ids = {p.stem for p in rd_images}
    if len(rd_images) != 1000 or rd_ids != {p.stem for p in rd_labels}:
        raise ValueError("RD images and groundtruth must form 1,000 pairs")
    if rd_ids != {p.stem for p in rd_pdfs}:
        raise ValueError("RD PDF/image IDs differ")
    records = [
        image_record("rd-tablebench", p, rd / "groundtruth" / f"{p.stem}.html")
        for p in rd_images
    ]
    mle_images = sorted(mle.glob("*.jpeg"))
    if len(mle_images) != 996:
        raise ValueError("Expected 996 MLE images")
    records.extend(image_record("mle-interview", p) for p in mle_images)
    judge_rows = [
        json.loads(line)
        for line in (judge / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    if len(judge_rows) != 538:
        raise ValueError("Expected 538 judge manifest rows")
    for row in judge_rows:
        for field, hash_key in [
            ("image_path", "image"),
            ("html_path", "html"),
            ("corrupted_html_path", "corrupted_html"),
        ]:
            path = (judge / row[field]).resolve()
            if not path.is_relative_to(judge.resolve()):
                raise ValueError("Unsafe manifest path")
            if sha256(path) != row["sha256"][hash_key]:
                raise ValueError(f"Upstream manifest hash mismatch: {row['stem']}")
        records.append(
            image_record(
                "table-judge-benchmark",
                judge / row["image_path"],
                judge / row["html_path"],
                corrupted_label=str(
                    (judge / row["corrupted_html_path"]).relative_to(ROOT)
                ),
                error_type=row["error"]["error_type"],
                error_subtype=row["error"]["error_subtype"],
            )
        )
    manifests = DATA / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for dataset in SOURCES:
        with (manifests / f"{dataset}.jsonl").open("w") as stream:
            for row in records:
                if row["dataset"] == dataset:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    duplicates = {}
    for field in ["image_sha256", "pixels_sha256", "label_sha256"]:
        groups = defaultdict(list)
        for row in records:
            if field in row:
                groups[row[field]].append(f"{row['dataset']}/{row['id']}")
        duplicates[field] = [ids for ids in groups.values() if len(ids) > 1]
    write_json(manifests / "exact_duplicates.json", duplicates)
    files = []
    for name in SOURCES:
        for path in sorted((DATA / "raw" / name).rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                files.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    write_json(manifests / "raw_files.json", files)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: {"repo_id": f"reducto/{name}", "revision": rev}
            for name, rev in SOURCES.items()
        },
        "counts": dict(Counter(row["dataset"] for row in records)),
        "rd_pdfs": len(rd_pdfs),
        "judge_manifest_hashes_verified": len(judge_rows) * 3,
        "judge_error_subtypes": dict(
            Counter(row["error"]["error_subtype"] for row in judge_rows)
        ),
        "all_images_decode": True,
        "exact_duplicate_groups": {
            key: len(value) for key, value in duplicates.items()
        },
        "cross_dataset_duplicate_groups": {
            key: sum(len({v.split("/")[0] for v in ids}) > 1 for ids in value)
            for key, value in duplicates.items()
        },
        "raw_bytes": sum(item["bytes"] for item in files),
        "split_status": "unassigned; exact duplicates checked, near-duplicates and document grouping still pending",
        "labels": {
            "rd-tablebench": "groundtruth HTML",
            "mle-interview": "none",
            "table-judge-benchmark": "clean/corrupted HTML plus error metadata",
        },
        "training_or_inference_started": False,
    }
    write_json(manifests / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    global ROOT, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract all three public datasets before the local audit",
    )
    args = parser.parse_args()
    ROOT = args.work_dir.resolve()
    DATA = ROOT / "data"
    if args.download and args.env_file:
        token = dotenv_values(args.env_file).get("HF_TOKEN")
        if token:
            os.environ.setdefault("HF_TOKEN", token)
    if args.download:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(download_source, SOURCES))
    audit()


if __name__ == "__main__":
    main()
