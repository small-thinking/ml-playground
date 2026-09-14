"""Create tiny, deterministic local tables for SFT wiring checks, not evaluation."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

VERSION = "synthetic-table-smoke-v1"


def make_grid(rng, index):
    """Include signed numbers and percentages, with a unique table identity."""
    headers = rng.choice(
        [
            ["Item", "Net", "Change"],
            ["Product", "Balance", "Rate", "Units"],
        ]
    )
    grid = [headers]
    for row in range(rng.randint(2, 5)):
        values = [
            f"Item {index:04d}-{row + 1}",
            str(-rng.randint(1, 999) if row == 0 else rng.randint(-999, 999)),
            f"{rng.randint(-999, 999) / 10:.1f}%",
        ]
        grid.append(
            values + [str(rng.randint(0, 999))] if len(headers) == 4 else values
        )
    return grid


def render_table(grid, image_path, label_path, shaded=True):
    """Render image and exact HTML from the same rectangular, header-first grid."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("expected a nonempty rectangular grid")
    font = ImageFont.load_default(size=20)
    boxes = [[font.getbbox(text) for text in row] for row in grid]
    widths = [
        max(row[c][2] - row[c][0] for row in boxes) + 24 for c in range(len(grid[0]))
    ]
    height = max(box[3] - box[1] for row in boxes for box in row) + 24
    image = Image.new("RGB", (sum(widths) + 1, height * len(grid) + 1), "white")
    draw = ImageDraw.Draw(image)
    html_rows = []
    for r, row in enumerate(grid):
        x, y = 0, r * height
        tag = "th" if r == 0 else "td"
        for c, text in enumerate(row):
            draw.rectangle(
                (x, y, x + widths[c], y + height),
                fill="#e8edf2" if shaded and r == 0 else "white",
                outline="#303030",
            )
            left, top, _, _ = boxes[r][c]
            draw.text((x + 12 - left, y + 12 - top), text, font=font, fill="black")
            x += widths[c]
        html_rows.append(
            "<tr>" + "".join(f"<{tag}>{escape(text)}</{tag}>" for text in row) + "</tr>"
        )
    image.save(image_path, format="PNG")
    Path(label_path).write_text(
        "<table>" + "".join(html_rows) + "</table>\n", encoding="utf-8"
    )


def generate(output_dir, train_count=8, dev_count=4, seed=42):
    """Write local images, HTML labels, manifests and reproducibility metadata."""
    if train_count < 1 or dev_count < 1 or train_count + dev_count > 1000:
        raise ValueError("require positive train/dev counts and at most 1000 total")
    root = Path(output_dir)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("output directory must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir()
    (root / "labels").mkdir()
    rng, index = random.Random(seed), 0
    for split, count in (("train", train_count), ("dev", dev_count)):
        records = []
        for sample in range(count):
            identity = f"{split}-{sample:04d}"
            image, label = f"images/{identity}.png", f"labels/{identity}.html"
            render_table(
                make_grid(rng, index), root / image, root / label, shaded=index % 2 == 0
            )
            records.append(
                {
                    "id": identity,
                    "split": split,
                    "image": image,
                    "label": label,
                    "image_sha256": hashlib.sha256(
                        (root / image).read_bytes()
                    ).hexdigest(),
                    "label_sha256": hashlib.sha256(
                        (root / label).read_bytes()
                    ).hexdigest(),
                    "source": "self_generated",
                    "generator_version": VERSION,
                }
            )
            index += 1
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    metadata = {
        "source": "self_generated",
        "generator_version": VERSION,
        "pillow_version": pillow_version,
        "seed": seed,
        "train_count": train_count,
        "dev_count": dev_count,
        "purpose": "synthetic smoke only; not real-world quality evidence",
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--dev-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(args.output_dir, args.train_count, args.dev_count, args.seed)
        )
    )


if __name__ == "__main__":
    main()
