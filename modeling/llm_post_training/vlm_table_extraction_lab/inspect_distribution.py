"""Create reproducible visual inspection sheets from existing local datasets."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

SEED = 20260913
COUNT = 24


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    ROOT = parser.parse_args().work_dir.resolve()
    output = ROOT / "outputs/distribution"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "sampling": "uniform random rows, without replacement"}
    for dataset in ["rd-tablebench", "mle-interview"]:
        records = [
            json.loads(line)
            for line in (ROOT / f"data/manifests/{dataset}.jsonl")
            .read_text()
            .splitlines()
        ]
        samples = random.Random(SEED).sample(records, COUNT)
        summary[dataset] = {
            "population": len(records),
            "portrait_fraction": sum(r["height"] > r["width"] for r in records)
            / len(records),
            "median_width": sorted(r["width"] for r in records)[len(records) // 2],
            "median_height": sorted(r["height"] for r in records)[len(records) // 2],
            "samples": [
                dict(index=i + 1, id=r["id"], image=r["image"])
                for i, r in enumerate(samples)
            ],
        }
        for page in range(2):
            sheet = Image.new("RGB", (1600, 1680), "#dddddd")
            draw = ImageDraw.Draw(sheet)
            for slot, row in enumerate(samples[page * 12 : (page + 1) * 12]):
                x, y = slot % 4 * 400, slot // 4 * 560
                with Image.open(ROOT / row["image"]) as im:
                    thumb = ImageOps.contain(im.convert("RGB"), (390, 515))
                    sheet.paste(thumb, (x + (400 - thumb.width) // 2, y + 30))
                draw.text(
                    (x + 8, y + 8),
                    f"{page * 12 + slot + 1:02d} {row['id'][:42]}",
                    fill="black",
                )
            sheet.save(output / f"{dataset}-{page + 1}.jpg", quality=92)
    (output / "sample_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {k: v for k, v in values.items() if k != "samples"}
                for name, values in summary.items()
                if isinstance(values, dict)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
