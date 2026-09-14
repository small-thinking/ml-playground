import hashlib
import json

from lxml import etree
from PIL import Image, ImageDraw
import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.metrics import parse_table
from modeling.llm_post_training.vlm_table_extraction_lab.synthetic_data import (
    generate,
    render_table,
)


def test_generation_is_deterministic_and_splits_are_disjoint(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    generate(first)
    generate(second)
    files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    assert files == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    hashes, grids = {}, {}
    for split, count in (("train", 8), ("dev", 4)):
        records = [
            json.loads(line)
            for line in (first / f"{split}.jsonl").read_text().splitlines()
        ]
        assert len(records) == count
        hashes[split], grids[split] = set(), set()
        for record in records:
            assert record["split"] == split
            assert record["source"] == "self_generated"
            for key in ("image", "label"):
                path = first / record[key]
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == record[f"{key}_sha256"]
                )
            hashes[split].add(record["image_sha256"])
            table = parse_table((first / record["label"]).read_text())
            assert len(table.cells) == table.rows * table.cols
            assert any(value.startswith("-") for value in table.cells.values())
            assert any(value.endswith("%") for value in table.cells.values())
            grids[split].add(tuple(sorted(table.cells.items())))
            with Image.open(first / record["image"]) as image:
                image.verify()
        assert len(hashes[split]) == len(grids[split]) == count
    assert hashes["train"].isdisjoint(hashes["dev"])
    assert grids["train"].isdisjoint(grids["dev"])
    assert json.loads((first / "metadata.json").read_text())["seed"] == 42
    generate(tmp_path / "different", seed=43)
    assert (first / "train.jsonl").read_bytes() != (
        tmp_path / "different/train.jsonl"
    ).read_bytes()


def test_rendered_full_grid_matches_html_and_text_fits_cells(tmp_path, monkeypatch):
    grid = [
        ["Name & type", "Change", "Amount"],
        ["<long sample>", "-15.5%", "-123456"],
        ["Other", "0.0%", "123"],
    ]
    rectangles, rendered = [], []
    original_rectangle, original_text = (
        ImageDraw.ImageDraw.rectangle,
        ImageDraw.ImageDraw.text,
    )

    def rectangle(draw, xy, *args, **kwargs):
        rectangles.append(xy)
        return original_rectangle(draw, xy, *args, **kwargs)

    def text(draw, xy, value, *args, **kwargs):
        rendered.append(value)
        x0, y0, x1, y1 = draw.textbbox(xy, value, font=kwargs["font"])
        left, top, right, bottom = rectangles[-1]
        assert left < x0 < x1 < right
        assert top < y0 < y1 < bottom
        return original_text(draw, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "rectangle", rectangle)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    render_table(grid, tmp_path / "table.png", tmp_path / "table.html")
    html = (tmp_path / "table.html").read_text()
    assert rendered == [cell for row in grid for cell in row]
    assert parse_table(html).cells == {
        (r, c): value for r, row in enumerate(grid) for c, value in enumerate(row)
    }
    table = etree.HTML(html).xpath("//table")[0]
    assert len(table.xpath("./tr[1]/th")) == 3
    assert len(table.xpath("./tr[position()>1]/td")) == 6
    assert len(rectangles) == 9


def test_refuses_overwrite_and_invalid_counts(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="absent or empty"):
        generate(output)
    assert marker.read_text() == "keep"
    assert list(output.iterdir()) == [marker]
    for train, dev in ((0, 4), (8, -1), (999, 2)):
        with pytest.raises(ValueError, match="counts"):
            generate(tmp_path / "invalid", train_count=train, dev_count=dev)
    assert not (tmp_path / "invalid").exists()
