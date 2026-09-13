"""Load unmodified, hash-pinned Reducto scoring code supplied by the caller."""

import hashlib
import importlib.util
from pathlib import Path

REVISION = "1cae108e6395ddc8389af17385f9769519070558"
HASHES = {
    "grading": "61506bb6bd37ee881154fdd749850d6d35fda96b4daf12a6ffe21269ca1a3cbf",
    "convert": "12ff0f629f9eaa7ee9fc8e3cfa123d247de305fa38c95ae39343ddd2a485c133",
}


class OfficialScorer:
    def __init__(self, source_dir):
        modules = {}
        # Verify BOTH files before executing either one. No runtime downloads.
        for name, expected in HASHES.items():
            path = Path(source_dir) / f"{name}.py"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "Official source hash mismatch; review the new revision"
                )
        for name in HASHES:
            spec = importlib.util.spec_from_file_location(
                f"rd_official_{name}", Path(source_dir) / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
        self.convert = modules["convert"].html_to_numpy
        self.compare = modules["grading"].table_similarity

    def __call__(self, reference, prediction):
        # Bound the upstream quadratic row/column alignment on malformed/huge input.
        work = reference.rows * reference.cols * prediction.rows * prediction.cols
        if work > 2_000_000:
            raise ValueError("official_alignment_work_limit")
        return float(
            self.compare(self.convert(reference.html), self.convert(prediction.html))
        )

    def score_raw(self, reference_html, prediction_html):
        """Direct upstream score, independent of our strict output contract."""
        from lxml import etree

        if not isinstance(prediction_html, str):
            return 0.0
        for text in (reference_html, prediction_html):
            if not isinstance(text, str) or len(text) > 1_000_000:
                raise ValueError("invalid_output_size")
            tree = etree.HTML(text, parser=etree.HTMLParser(no_network=True))
            if tree is None:
                return 0.0
            cells = tree.xpath("//td|//th")
            if len(tree.xpath("//tr")) > 500 or len(cells) > 10000:
                raise ValueError("table_too_large")
            for cell in cells:
                if not (
                    1 <= int(cell.get("rowspan", "1")) <= 500
                    and 1 <= int(cell.get("colspan", "1")) <= 256
                ):
                    raise ValueError("invalid_span")
            # Conservatively bound the padded array BEFORE upstream allocation.
            # Carry-over spans can add columns to any subsequent row.
            rows = tree.xpath("//tr")
            carried_columns = sum(
                int(cell.get("colspan", "1"))
                for cell in cells
                if int(cell.get("rowspan", "1")) > 1
            )
            width_bound = carried_columns + max(
                (
                    sum(int(c.get("colspan", "1")) for c in row.xpath("td|th"))
                    for row in rows
                ),
                default=0,
            )
            slots_bound = len(rows) * width_bound
            max_chars = max((len("".join(c.itertext())) for c in cells), default=0)
            if slots_bound > 10000 or slots_bound * max_chars * 4 > 64 * 1024**2:
                raise ValueError("official_expansion_limit")
        reference, prediction = self.convert(reference_html), self.convert(
            prediction_html
        )
        if not reference.size or not prediction.size:
            return 0.0
        if reference.size * prediction.size > 2_000_000:
            raise ValueError("official_alignment_work_limit")
        return float(self.compare(reference, prediction))
