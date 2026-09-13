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
