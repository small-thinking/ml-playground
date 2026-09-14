"""Deterministic table diagnostics; no model calls or external logging."""

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

from lxml import etree

VERSION = "table-eval-v1"
MAX_SLOTS = 10000


@dataclass
class Table:
    cells: dict
    spans: set
    rows: int
    cols: int
    html: str


def normalize(text):
    # Preserve case, punctuation, decimal separators, signs and units.
    return " ".join(unicodedata.normalize("NFKC", text).replace("−", "-").split())


def parse_table(text):
    """Accept one closed table (optionally fenced); reject ambiguous geometry.

    HTML recovery is permitted inside the table, so this is a parse/contract
    check, not full HTML standards validation. Never fetch external resources.
    """
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("invalid_output_size")
    text = text.strip()
    fence = re.fullmatch(r"```(?:html)?\s*\n?(.*?)\n?```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    if not re.search(r"</table\s*>\s*(?:</(?:body|html)\s*>\s*)*$", text, re.I):
        raise ValueError("unclosed_table")
    tree = etree.HTML(text, parser=etree.HTMLParser(no_network=True))
    tables = tree.xpath("//table") if tree is not None else []
    if len(tables) != 1:
        raise ValueError("expected_one_table")
    node = tables[0]
    # Full HTML/body wrappers are accepted; surrounding prose is not.
    outside = [s for s in tree.xpath("//text()[not(ancestor::table)]") if s.strip()]
    if outside or node.xpath(".//script|.//style|.//img"):
        raise ValueError("outside_text_or_unsupported_content")
    trs = node.xpath(".//tr")
    if not trs or len(trs) > 500:
        raise ValueError("invalid_row_count")
    cells, spans = {}, set()
    for r, tr in enumerate(trs):
        col = 0
        for cell in tr.xpath("./td|./th"):
            while (r, col) in cells:
                col += 1
            try:
                rs, cs = int(cell.get("rowspan", "1")), int(cell.get("colspan", "1"))
            except ValueError as exc:
                raise ValueError("invalid_span") from exc
            if not (1 <= rs <= 500 and 1 <= cs <= 256 and col + cs <= 256):
                raise ValueError("invalid_span")
            if len(cells) + rs * cs > MAX_SLOTS:
                raise ValueError("table_too_large")
            value = normalize("".join(cell.itertext()))
            spans.add((r, col, rs, cs))
            for dr in range(min(rs, len(trs) - r)):
                for dc in range(cs):
                    key = (r + dr, col + dc)
                    if key in cells:
                        raise ValueError("overlapping_span")
                    cells[key] = value
            col += cs
    if not cells:
        raise ValueError("empty_table")
    return Table(cells, spans, len(trs), max(c for _, c in cells) + 1, text)


def prf(reference, prediction):
    """Multiset P/R/F1; both-empty is perfect, one-empty is zero."""
    reference, prediction = Counter(reference), Counter(prediction)
    matched = sum((reference & prediction).values())
    nref, npred = sum(reference.values()), sum(prediction.values())
    if nref == npred == 0:
        return (1.0, 1.0, 1.0)
    precision = matched / npred if npred else 0.0
    recall = matched / nref if nref else 0.0
    f1 = 2 * matched / (nref + npred) if nref + npred else 0.0
    return precision, recall, f1


NUMBER = re.compile(r"(?<!\w)\(?[+-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][+-]?\d+)?%?\)?")


def numeric_slots(table):
    return [
        (r, c, i, token)
        for (r, c), text in table.cells.items()
        for i, token in enumerate(NUMBER.findall(text))
    ]


def score_tables(reference, prediction):
    """Position-sensitive exact cells/numbers plus a position-free diagnostic.

    Missing/invalid predictions are zeros, never silently removed from means.
    Numeric F1 is undefined only if BOTH sides contain no numeric tokens.
    """
    keys = [
        "parse_success",
        "table_exact",
        "structure_exact",
        "row_count_exact",
        "column_count_exact",
        "cell_precision",
        "cell_recall",
        "cell_f1",
        "cell_bag_f1",
        "span_f1",
        "numeric_precision",
        "numeric_recall",
        "numeric_f1",
    ]
    result = dict.fromkeys(keys, 0.0)
    ref_nums = numeric_slots(reference)
    if prediction is None:
        result["numeric_f1"] = 0.0 if ref_nums else None
        result["numeric_precision"] = result["numeric_recall"] = result["numeric_f1"]
        return result
    same_dimensions = (reference.rows, reference.cols) == (
        prediction.rows,
        prediction.cols,
    )
    result.update(
        parse_success=1.0,
        structure_exact=float(same_dimensions and reference.spans == prediction.spans),
        table_exact=float(
            same_dimensions
            and reference.cells == prediction.cells
            and reference.spans == prediction.spans
        ),
        row_count_exact=float(reference.rows == prediction.rows),
        column_count_exact=float(reference.cols == prediction.cols),
    )
    p, r, f = prf(reference.cells.items(), prediction.cells.items())
    result.update(cell_precision=p, cell_recall=r, cell_f1=f)
    result["cell_bag_f1"] = prf(reference.cells.values(), prediction.cells.values())[2]
    result["span_f1"] = prf(reference.spans, prediction.spans)[2]
    pred_nums = numeric_slots(prediction)
    p, r, f = prf(ref_nums, pred_nums) if ref_nums or pred_nums else (None, None, None)
    result.update(numeric_precision=p, numeric_recall=r, numeric_f1=f)
    return result
