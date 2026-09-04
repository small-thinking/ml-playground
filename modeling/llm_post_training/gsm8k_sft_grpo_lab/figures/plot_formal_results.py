"""Render the curated formal-results figure from the audited GSM8K ledger.

The values intentionally live in this small, reviewable script rather than in
an ad-hoc plotting notebook. Update a value only together with the W&B-backed
row in ``../experiment_log.md`` and the report in ``../README.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FormalResult:
    """One selected checkpoint evaluated by the shared frozen G4 protocol."""

    experiment: str
    label: str
    pass_at_1: float
    pass_at_4: float


RESULTS: tuple[FormalResult, ...] = (
    FormalResult("E0", "E0\nBase", 0.6603, 0.6884),
    FormalResult("E2", "E2\nSFT", 0.6898, 0.7195),
    FormalResult("E4", "E4\nclean GRPO", 0.7197, 0.7506),
    FormalResult("E5", "E5\nsignal-pack", 0.7038, 0.7296),
    FormalResult("E6", "E6\nfixed budget", 0.7034, 0.7343),
    FormalResult("E7", "E7\nfixed-sign", 0.7675, 0.7879),
    FormalResult("E8", "E8\ntoken-matched", 0.7020, 0.7296),
    FormalResult("E9", "E9\nhard KD", 0.9126, 0.9308),
)


def _add_value_labels(axis: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.05,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )


def render(output: Path) -> None:
    """Create a two-panel result figure with comparison-scope caveats visible."""

    labels = [result.label for result in RESULTS]
    pass_at_1 = np.array([100 * result.pass_at_1 for result in RESULTS])
    pass_at_4 = np.array([100 * result.pass_at_4 for result in RESULTS])
    locations = np.arange(len(RESULTS))
    width = 0.34

    figure = plt.figure(figsize=(18, 10), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(3.45, 1.2),
        left=0.055,
        right=0.97,
        bottom=0.16,
        top=0.80,
        wspace=0.04,
    )
    axis = figure.add_subplot(grid[0, 0])
    notes = figure.add_subplot(grid[0, 1])

    # Background bands communicate comparison scope without changing metric color.
    band_specs = {
        "E4": ("#e7f3e9", "controlled GRPO\nreference"),
        "E7": ("#fff3dc", "12.7× E4\noptimization tokens"),
        "E8": ("#f1f1f1", "≈ E4 optimization\ntoken budget"),
        "E9": ("#eee9f7", "E9: Base → KD"),
    }
    for index, result in enumerate(RESULTS):
        if result.experiment in band_specs:
            color, label = band_specs[result.experiment]
            axis.axvspan(index - 0.48, index + 0.48, color=color, zorder=0)
            axis.text(
                index,
                98.2,
                label,
                ha="center",
                va="top",
                fontsize=8.3,
                color="#31343a",
                linespacing=1.15,
            )

    first_bars = list(
        axis.bar(
            locations - width / 2,
            pass_at_1,
            width,
            label="Pass@1",
            color="#78808b",
            zorder=3,
        )
    )
    fourth_bars = list(
        axis.bar(
            locations + width / 2,
            pass_at_4,
            width,
            label="Pass@4",
            color="#12799a",
            zorder=3,
        )
    )
    _add_value_labels(axis, first_bars)
    _add_value_labels(axis, fourth_bars)

    # Bars use a zero baseline so visual distance is not mistaken for effect size.
    axis.set_ylim(0, 100)
    axis.set_xlim(-0.62, len(RESULTS) - 0.38)
    axis.set_ylabel("Formal score (%)", fontsize=14, fontweight="semibold")
    axis.set_xticks(locations, labels, fontsize=10, fontweight="semibold")
    axis.set_yticks(np.arange(0, 101, 10))
    axis.grid(axis="y", color="#c9cdd2", linewidth=0.8, alpha=0.9, zorder=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#5c626b",
        fontsize=11,
    )

    notes.axis("off")
    notes.set_xlim(0, 1)
    notes.set_ylim(0, 1)
    notes.text(0.04, 0.92, "How to read this figure", fontsize=16, fontweight="bold")
    notes.text(
        0.04,
        0.78,
        "E4 is the controlled\nSFT → GRPO reference.",
        fontsize=13,
        fontweight="semibold",
        linespacing=1.45,
    )
    notes.text(
        0.04,
        0.58,
        "E7 has a higher point estimate,\nbut its 696,641 optimized tokens\nare 12.7× E4's 54,760.",
        fontsize=11.5,
        linespacing=1.45,
    )
    notes.text(
        0.04,
        0.35,
        "E8 approximately matches E4's\noptimization-token budget and\ndoes not beat it.",
        fontsize=11.5,
        linespacing=1.45,
    )
    notes.text(
        0.04,
        0.12,
        "E9 is a strong parallel\nBase → KD result, not an\nequal-coverage causal control\nagainst historical E4.",
        fontsize=11.5,
        linespacing=1.45,
    )

    figure.suptitle(
        "GSM8K post-training: formal results and comparison scope",
        fontsize=24,
        fontweight="bold",
        y=0.955,
    )
    figure.text(
        0.5,
        0.895,
        "Shared frozen protocol: 1,287 prompts × 4 independent samples · temperature 1.0",
        ha="center",
        fontsize=14,
    )
    figure.text(
        0.055,
        0.055,
        "Pass@1 = mean correctness across four rollouts; Pass@4 = at least one correct rollout. "
        "All results reuse one public GSM8K formal protocol; do not treat it as a fresh generalization claim.",
        fontsize=10,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("gsm8k-posttraining-formal-results-v2.png"),
        help="PNG path to write.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args().output)
