"""Render the static PR chart for the validated 10-step DeepMath SFT run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "assets" / "2026-07-22-deepmath-sft-10-step-metrics.png"
BLUE = "#2563EB"
BLUE_DARK = "#1E3A8A"
GOLD = "#C78B00"
RED = "#C2413B"
INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
BACKGROUND = "#FAFAF8"

# Read back from the W&B API after run 2ta5u0vj reached state=finished.
WANDB_HISTORY = [
    (1, 0.4623094529, 6280, 6280, 3.8835, 0.004628360),
    (2, 0.5173393833, 6283, 12563, 2.8801, 0.009258931),
    (3, 0.5005686416, 5314, 17877, 5.0062, 0.013175349),
    (4, 0.3893148619, 6634, 24511, 2.9210, 0.018064607),
    (5, 0.4231030112, 3792, 28303, 2.7256, 0.020859311),
    (6, 0.3833002950, 5544, 33847, 4.0043, 0.024945239),
    (7, 0.3333928656, 5379, 39226, 3.2448, 0.028909562),
    (8, 0.5990941256, 6643, 45869, 3.1662, 0.033805453),
    (9, 0.3919607414, 5535, 51404, 2.9143, 0.037884748),
    (10, 0.4397435708, 4895, 56299, 2.5439, 0.041492363),
]


def _style_axis(axis: Any) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)


def _panel_header(axis: Any, title: str, subtitle: str) -> None:
    axis.text(
        0.0,
        1.16,
        title,
        transform=axis.transAxes,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.0,
        1.065,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        color=MUTED,
        fontsize=9,
    )


def render_chart(output_path: Path) -> Path:
    steps = [row[0] for row in WANDB_HISTORY]
    losses = [row[1] for row in WANDB_HISTORY]
    train_tokens = [row[2] for row in WANDB_HISTORY]
    cumulative_tokens = [row[3] for row in WANDB_HISTORY]
    seconds = [row[4] for row in WANDB_HISTORY]
    cumulative_cost = [row[5] for row in WANDB_HISTORY]

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.8), dpi=180)
    figure.patch.set_facecolor(BACKGROUND)
    figure.suptitle(
        "DeepMath SFT pilot — 10-step Tinker + W&B validation",
        x=0.06,
        y=0.968,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.925,
        "Execution PASS • quality comparison INCONCLUSIVE • "
        "8/8 baseline and final samples truncated",
        ha="left",
        color=RED,
        fontsize=10,
        fontweight="bold",
    )

    loss_axis, time_axis, token_axis, cost_axis = axes.flatten()
    for axis in axes.flatten():
        _style_axis(axis)
        axis.set_xticks(steps)

    loss_axis.plot(
        steps,
        losses,
        color=BLUE,
        marker="o",
        markeredgecolor=BLUE_DARK,
        linewidth=2,
    )
    loss_axis.set_ylim(0, max(losses) * 1.22)
    _panel_header(
        loss_axis,
        "Mean cross-entropy loss",
        "Different two-example batches; fluctuations are expected",
    )
    loss_axis.annotate(
        "min 0.333",
        (7, losses[6]),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        color=INK,
        fontsize=9,
    )

    time_axis.bar(steps, seconds, color=BLUE, edgecolor=BLUE_DARK, width=0.65)
    time_axis.set_ylim(0, max(seconds) * 1.22)
    _panel_header(
        time_axis, "Step duration", "Seconds per forward/backward + optimizer"
    )

    token_axis.bar(
        steps,
        train_tokens,
        color=BLUE,
        edgecolor=BLUE_DARK,
        width=0.65,
    )
    token_axis.set_ylim(0, max(train_tokens) * 1.22)
    _panel_header(
        token_axis,
        "Train tokens per step",
        f"Variable-length batches; {cumulative_tokens[-1]:,} total train tokens",
    )
    token_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )

    cost_axis.plot(
        steps,
        cumulative_cost,
        color=GOLD,
        marker="o",
        markeredgecolor=INK,
        linewidth=2,
    )
    cost_axis.fill_between(steps, cumulative_cost, color=GOLD, alpha=0.12)
    cost_axis.set_ylim(0, max(cumulative_cost) * 1.25)
    cost_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.3f}"))
    _panel_header(
        cost_axis,
        "Estimated cumulative train cost",
        "Token-price estimate; total with both eval passes was $0.05037",
    )

    figure.text(
        0.06,
        0.025,
        "Source: W&B API readback, run 2ta5u0vj (state=finished). "
        "Loss alone is not a quality metric.",
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.79,
        hspace=0.65,
        wspace=0.25,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(figure)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(render_chart(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
