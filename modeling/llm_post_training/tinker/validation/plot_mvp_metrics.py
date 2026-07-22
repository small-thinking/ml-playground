"""Render the static PR chart for the validated three-step W&B run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "assets" / "2026-07-21-sft-wandb-mvp-metrics.png"
BLUE = "#2563EB"
BLUE_DARK = "#1E3A8A"
GOLD = "#C78B00"
INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
BACKGROUND = "#FAFAF8"
WANDB_HISTORY = [
    {
        "step": 1,
        "loss": 0.02475300927956899,
        "train_tokens": 30,
        "cumulative_tokens": 30,
        "seconds": 2.160577667003963,
        "step_cost": 0.00002211,
        "cumulative_cost": 0.00002211,
    },
    {
        "step": 2,
        "loss": 0.000018099503904522862,
        "train_tokens": 30,
        "cumulative_tokens": 60,
        "seconds": 2.1245552920154296,
        "step_cost": 0.00002211,
        "cumulative_cost": 0.00004422,
    },
    {
        "step": 3,
        "loss": 0.0000019669454710917003,
        "train_tokens": 30,
        "cumulative_tokens": 90,
        "seconds": 2.1796885000076145,
        "step_cost": 0.00002211,
        "cumulative_cost": 0.00006633,
    },
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
        1.18,
        title,
        transform=axis.transAxes,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.0,
        1.08,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        color=MUTED,
        fontsize=9,
    )


def render_chart(output_path: Path) -> Path:
    """Render four honest discrete-step views; three points are not a trend."""
    rows = WANDB_HISTORY
    steps = [row["step"] for row in rows]
    loss = [row["loss"] for row in rows]
    seconds = [row["seconds"] for row in rows]
    cumulative_tokens = [row["cumulative_tokens"] for row in rows]
    cumulative_cost = [row["cumulative_cost"] for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=180)
    figure.patch.set_facecolor(BACKGROUND)
    figure.suptitle(
        "Tinker SFT + Weights & Biases MVP — synced metrics",
        x=0.06,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.922,
        "Qwen/Qwen3.5-4B • 3 steps • 2 smoke examples per step • W&B run 3zne613h • 2026-07-21",
        ha="left",
        color=MUTED,
        fontsize=10,
    )

    loss_axis, time_axis, token_axis, cost_axis = axes.flatten()
    for axis in axes.flatten():
        _style_axis(axis)
        axis.set_xticks(steps, [f"Step {step}" for step in steps])

    loss_axis.scatter(steps, loss, color=BLUE, edgecolor=BLUE_DARK, s=90, zorder=3)
    loss_axis.set_yscale("log")
    loss_axis.set_ylim(min(loss) / 1.7, max(loss) * 2.8)
    _panel_header(
        loss_axis,
        "Mean cross-entropy loss",
        "Log scale; exact values labeled",
    )
    for step, value in zip(steps, loss):
        x_offset = 8 if step == steps[0] else 0
        horizontal_alignment = "left" if step == steps[0] else "center"
        loss_axis.annotate(
            f"{value:.3g}",
            (step, value),
            xytext=(x_offset, 9),
            textcoords="offset points",
            ha=horizontal_alignment,
            color=INK,
            fontsize=9,
        )

    time_axis.bar(steps, seconds, color=BLUE, edgecolor=BLUE_DARK, width=0.58)
    time_axis.set_ylim(0, max(seconds) * 1.3)
    _panel_header(
        time_axis,
        "Step duration",
        "Seconds; zero baseline",
    )
    for step, value in zip(steps, seconds):
        time_axis.text(step, value + 0.05, f"{value:.2f}s", ha="center", color=INK)

    token_axis.bar(
        steps,
        cumulative_tokens,
        color=BLUE,
        edgecolor=BLUE_DARK,
        width=0.58,
    )
    token_axis.set_ylim(0, max(cumulative_tokens) * 1.25)
    _panel_header(
        token_axis,
        "Cumulative train tokens",
        "30 input tokens processed per step",
    )
    for step, value in zip(steps, cumulative_tokens):
        token_axis.text(step, value + 2.5, str(value), ha="center", color=INK)

    cost_axis.bar(
        steps,
        cumulative_cost,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.7,
        width=0.58,
    )
    cost_axis.set_ylim(0, max(cumulative_cost) * 1.3)
    cost_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.5f}"))
    _panel_header(
        cost_axis,
        "Estimated cumulative train cost",
        "USD estimate; excludes the two bounded samples",
    )
    for step, value in zip(steps, cumulative_cost):
        cost_axis.text(
            step,
            value + 0.000002,
            f"${value:.5f}",
            ha="center",
            color=INK,
            fontsize=9,
        )

    figure.text(
        0.06,
        0.025,
        "Source: W&B API readback for run 3zne613h. Plumbing validation only; not evidence of model quality.",
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.77,
        hspace=0.72,
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
