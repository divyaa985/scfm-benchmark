"""Figures for a benchmark run.

Three plots, each answering a question the summary table cannot:

1. Per-fold spread. A mean hides whether a method is reliably good or wildly
   variable across assays. Every fold is drawn as its own point.
2. Which assay is hard. A method can average well and still collapse on one
   technology, which is the case a user actually cares about.
3. Distance from the control. The benchmark's whole claim is about the gap to
   a random projection, so that gap gets its own figure rather than being
   left for the reader to compute.

Every function takes the fold-level dataframe, so the same code works for
synthetic data, pancreas, or anything added later.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONTROL = "random_projection"

INK = "#2C2C2A"
MUTED = "#73726C"
HAIRLINE = "#D3D1C7"
ACCENT = "#1D9E75"
WARN = "#D85A30"
PALETTE = ["#378ADD", "#1D9E75", "#7F77DD", "#D85A30", "#BA7517", "#D4537E"]


def _style(ax) -> None:
    """Strip the chart down to ink that carries information."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)
    ax.set_axisbelow(True)


def _order(results: pd.DataFrame) -> list[str]:
    means = results.groupby("embedder")["macro_f1"].mean().sort_values()
    return list(means.index)


def plot_macro_f1(results: pd.DataFrame, out: Path, dataset: str) -> Path:
    """Headline figure: every fold as a point, mean and spread behind it."""
    order = _order(results)
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(order) + 2.2))
    rng = np.random.default_rng(0)

    control_mean = None
    if CONTROL in results["embedder"].values:
        control_mean = results.loc[results.embedder == CONTROL, "macro_f1"].mean()
        ax.axvline(control_mean, color=WARN, lw=1.2, ls="--", zorder=1)
        ax.text(
            control_mean,
            -0.78,
            "random projection control",
            color=WARN,
            fontsize=9,
            ha="center",
            va="center",
        )

    for i, name in enumerate(order):
        sub = results[results.embedder == name]["macro_f1"]
        colour = WARN if name == CONTROL else PALETTE[i % len(PALETTE)]
        jitter = rng.uniform(-0.13, 0.13, len(sub))
        ax.scatter(sub, i + jitter, s=26, color=colour, alpha=0.45, zorder=3, lw=0)
        ax.errorbar(
            sub.mean(),
            i,
            xerr=sub.std(),
            fmt="o",
            ms=9,
            color=colour,
            ecolor=colour,
            elinewidth=1.4,
            capsize=4,
            zorder=4,
        )
        ax.text(
            sub.mean(),
            i + 0.3,
            f"{sub.mean():.3f}",
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1.05, len(order) - 0.3)
    ax.set_xlabel("macro F1 on the held-out assay")
    ax.set_title(
        f"Cross-assay label transfer — {dataset}\n"
        "one point per held-out assay and seed; bars are mean ± sd",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.grid(axis="x", color=HAIRLINE, lw=0.6, alpha=0.7)
    _style(ax)

    fig.tight_layout()
    path = out / f"{dataset}_macro_f1.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_by_batch(results: pd.DataFrame, out: Path, dataset: str) -> Path:
    """Which held-out assay is hard, and for whom."""
    grid = (
        results.pivot_table(
            index="embedder", columns="held_out_batch", values="macro_f1"
        )
        .reindex(_order(results)[::-1])
    )

    fig, ax = plt.subplots(figsize=(1.5 * grid.shape[1] + 3.5, 0.75 * len(grid) + 2.2))
    im = ax.imshow(grid.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if value > 0.55 else INK,
            )

    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(grid.columns, rotation=20, ha="right")
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("held-out assay")
    ax.set_title(
        f"Macro F1 by held-out assay — {dataset}\naveraged over seeds",
        loc="left",
        fontsize=11,
        color=INK,
    )
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).outline.set_visible(False)

    fig.tight_layout()
    path = out / f"{dataset}_by_batch.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_gap_to_control(results: pd.DataFrame, out: Path, dataset: str) -> Path:
    """Per-fold margin over the random projection. Zero is the line that matters."""
    if CONTROL not in results["embedder"].values:
        return None

    control = results[results.embedder == CONTROL].set_index(
        ["held_out_batch", "seed"]
    )["macro_f1"]
    others = results[results.embedder != CONTROL].copy()
    others["gap"] = others.apply(
        lambda r: r.macro_f1 - control.get((r.held_out_batch, r.seed), np.nan), axis=1
    )

    order = [n for n in _order(results) if n != CONTROL]
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(order) + 2.2))
    rng = np.random.default_rng(1)

    ax.axvline(0, color=WARN, lw=1.2, ls="--", zorder=1)
    for i, name in enumerate(order):
        gaps = others[others.embedder == name]["gap"].dropna()
        colour = ACCENT if gaps.mean() > 0 else WARN
        ax.scatter(
            gaps,
            i + rng.uniform(-0.13, 0.13, len(gaps)),
            s=26,
            color=colour,
            alpha=0.45,
            zorder=3,
            lw=0,
        )
        ax.errorbar(
            gaps.mean(),
            i,
            xerr=gaps.std(),
            fmt="o",
            ms=9,
            color=colour,
            elinewidth=1.4,
            capsize=4,
            zorder=4,
        )
        ax.text(
            gaps.mean(),
            i + 0.3,
            f"{gaps.mean():+.3f}",
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.75, len(order) - 0.35)
    ax.set_xlabel("macro F1 minus the random projection, same assay and seed")
    ax.set_title(
        f"Margin over the control — {dataset}\n"
        "left of the dashed line means the method lost to random",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.grid(axis="x", color=HAIRLINE, lw=0.6, alpha=0.7)
    _style(ax)

    fig.tight_layout()
    path = out / f"{dataset}_gap_to_control.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def make_all(results: pd.DataFrame, out_dir: Path, dataset: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_macro_f1(results, out_dir, dataset),
        plot_by_batch(results, out_dir, dataset),
        plot_gap_to_control(results, out_dir, dataset),
    ]
    return [p for p in paths if p is not None]
