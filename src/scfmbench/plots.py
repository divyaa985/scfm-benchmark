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


def make_all(
    results: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    per_class: pd.DataFrame | None = None,
    confusion: pd.DataFrame | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_macro_f1(results, out_dir, dataset),
        plot_by_batch(results, out_dir, dataset),
        plot_gap_to_control(results, out_dir, dataset),
    ]
    if per_class is not None and len(per_class):
        paths.append(plot_per_celltype_f1(per_class, out_dir, dataset))
    if confusion is not None and len(confusion):
        paths.append(plot_confusion(confusion, results, out_dir, dataset))
    return [p for p in paths if p is not None]


def plot_per_celltype_f1(per_class: pd.DataFrame, out: Path, dataset: str) -> Path:
    """Which cell types each method fails on.

    Macro F1 is an average over this plot. A method can look respectable in
    aggregate while scoring near zero on every rare type, and that is exactly
    the situation a practitioner needs to see before trusting an annotation.
    """
    support = (
        per_class.groupby("celltype")["support"].mean().sort_values(ascending=True)
    )
    types = list(support.index)
    embedders = list(per_class.groupby("embedder")["f1"].mean().sort_values().index)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(types) + 2.4))
    offsets = np.linspace(-0.26, 0.26, max(len(embedders), 2))

    for k, name in enumerate(embedders):
        colour = WARN if name == CONTROL else PALETTE[k % len(PALETTE)]
        means = (
            per_class[per_class.embedder == name]
            .groupby("celltype")["f1"]
            .mean()
            .reindex(types)
        )
        ax.scatter(
            means.to_numpy(),
            np.arange(len(types)) + offsets[k],
            s=42,
            color=colour,
            label=name,
            zorder=3,
            lw=0,
        )

    for i, celltype in enumerate(types):
        ax.text(
            1.03,
            i,
            f"n={int(support[celltype])}",
            fontsize=8,
            color=MUTED,
            va="center",
        )

    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(types) - 0.3)
    ax.set_xlabel("per-cell-type F1, averaged over held-out assays and seeds")
    ax.set_title(
        f"Where each method fails — {dataset}\n"
        "cell types ordered by abundance, rarest at the bottom",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.grid(axis="x", color=HAIRLINE, lw=0.6, alpha=0.7)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(len(embedders), 4),
        labelcolor=INK,
    )
    _style(ax)

    fig.tight_layout()
    path = out / f"{dataset}_per_celltype_f1.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_confusion(confusion: pd.DataFrame, results: pd.DataFrame, out: Path,
                   dataset: str) -> Path:
    """Best method against the control, as row-normalised confusion matrices.

    Two panels rather than one, because the interesting question is not what
    the good method confuses but whether the control confuses anything else.
    """
    ranked = results.groupby("embedder")["macro_f1"].mean().sort_values(ascending=False)
    best = ranked.index[0]
    panels = [best] + ([CONTROL] if CONTROL in ranked.index and CONTROL != best else [])

    labels = sorted(set(confusion.true_label) | set(confusion.pred_label))
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.2 * len(panels) + 1.5, 0.42 * len(labels) + 3.2)
    )
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, panels):
        sub = confusion[confusion.embedder == name]
        grid = (
            sub.pivot_table(
                index="true_label", columns="pred_label", values="count", aggfunc="sum"
            )
            .reindex(index=labels, columns=labels)
            .fillna(0)
        )
        totals = grid.sum(axis=1).replace(0, 1)
        frac = grid.div(totals, axis=0)

        ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = frac.to_numpy()[i, j]
                if value < 0.01:
                    continue
                ax.text(
                    j, i, f"{value:.2f}".lstrip("0"),
                    ha="center", va="center", fontsize=7.5,
                    color="white" if value > 0.55 else INK,
                )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_title(f"{name}  (macro F1 {ranked[name]:.3f})", fontsize=10, color=INK)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)

    axes[0].set_ylabel("true cell type")
    fig.suptitle(
        f"Confusion on held-out assays — {dataset}   (rows sum to 1)",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / f"{dataset}_confusion.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_umap_panels(dataset_obj, embedder_names: list[str], out: Path,
                     held_out: str | None = None, seed: int = 0) -> Path:
    """UMAP of every embedding space, for illustration only.

    A deliberate warning lives in this docstring and in the figure caption.
    UMAP distances between clusters are not meaningful, cluster density is not
    meaningful, and n_neighbors and min_dist can be tuned until almost any
    embedding looks well separated. Nothing in this repository's conclusions
    rests on these panels.

    They are here for one reason: put the random projection beside the real
    embedding and look at how similar the pictures are. A representation with
    no learned biology in it produces a perfectly respectable UMAP. That is
    the argument against reading these plots as evidence, made visually.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "UMAP panels need umap-learn: pip install umap-learn"
        ) from exc

    from .embedders import build

    held_out = held_out or dataset_obj.batches[0]
    batches = dataset_obj.adata.obs[dataset_obj.batch_key].astype(str).to_numpy()
    labels = dataset_obj.labels
    test_mask = batches == held_out

    x = dataset_obj.adata.X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    n = len(embedder_names)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 9.0), squeeze=False)

    celltypes = sorted(set(labels))
    colour_of = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(celltypes)}

    for col, name in enumerate(embedder_names):
        embedder = build(name, seed=seed).fit(x[~test_mask])
        z = embedder.transform(x)
        coords = umap.UMAP(
            n_neighbors=15, min_dist=0.3, random_state=seed
        ).fit_transform(z)

        ax = axes[0][col]
        for c in celltypes:
            m = labels == c
            ax.scatter(coords[m, 0], coords[m, 1], s=4, lw=0, alpha=0.7,
                       color=colour_of[c], label=c if col == 0 else None)
        ax.set_title(name, fontsize=11, color=INK)
        if col == 0:
            ax.set_ylabel("coloured by cell type", fontsize=10, color=MUTED)

        ax = axes[1][col]
        ax.scatter(coords[~test_mask, 0], coords[~test_mask, 1], s=4, lw=0,
                   alpha=0.45, color=HAIRLINE, label="training assays" if col == 0 else None)
        ax.scatter(coords[test_mask, 0], coords[test_mask, 1], s=4, lw=0,
                   alpha=0.8, color=WARN, label=f"held out: {held_out}" if col == 0 else None)
        if col == 0:
            ax.set_ylabel("held-out assay highlighted", fontsize=10, color=MUTED)

        for row in (0, 1):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            for side in ("top", "right", "left", "bottom"):
                axes[row][col].spines[side].set_color(HAIRLINE)

    axes[0][0].legend(frameon=False, fontsize=8, markerscale=2.5, loc="best",
                      labelcolor=INK)
    axes[1][0].legend(frameon=False, fontsize=8, markerscale=2.5, loc="best",
                      labelcolor=INK)
    fig.suptitle(
        f"UMAP of each embedding space — {dataset_obj.name}\n"
        "Illustration only. UMAP distances and densities are not meaningful, and no "
        "conclusion here rests on these panels.\n"
        "Compare the random projection against the learned embedding: a representation "
        "with no biology in it still looks convincing.",
        x=0.01, y=0.995, ha="left", va="top", fontsize=10.5, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    path = out / f"{dataset_obj.name}_umap.png"
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path
