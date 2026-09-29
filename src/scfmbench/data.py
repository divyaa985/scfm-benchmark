"""Dataset loading and leakage-free preprocessing.

The one rule this module enforces: every preprocessing step that *learns*
something from the data (highly variable gene selection, scaling, PCA) is fit
on training cells only and then applied to held-out cells. Fitting HVGs on the
full matrix before splitting is a common and quietly serious leak in
single-cell benchmarks, because gene selection sees the test batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


@dataclass
class Dataset:
    """A benchmark dataset with a batch key and a cell-type label key."""

    adata: ad.AnnData
    batch_key: str
    label_key: str
    name: str

    @property
    def batches(self) -> list[str]:
        return sorted(self.adata.obs[self.batch_key].astype(str).unique())

    @property
    def labels(self) -> np.ndarray:
        return self.adata.obs[self.label_key].astype(str).to_numpy()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, n_cells={self.adata.n_obs}, "
            f"n_genes={self.adata.n_vars}, n_batches={len(self.batches)})"
        )


def load_h5ad(
    path: str | Path,
    batch_key: str,
    label_key: str,
    name: str | None = None,
    min_cells_per_label: int = 20,
    counts_layer: str | None = "counts",
) -> Dataset:
    """Load an .h5ad file and drop cell types too rare to evaluate.

    Parameters
    ----------
    counts_layer
        If present in ``adata.layers``, raw counts are moved into ``adata.X``.
        The scIB pancreas file ships normalised values in ``X`` and raw counts
        in ``layers['counts']``; we always start from counts so preprocessing
        is under our control rather than the file author's.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See README section 'Getting the data'."
        )

    adata = ad.read_h5ad(path)

    if counts_layer and counts_layer in adata.layers:
        adata.X = adata.layers[counts_layer].copy()

    for key in (batch_key, label_key):
        if key not in adata.obs:
            raise KeyError(f"{key!r} not in adata.obs. Found: {list(adata.obs)}")

    counts = adata.obs[label_key].value_counts()
    keep_labels = counts[counts >= min_cells_per_label].index
    adata = adata[adata.obs[label_key].isin(keep_labels)].copy()

    return Dataset(
        adata=adata,
        batch_key=batch_key,
        label_key=label_key,
        name=name or path.stem,
    )


def make_synthetic(
    n_cells: int = 600,
    n_genes: int = 400,
    n_labels: int = 5,
    n_batches: int = 3,
    batch_effect: float = 0.8,
    seed: int = 0,
) -> Dataset:
    """Small synthetic dataset with real cell-type signal and a batch effect.

    Used by the test suite and CI so the pipeline is exercised without a
    multi-gigabyte download. Counts are Poisson draws from a mean that depends
    on both cell type and batch, so a good embedding should recover the type
    and a bad one should collapse into batches.
    """
    rng = np.random.default_rng(seed)

    label_ids = rng.integers(0, n_labels, size=n_cells)
    batch_ids = rng.integers(0, n_batches, size=n_cells)

    label_programs = rng.gamma(shape=2.0, scale=1.0, size=(n_labels, n_genes))
    batch_programs = rng.gamma(shape=2.0, scale=1.0, size=(n_batches, n_genes))

    mean = label_programs[label_ids] + batch_effect * batch_programs[batch_ids]
    counts = rng.poisson(mean).astype(np.float32)

    adata = ad.AnnData(sp.csr_matrix(counts))
    adata.obs["celltype"] = [f"type_{i}" for i in label_ids]
    adata.obs["batch"] = [f"batch_{i}" for i in batch_ids]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    return Dataset(adata, batch_key="batch", label_key="celltype", name="synthetic")


def normalize_log1p(x: np.ndarray | sp.spmatrix, target_sum: float = 1e4) -> np.ndarray:
    """Counts-per-target_sum normalisation followed by log1p. Returns dense."""
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    x = x.astype(np.float64, copy=True)
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    x = x / totals * target_sum
    return np.log1p(x)


def select_hvg(x_train: np.ndarray, n_top_genes: int = 2000) -> np.ndarray:
    """Pick highly variable genes by normalised dispersion, fit on train only.

    A binned dispersion criterion in the spirit of Seurat: within bins of
    similar mean expression, keep the genes whose variance-to-mean ratio stands
    out. Implemented here rather than called from scanpy so the selection is
    explicitly fit-then-transform and testable in isolation.
    """
    n_top_genes = min(n_top_genes, x_train.shape[1])

    mean = x_train.mean(axis=0)
    var = x_train.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean > 0, var / mean, 0.0)

    n_bins = 20
    # Rank-based bins keep every bin populated even for skewed mean profiles.
    order = np.argsort(mean)
    bin_of_gene = np.empty(len(mean), dtype=int)
    bin_of_gene[order] = np.minimum(
        (np.arange(len(mean)) * n_bins) // len(mean), n_bins - 1
    )

    z = np.zeros_like(dispersion)
    for b in range(n_bins):
        idx = bin_of_gene == b
        if idx.sum() < 2:
            continue
        d = dispersion[idx]
        scale = d.std()
        z[idx] = (d - d.mean()) / scale if scale > 0 else 0.0

    return np.sort(np.argsort(-z)[:n_top_genes])
