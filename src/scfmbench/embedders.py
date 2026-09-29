"""Embedders under test.

Every embedder is fit on training cells and then applied to held-out cells,
via the same two-method interface. Adding a foundation model means writing one
class here and registering it — nothing else in the benchmark changes.

The random projection embedder is the negative control and is the most
important object in this file. It has no biology in it at all. Any embedder
that fails to beat it is not doing useful work, and reporting that number
keeps the rest of the table honest.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from .data import normalize_log1p, select_hvg


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a cell-by-gene count matrix into a cell embedding."""

    name: str

    def fit(self, x_train: np.ndarray) -> Embedder: ...

    def transform(self, x: np.ndarray) -> np.ndarray: ...


class HVGPCAEmbedder:
    """The baseline that keeps beating things: log1p, HVGs, scale, PCA."""

    def __init__(self, n_hvg: int = 2000, n_comps: int = 50, seed: int = 0):
        self.n_hvg = n_hvg
        self.n_comps = n_comps
        self.seed = seed
        self.name = f"hvg{n_hvg}_pca{n_comps}"

    def fit(self, x_train: np.ndarray) -> HVGPCAEmbedder:
        x = normalize_log1p(x_train)
        self.hvg_idx_ = select_hvg(x, self.n_hvg)
        x = x[:, self.hvg_idx_]

        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        x = (x - self.mean_) / self.std_

        n_comps = min(self.n_comps, *x.shape)
        self.pca_ = PCA(n_components=n_comps, random_state=self.seed).fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = normalize_log1p(x)[:, self.hvg_idx_]
        x = (x - self.mean_) / self.std_
        return self.pca_.transform(x)


class RandomProjectionEmbedder:
    """Negative control: same preprocessing, then a random linear map.

    Carries no learned biological structure, so it measures how much of a
    method's score comes from the preprocessing and the evaluation protocol
    rather than from the representation itself.
    """

    def __init__(self, n_hvg: int = 2000, n_comps: int = 50, seed: int = 0):
        self.n_hvg = n_hvg
        self.n_comps = n_comps
        self.seed = seed
        self.name = f"random_projection{n_comps}"

    def fit(self, x_train: np.ndarray) -> RandomProjectionEmbedder:
        x = normalize_log1p(x_train)
        self.hvg_idx_ = select_hvg(x, self.n_hvg)
        x = x[:, self.hvg_idx_]
        n_comps = min(self.n_comps, x.shape[1])
        self.proj_ = GaussianRandomProjection(
            n_components=n_comps, random_state=self.seed
        ).fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = normalize_log1p(x)[:, self.hvg_idx_]
        return self.proj_.transform(x)


class TotalCountsEmbedder:
    """Second control: library size and detected-gene count, nothing else.

    Two numbers per cell. If this scores well above chance on a dataset, the
    cell types in that dataset are partly separable by sequencing depth alone,
    which is worth knowing before interpreting anyone's benchmark.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.name = "total_counts_only"

    def fit(self, x_train: np.ndarray) -> TotalCountsEmbedder:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        total = np.log1p(x.sum(axis=1))
        n_genes = np.log1p((x > 0).sum(axis=1))
        return np.column_stack([total, n_genes])


# Registry: name -> factory taking a seed. Foundation models get added here.
REGISTRY: dict[str, callable] = {
    "hvg_pca": lambda seed: HVGPCAEmbedder(seed=seed),
    "random_projection": lambda seed: RandomProjectionEmbedder(seed=seed),
    "total_counts": lambda seed: TotalCountsEmbedder(seed=seed),
}


def build(name: str, seed: int = 0) -> Embedder:
    if name not in REGISTRY:
        raise KeyError(f"Unknown embedder {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](seed)
