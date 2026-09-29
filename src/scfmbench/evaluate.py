"""The evaluation protocol.

Cell-type annotation accuracy measured *within* a batch is close to free, and
most published comparisons report it anyway. The question that matters for a
practitioner is whether an embedding transfers to a batch the model has never
seen, produced on a different technology. So the protocol here is
leave-one-batch-out: fit the embedder and the classifier on every other batch,
predict the held-out one, repeat across batches and seeds.

Macro F1 is the headline metric rather than accuracy, because cell-type
abundances are heavily skewed and accuracy rewards a model that only gets the
two dominant types right.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .embedders import build


@dataclass
class FoldResult:
    dataset: str
    embedder: str
    held_out_batch: str
    seed: int
    macro_f1: float
    balanced_accuracy: float
    ari: float
    nmi: float
    n_train_cells: int
    n_test_cells: int
    n_labels_test: int
    fit_seconds: float
    embed_dim: int


def _matrix(dataset: Dataset, mask: np.ndarray) -> np.ndarray:
    x = dataset.adata[mask].X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def run_fold(
    dataset: Dataset,
    embedder_name: str,
    held_out_batch: str,
    seed: int = 0,
) -> FoldResult:
    """Fit on all batches but one, evaluate on the held-out batch."""
    batches = dataset.adata.obs[dataset.batch_key].astype(str).to_numpy()
    labels = dataset.labels

    test_mask = batches == held_out_batch
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(f"Empty split for held-out batch {held_out_batch!r}")

    x_train, x_test = _matrix(dataset, train_mask), _matrix(dataset, test_mask)
    y_train, y_test = labels[train_mask], labels[test_mask]

    start = time.perf_counter()
    embedder = build(embedder_name, seed=seed).fit(x_train)
    z_train = embedder.transform(x_train)
    z_test = embedder.transform(x_test)
    fit_seconds = time.perf_counter() - start

    # Scale on train statistics only, then transfer labels.
    scaler = StandardScaler().fit(z_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
        scaler.transform(z_train), y_train
    )
    y_pred = clf.predict(scaler.transform(z_test))

    # Unsupervised structure in the held-out batch, k set to the true number of
    # types so the score reflects separability rather than a lucky k.
    k = len(np.unique(y_test))
    if k > 1 and z_test.shape[0] > k:
        clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z_test)
        ari = adjusted_rand_score(y_test, clusters)
        nmi = normalized_mutual_info_score(y_test, clusters)
    else:
        ari = nmi = float("nan")

    return FoldResult(
        dataset=dataset.name,
        embedder=embedder_name,
        held_out_batch=held_out_batch,
        seed=seed,
        macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        ari=float(ari),
        nmi=float(nmi),
        n_train_cells=int(train_mask.sum()),
        n_test_cells=int(test_mask.sum()),
        n_labels_test=int(k),
        fit_seconds=round(fit_seconds, 2),
        embed_dim=int(z_train.shape[1]),
    )


def run_benchmark(
    dataset: Dataset,
    embedder_names: list[str],
    seeds: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Every embedder x every held-out batch x every seed."""
    rows = []
    for embedder_name in embedder_names:
        for batch in dataset.batches:
            for seed in seeds:
                result = run_fold(dataset, embedder_name, batch, seed)
                rows.append(asdict(result))
                print(
                    f"  {embedder_name:<24} held out {batch:<16} "
                    f"seed {seed}  macro-F1 {result.macro_f1:.3f}"
                )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation per embedder, across batches and seeds."""
    agg = (
        results.groupby("embedder")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_acc_mean=("balanced_accuracy", "mean"),
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            seconds_mean=("fit_seconds", "mean"),
            embed_dim=("embed_dim", "max"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .round(3)
    )
    return agg.reset_index()


def to_markdown(summary: pd.DataFrame) -> str:
    header = "| embedder | macro-F1 | ± sd | balanced acc | ARI | NMI | dim | sec |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r.embedder}` | **{r.macro_f1_mean:.3f}** | {r.macro_f1_std:.3f} | "
            f"{r.balanced_acc_mean:.3f} | {r.ari_mean:.3f} | {r.nmi_mean:.3f} | "
            f"{int(r.embed_dim)} | {r.seconds_mean:.1f} |"
        )
    return "\n".join(lines)
