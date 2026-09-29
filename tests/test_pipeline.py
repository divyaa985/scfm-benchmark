"""Tests that run in seconds on CPU, no download required."""

import numpy as np
import pytest

from scfmbench.data import make_synthetic, normalize_log1p, select_hvg
from scfmbench.embedders import REGISTRY, build
from scfmbench.evaluate import run_benchmark, run_fold, summarise


def test_synthetic_dataset_shape():
    ds = make_synthetic(n_cells=200, n_genes=100, n_labels=4, n_batches=3)
    assert ds.adata.n_obs == 200
    assert len(ds.batches) == 3
    assert set(ds.labels) <= {f"type_{i}" for i in range(4)}


def test_normalize_log1p_is_finite_and_nonnegative():
    ds = make_synthetic(n_cells=50, n_genes=30)
    x = normalize_log1p(ds.adata.X)
    assert np.isfinite(x).all()
    assert (x >= 0).all()


def test_select_hvg_returns_requested_count_and_is_sorted():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=(100, 80))
    idx = select_hvg(x, n_top_genes=20)
    assert len(idx) == 20
    assert (np.diff(idx) > 0).all()
    assert idx.max() < 80


def test_select_hvg_caps_at_available_genes():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=(40, 15))
    assert len(select_hvg(x, n_top_genes=999)) == 15


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_embedder_fit_transform_roundtrip(name):
    ds = make_synthetic(n_cells=150, n_genes=80, n_batches=2)
    x = ds.adata.X.toarray()
    emb = build(name, seed=0).fit(x[:100])
    z = emb.transform(x[100:])
    assert z.shape[0] == 50
    assert np.isfinite(z).all()


def test_build_rejects_unknown_embedder():
    with pytest.raises(KeyError):
        build("no_such_embedder")


def test_fold_holds_out_the_named_batch_entirely():
    ds = make_synthetic(n_cells=300, n_genes=120, n_batches=3, seed=1)
    held = ds.batches[0]
    result = run_fold(ds, "hvg_pca", held, seed=0)
    n_held = int((ds.adata.obs["batch"].astype(str) == held).sum())
    assert result.n_test_cells == n_held
    assert result.n_train_cells == ds.adata.n_obs - n_held
    assert 0.0 <= result.macro_f1 <= 1.0


def test_pca_baseline_beats_the_negative_control():
    """The sanity check the whole benchmark rests on.

    Synthetic data has real cell-type structure, so a learned representation
    should transfer better than a random linear map. If this ever fails, the
    evaluation is measuring something other than biology.
    """
    ds = make_synthetic(n_cells=600, n_genes=300, n_labels=5, n_batches=3, seed=2)
    bench = run_benchmark(ds, ["hvg_pca", "random_projection"], seeds=(0,))
    summary = summarise(bench.folds)
    scores = summary.set_index("embedder")["macro_f1_mean"]
    assert scores["hvg_pca"] > scores["random_projection"]


def test_figures_are_written(tmp_path):
    """Plotting runs end to end and produces all three files."""
    from scfmbench.plots import make_all

    ds = make_synthetic(n_cells=300, n_genes=120, n_batches=3, seed=3)
    bench = run_benchmark(ds, ["hvg_pca", "random_projection"], seeds=(0,))
    paths = make_all(
        bench.folds, tmp_path, "test",
        per_class=bench.per_class, confusion=bench.confusion,
    )
    assert len(paths) == 5
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_per_class_and_confusion_tables_are_consistent():
    """Confusion counts must total the number of held-out cells in each fold."""
    ds = make_synthetic(n_cells=400, n_genes=150, n_batches=4, seed=7)
    bench = run_benchmark(ds, ["hvg_pca"], seeds=(0,))

    for batch in ds.batches:
        n_held = int((ds.adata.obs["batch"].astype(str) == batch).sum())
        counted = bench.confusion.loc[
            bench.confusion.held_out_batch == batch, "count"
        ].sum()
        assert counted == n_held

    assert bench.per_class["f1"].between(0, 1).all()
    assert (bench.per_class["support"] > 0).all()
