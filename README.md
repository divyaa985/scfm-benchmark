# scfm-benchmark

**Do single-cell foundation model embeddings transfer to an unseen batch better than a PCA baseline — and better than a random projection?**

Single-cell foundation models are pretrained on tens of millions of cells and released with strong claims about general-purpose cell representations. A parallel literature keeps finding that simple, parameter-free representations match or beat them on downstream tasks, and that evaluating these models is harder than it looks. Both things are reported honestly; the disagreement usually comes down to protocol.

This repository is one protocol, applied identically to every method, with two deliberately stupid controls included so the numbers stay interpretable.

> **Status: baselines complete, foundation models in progress.** The controls and the evaluation harness came first on purpose — a foundation model number is only meaningful next to a baseline measured the same way. See [Roadmap](#roadmap).

---

## The protocol

**Leave-one-batch-out label transfer.** For each batch in turn: fit the embedder on all *other* batches, fit a logistic regression cell-type classifier on those embeddings, then predict the held-out batch. Repeat across every batch and three seeds.

Three design choices do the real work here:

**Nothing is fit on the held-out batch — including gene selection.** Highly variable gene selection, scaling and PCA are all fit on training cells and applied to test cells. Selecting HVGs on the full matrix before splitting is a quiet leak that inflates every method in the table, and it is common enough in published comparisons to be worth ruling out explicitly. See `select_hvg` in `src/scfmbench/data.py`.

**Held-out batches are different technologies, not random cell splits.** A random 80/20 split over pooled cells leaves near-duplicate cells on both sides and measures almost nothing. Holding out an entire assay is the situation an actual user faces.

**Macro F1 is the headline, not accuracy.** Cell type abundances are heavily skewed. A model that nails the two dominant types and fails on the other eleven can post excellent accuracy, which is precisely the failure mode worth catching.

### The controls

| Control | What it is | What it tells you |
|---|---|---|
| `random_projection` | Same preprocessing, then a random Gaussian linear map | Contains no learned biology. Anything that fails to beat it is not doing useful work. |
| `total_counts` | Two numbers per cell: log library size, log genes detected | If this scores well above chance, cell types in that dataset are partly separable by sequencing depth alone — worth knowing before interpreting anybody's benchmark. |

---

## Results

### Synthetic data (plumbing check, not a finding)

```
| embedder            | macro-F1 | ± sd  | balanced acc | ARI   | NMI   | dim | sec |
|---------------------|----------|-------|--------------|-------|-------|-----|-----|
| hvg_pca             | 1.000    | 0.000 | 1.000        | 1.000 | 1.000 | 50  | 0.0 |
| random_projection   | 0.937    | 0.028 | 0.937        | 0.956 | 0.949 | 50  | 0.0 |
| total_counts        | 0.217    | 0.032 | 0.277        | 0.063 | 0.120 | 2   | 0.0 |
```

Reproduce with `python -m scfmbench.run --synthetic`.

Read this as a wiring test only. The synthetic generator draws cell types from well-separated gamma programs, so the task is easy enough that even a random projection reaches 0.94 — which is itself the point of having the control. Note also how small the gap is between a real representation and a random one on easy data. Any benchmark reporting only the top row would look far more impressive than it deserves.

### Human pancreas — *pending*

Target dataset below. Results will be committed here with the full fold-level CSV, not just the summary.

---

## Getting the data

The first real dataset is the human pancreas benchmark from Luecken et al. 2022, which is the standard integration benchmark for exactly this question: roughly 16k cells across four sequencing technologies (inDrop, CEL-Seq2, Smart-Seq2, SMARTer) with curated cell type labels, so batch is confounded with assay in a realistic way.

Download `human_pancreas_norm_complexBatch.h5ad` from the Luecken et al. figshare collection ([10.6084/m9.figshare.12420968](https://doi.org/10.6084/m9.figshare.12420968)) into `data/`. Raw counts live in `layers['counts']`, the batch key is `tech` and the label key is `celltype`; the loader handles all three.

```bash
python -m scfmbench.run \
  --data data/human_pancreas_norm_complexBatch.h5ad \
  --batch-key tech --label-key celltype --name pancreas
```

---

## Install and run

```bash
git clone https://github.com/divyaa985/scfm-benchmark.git
cd scfm-benchmark
pip install -e ".[dev]"

pytest -q                              # 10 tests, no download needed
python -m scfmbench.run --synthetic    # full pipeline in under a minute
```

Everything so far runs on CPU. The foundation model embedders will need a GPU for the forward pass but no training — a free Colab T4 is enough for a dataset this size.

---

## Adding an embedder

One class, two methods, one registry line. The harness handles splits, seeds, metrics and reporting.

```python
class MyEmbedder:
    name = "my_embedder"

    def fit(self, x_train):      # cells x genes counts, training batches only
        return self

    def transform(self, x):      # -> (n_cells, embedding_dim)
        ...

REGISTRY["my_embedder"] = lambda seed: MyEmbedder()
```

---

## Roadmap

- [x] Leave-one-batch-out harness, leak-free preprocessing, two controls
- [x] `hvg_pca` baseline, test suite, CI
- [ ] Human pancreas results
- [ ] scVI baseline (trained per fold on training batches only)
- [ ] Geneformer zero-shot embeddings
- [ ] scGPT zero-shot embeddings
- [ ] A second dataset with a different confounding structure
- [ ] Write-up of where foundation models do and do not earn their cost

---

## Limitations

Stated up front rather than discovered by a reader.

Zero-shot embedding extraction is not the only way to use these models, and fine-tuning may well change the ranking; that is a separate and more expensive experiment. Logistic regression is a deliberately weak probe, chosen so the score reflects the embedding rather than the classifier, which means these numbers are a lower bound on achievable accuracy. Cell type labels in public atlases are themselves model-derived in places, so "ground truth" is doing some work in that phrase. And results on four pancreas technologies do not generalise to every tissue — hence a second dataset on the roadmap.

## Licence

MIT.
