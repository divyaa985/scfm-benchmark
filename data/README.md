# data/

Datasets are not committed. This folder is where they go.

## human_pancreas_norm_complexBatch.h5ad

The Luecken et al. 2022 integration benchmark: roughly 16k cells across four
sequencing technologies with curated cell type labels.

Download from the figshare collection
[10.6084/m9.figshare.12420968](https://doi.org/10.6084/m9.figshare.12420968)
and place the `.h5ad` file in this directory, then:

```bash
python -m scfmbench.run \
  --data data/human_pancreas_norm_complexBatch.h5ad \
  --batch-key tech --label-key celltype --name pancreas
```

Raw counts are in `layers['counts']`, the batch key is `tech`, the label key
is `celltype`. The loader moves counts into `.X` so every method starts from
the same place regardless of how the file was normalised.
