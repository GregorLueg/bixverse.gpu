# Run UMAP on a SingleCells object (GPU)

GPU-accelerated counterpart to
[`bixverse::umap_sc()`](https://gregorlueg.github.io/bixverse/reference/umap_sc.html).
Pulls an embedding (defaulting to PCA) off the object, runs
[`umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu.md)
on it (GPU kNN plus GPU Adam optimiser by default), and writes the
resulting embedding back into `sc_cache$other_embeddings[[slot_name]]`.

When `use_knn = TRUE` (the default), the kNN graph already stored on the
object is reused via
[`bixverse::sc_knn_to_nearest_neighbours()`](https://gregorlueg.github.io/bixverse/reference/sc_knn_to_nearest_neighbours.html).
Otherwise a fresh GPU kNN is built from the chosen embedding.

## Usage

``` r
umap_gpu_sc(
  object,
  use_knn = TRUE,
  embd_to_use = "pca",
  slot_name = "umap",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  k = 15L,
  min_dist = 0.5,
  spread = 1,
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- use_knn:

  Boolean. Use the kNN graph found in the object. Defaults to `TRUE`.
  Only reused if the modality lines up; otherwise a fresh GPU kNN is
  generated.

- embd_to_use:

  String. The embedding to use for UMAP. Must be available in the object
  for the chosen modality.

- slot_name:

  String. The name of this embedding within the object. Defaults to
  `"umap"`.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL`,
  all will be used.

- modality:

  String. On which modality to run UMAP. One of
  `c("rna", "adt", "wnn")`. The two latter options are only available on
  `SingleCellsMultiModal`.

- n_dim:

  Integer. Number of UMAP dimensions. Defaults to `2L`.

- k:

  Integer. Number of nearest neighbours. Defaults to `15L`.

- min_dist:

  Numeric. Minimum distance between embedded points. Defaults to `0.5`.

- spread:

  Numeric. Effective scale of embedded points. Defaults to `1.0`.

- knn_method:

  String. GPU (approximate) nearest neighbour method. One of
  `c("nndescent", "exhaustive", "ivf")`.

- nn_params:

  Named list. GPU kNN parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- umap_params:

  Named list. UMAP (GPU) parameters, see
  [`params_umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_umap_gpu.md).

- seed:

  Integer. For reproducibility.

- use_high_precision:

  Optional boolean. Fine-grained fp32 vs fp64 control for the optimiser.
  GPU kNN is always fp32.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

The object with a `"umap"` embedding added. If the requested embedding
is missing, returns the object unchanged with a warning.

## See also

[`umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu.md),
[`bixverse::umap_sc()`](https://gregorlueg.github.io/bixverse/reference/umap_sc.html),
[`tsne_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu_sc.md)
