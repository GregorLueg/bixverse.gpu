# Generate GPU kNN data for single cells (exhaustive / IVF)

This function generates a `SingleCellNearestNeighbour` object using
GPU-accelerated kNN algorithms via the `bixverse.gpu` package. Two
methods are available: `"exhaustive"` performs an exact brute-force
search on the GPU; `"ivf"` builds an inverted file index that partitions
the embedding space into Voronoi cells and probes only a subset at query
time, trading a small amount of precision for considerably faster search
on larger data sets. This function is the GPU counterpart of
[`generate_knn_sc()`](https://gregorlueg.github.io/bixverse/reference/generate_knn_sc.html).

## Usage

``` r
generate_gpu_knn_sc(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  gpu_method = c("ivf", "exhaustive"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "euclidean",
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- embd_to_use:

  String. The embedding to use. Whichever you choose, it needs to be
  part of the object for the selected modality.

- cells_to_use:

  Optional string vector. Cell names to include. If `NULL` all cells in
  the object will be used.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL` all
  will be used.

- modality:

  String. One of `c("rna", "adt")`. You can only use `"adt"` on
  `SingleCellsMultiModal` class.

- gpu_method:

  String. One of `c("exhaustive", "ivf")`.

- ivf_params:

  List. Output of
  [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md).
  Only used when `gpu_method = "ivf"`.

- k:

  Integer. Number of neighbours. Only used when
  `gpu_method = "exhaustive"`.

- dist_metric:

  String. One of `c("euclidean", "cosine")`. Only used when
  `gpu_method = "exhaustive"`.

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

Initialised `sc_knn` with the kNN data.
