# Generate GPU kNN data for single cells

This function generates a `SingleCellNearestNeighbour` object using
GPU-accelerated kNN algorithms via the `bixverse.gpu` package. Three
methods are available: `"exhaustive"` performs an exact brute-force
search on the GPU; `"ivf"` builds an inverted file index that partitions
the embedding space into Voronoi cells and probes only a subset at query
time; and `"nndescent"` builds a dense NNDescent graph and prunes it
into a CAGRA navigational graph, which is then either beam searched or
handed back as the descent left it (`params_nn_gpu(extract_knn = TRUE)`,
faster, lower recall). This function is the GPU counterpart of
[`generate_knn_sc()`](https://gregorlueg.github.io/bixverse/reference/generate_knn_sc.html).

## Usage

``` r
generate_gpu_knn_sc(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  k = 15L,
  seed = 42L,
  gpu_method = lifecycle::deprecated(),
  ivf_params = lifecycle::deprecated(),
  dist_metric = lifecycle::deprecated(),
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

- knn_method:

  String. One of `c("nndescent", "exhaustive", "ivf")`.

- nn_params:

  List. Output of
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- k:

  Integer. Number of neighbours.

- seed:

  Integer. For reproducibility.

- gpu_method:

  **\[deprecated\]** Use `knn_method`.

- ivf_params:

  **\[deprecated\]** Use `nn_params`.

- dist_metric:

  **\[deprecated\]** Use `params_nn_gpu(dist_metric = )`.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

Initialised `sc_knn` with the kNN data.
