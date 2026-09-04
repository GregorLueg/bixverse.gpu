# Find GPU-accelerated neighbours for single cells

This function generates kNN data using GPU-accelerated algorithms via
the `bixverse.gpu` package, then turns it into an sNN igraph for
downstream clustering. See
[`generate_gpu_knn_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_gpu_knn_sc.md)
for the three searches on offer. This function lives in a separate
package from the CPU-based
[`find_neighbours_sc()`](https://gregorlueg.github.io/bixverse/reference/find_neighbours_sc.html)
so that users without GPU hardware do not need to install the GPU
dependencies.

## Usage

``` r
find_neighbours_gpu_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  k = 15L,
  snn_params = params_sc_neighbours(),
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

  String. The embedding to use.

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

- snn_params:

  List. Output of
  [`bixverse::params_sc_neighbours()`](https://gregorlueg.github.io/bixverse/reference/params_sc_neighbours.html).
  The kNN graph-related parameters will be ignored in favour of
  `nn_params`.

- seed:

  Integer. For reproducibility.

- gpu_method:

  **\[deprecated\]** Use `knn_method`.

- ivf_params:

  **\[deprecated\]** Use `nn_params`.

- dist_metric:

  **\[deprecated\]** Use `params_nn_gpu(dist_metric = )`.

- .verbose:

  Boolean. Controls verbosity.

## Value

The object with added kNN matrix and sNN graph in the selected modality
slot.
