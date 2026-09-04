# Find neighbours via CAGRA GPU-acceleration for single cells (deprecated)

**\[deprecated\]**

CAGRA now sits behind
[`find_neighbours_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_gpu_sc.md)
as `knn_method = "nndescent"`.

## Usage

``` r
find_neighbours_cagra_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  cagra_params = params_nn_gpu(),
  extract_knn = FALSE,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- embd_to_use:

  String. The embedding to use.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions.

- modality:

  String. One of `c("rna", "adt")`.

- cagra_params:

  List. Output of the deprecated
  [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md),
  or
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- extract_knn:

  Logical. Skip the beam search.

- snn_params:

  List. Output of
  [`bixverse::params_sc_neighbours()`](https://gregorlueg.github.io/bixverse/reference/params_sc_neighbours.html).

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean. Controls verbosity.

## Value

The object with added kNN matrix and sNN graph.
