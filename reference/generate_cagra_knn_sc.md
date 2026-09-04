# Generate CAGRA GPU kNN data for single cells (deprecated)

**\[deprecated\]**

CAGRA now sits behind
[`generate_gpu_knn_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_gpu_knn_sc.md)
as `knn_method = "nndescent"`.

## Usage

``` r
generate_cagra_knn_sc(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  cagra_params = params_nn_gpu(),
  extract_knn = TRUE,
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- embd_to_use:

  String. The embedding to use.

- cells_to_use:

  Optional string vector. Cell names to include.

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

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

Initialised `sc_knn` with the kNN data.
