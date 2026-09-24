# Shared implementation of the GPU BBKNN

Body behind both
[`bbknn_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/bbknn_gpu_sc.md)
methods. Mirrors
[`bixverse::bbknn_sc()`](https://gregorlueg.github.io/bixverse/reference/bbknn_sc.html)
step for step, with
[`rs_bbknn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_bbknn_gpu.md)
in place of the CPU search.

## Usage

``` r
.bbknn_gpu(
  object,
  batch_column,
  no_neighbours_to_keep,
  embd_to_use,
  no_embd_to_use,
  bbknn_params,
  seed,
  .verbose
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- batch_column:

  String. The column with the batch information in the obs data of the
  class.

- no_neighbours_to_keep:

  Integer. Maximum number of neighbours to keep from the BBKNN
  algorithm. Generating neighbours per batch can produce a lot of them,
  so this keeps the top `no_neighbours_to_keep`. Defaults to `5L`.

- embd_to_use:

  String. The embedding to use. Atm, the only option is `"pca"`.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL` all
  will be used.

- bbknn_params:

  List. Output of
  [`params_sc_bbknn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_bbknn_gpu.md).

- seed:

  Integer. Random seed.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

The object with the kNN matrix and the connectivity graph set.
