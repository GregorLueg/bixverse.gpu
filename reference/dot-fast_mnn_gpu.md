# Shared implementation of the GPU fastMNN

Body behind both
[`fast_mnn_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/fast_mnn_gpu_sc.md)
methods.

## Usage

``` r
.fast_mnn_gpu(
  object,
  batch_column,
  no_embd_to_use,
  fastmnn_params,
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

- no_embd_to_use:

  Optional integer. Number of PCs to use. If `NULL` all will be used.

- fastmnn_params:

  List. Output of
  [`params_sc_fastmnn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fastmnn_gpu.md).

- seed:

  Integer. Random seed.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

The object with the `"mnn_gpu"` embedding set.
