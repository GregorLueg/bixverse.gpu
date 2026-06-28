# Run Harmony v2 (GPU)

A GPU-accelerated version of Harmony v2 by Patikas et al., 2026,
implemented in Rust. Performs batch correction on PCA embeddings and
stores the result as a `"harmony_gpu"` embedding in the object. Only a
single batch covariate is supported on the GPU path.

## Usage

``` r
harmony_v2_gpu_sc(
  object,
  batch_column,
  modality = c("rna", "adt"),
  harmony_params = params_sc_harmony_v2_gpu(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` class.

- batch_column:

  String. Column name in the object containing the batch labels.

- modality:

  String. One of `c("rna", "adt")`. You can only use `"adt"` on
  `SingleCellsMultiModal` class.

- harmony_params:

  List. Output of
  [`params_sc_harmony_v2_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_harmony_v2_gpu.md).

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

The object with a `"harmony_gpu"` embedding added. If no PCA embeddings
are found, returns the object unchanged with a warning.
