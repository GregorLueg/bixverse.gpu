# GPU-accelerated SCENIC GRN inference

GPU counterpart of
[`bixverse::scenic_grn_sc()`](https://gregorlueg.github.io/bixverse/reference/scenic_grn_sc.html).
Runs multi-output tree regression on the WGPU backend and returns a
`ScenicGrn` object. Dispatches on `SingleCells` (disk-backed .bin
counts) or `MetaCells` (in-memory sparse) from the `bixverse` package.
GBM is not supported on GPU; use the CPU version for
`learner_type = "grnboost2"`.

If `genes_to_take` is `NULL`, the CPU
[`bixverse::scenic_gene_filter_sc()`](https://gregorlueg.github.io/bixverse/reference/scenic_gene_filter_sc.html)
is used to shortlist targets (cheap min-counts / min-cells scan).

## Usage

``` r
scenic_grn_sc_gpu(
  object,
  tf_ids,
  scenic_params = bixverse::params_scenic(),
  wave_byte_budget = 4 * 1024^3,
  genes_to_take = NULL,
  cells_to_take = NULL,
  streaming = NULL,
  random_seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `MetaCells` class from `bixverse`.

- tf_ids:

  Character vector. Gene identifiers of the transcription factors to use
  as predictors.

- scenic_params:

  Named list. See
  [`bixverse::params_scenic()`](https://gregorlueg.github.io/bixverse/reference/params_scenic.html).

- wave_byte_budget:

  Numeric. VRAM ceiling for per-wave histogram and cumulative tensors
  (bytes). Default 4 GiB. Shrink on tight VRAM budgets, raise on 16 GB+
  adapters to let the scheduler pick a wider wave.

- genes_to_take:

  Optional character vector. Target genes to include. If `NULL`, the CPU
  gene filter runs first.

- cells_to_take:

  Optional character vector. Cell names to include. If `NULL`, all
  filtered cells are used.

- streaming:

  Optional boolean. Only used on `SingleCells`. If `TRUE`, the streaming
  GPU driver is used (bounded host memory). If `NULL`, is auto-picked
  from cell count via bixverse's internal `auto_streaming`. Ignored for
  `MetaCells`.

- random_seed:

  Integer. For reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity. `FALSE` -\> quiet, `TRUE` or
  `1L` -\> normal, `2L` -\> detailed.

## Value

A `ScenicGrn` object with the gene x TF importance matrix.

## References

Aibar et al., Nat Methods, 2017.
