# GPU: SCENIC GRN inference on MetaCells (in-memory sparse)

**\[experimental\]** GPU equivalent of
[`bixverse::rs_mc_scenic`](https://gregorlueg.github.io/bixverse/reference/rs_mc_scenic.html).
Assumes the sparse data is pre-filtered for the genes / cells to
include. Indices are 0-indexed. GBM is not supported on GPU; use the CPU
version.

## Usage

``` r
rs_mc_scenic_gpu(
  sparse_data,
  tf_indices,
  scenic_params,
  wave_byte_budget,
  seed,
  verbose
)
```

## Arguments

- sparse_data:

  Named list with `data`, `indptr`, `indices`, `nrow`, `ncol`, `format`.

- tf_indices:

  Integer vector. 0-indexed positions of TF predictors.

- scenic_params:

  Named list. See
  [`bixverse::params_scenic()`](https://gregorlueg.github.io/bixverse/reference/params_scenic.html).

- wave_byte_budget:

  Double. VRAM ceiling for per-wave histogram + cumulative tensors
  (bytes). Default 4 GiB.

- seed:

  Integer. Random seed.

- verbose:

  Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.

## Value

A gene x TF importance matrix.
