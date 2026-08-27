# Run multiple NMF (HALS) restarts on the GPU over meta cells

**\[experimental\]** GPU counterpart of
[`bixverse::rs_nmf_multi_mc()`](https://gregorlueg.github.io/bixverse/reference/rs_nmf_multi_mc.html).
Assumes that the sparse data is pre-filtered for the cells/genes you
wish to include. Indices in the sparse data need to be 0-indexed.

## Usage

``` r
rs_nmf_multi_mc_gpu(
  sparse_data,
  k,
  preprocessing,
  use_second_layer,
  nmf_hals_params,
  n_runs,
  seed,
  verbose
)
```

## Arguments

- sparse_data:

  A named list with `data`, `indptr`, `indices`, `nrow`, `ncol` and
  `cs_type`.

- k:

  Integer. Number of latent factors per run. At most 128, the GPU
  solver's rank cap.

- preprocessing:

  String. One of `c("none", "sd", "sqrt_sd")`.

- use_second_layer:

  Boolean. If `TRUE`, runs NMF on normalised counts.

- nmf_hals_params:

  Named list. Contains the NMF parameters.

- n_runs:

  Integer. Number of random restarts.

- seed:

  Integer. Base random seed. Run `i` uses `seed + i`.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list with `w_all`, `h_per_run`, `losses`, `converged`, `best_idx`
(1-indexed).
