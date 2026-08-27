# Run NMF (HALS) on the GPU over meta cells

**\[experimental\]** GPU counterpart of
[`bixverse::rs_nmf_single_mc()`](https://gregorlueg.github.io/bixverse/reference/rs_nmf_single_mc.html).
Assumes that the sparse data is pre-filtered for the cells/genes you
wish to include. Indices in the sparse data need to be 0-indexed.

## Usage

``` r
rs_nmf_single_mc_gpu(
  sparse_data,
  k,
  preprocessing,
  use_second_layer,
  nmf_hals_params,
  seed,
  verbose
)
```

## Arguments

- sparse_data:

  A named list with `data`, `indptr`, `indices`, `nrow`, `ncol` and
  `cs_type`.

- k:

  Integer. Number of latent factors to return. At most 128, the GPU
  solver's rank cap.

- preprocessing:

  String. One of `c("none", "sd", "sqrt_sd")`.

- use_second_layer:

  Boolean. If `TRUE`, runs NMF on normalised counts.

- nmf_hals_params:

  Named list. Contains the NMF parameters.

- seed:

  Integer. Random seed for initialisation.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list with `w`, `h`, `final_loss`, `n_iter`, `converged`.
