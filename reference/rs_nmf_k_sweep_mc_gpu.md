# Sweep k on the GPU over meta cells

**\[experimental\]** GPU counterpart of
[`bixverse::rs_nmf_k_sweep_mc()`](https://gregorlueg.github.io/bixverse/reference/rs_nmf_k_sweep_mc.html).
Returns diagnostics only, no factors. The matrix is uploaded once and
serves every one of the `length(k_range) * n_runs` solves, which is
where the GPU path pays off.

## Usage

``` r
rs_nmf_k_sweep_mc_gpu(
  sparse_data,
  k_range,
  preprocessing,
  use_second_layer,
  nmf_hals_params,
  nmf_consensus_params,
  n_runs,
  seed,
  verbose
)
```

## Arguments

- sparse_data:

  A named list with `data`, `indptr`, `indices`, `nrow`, `ncol` and
  `cs_type`.

- k_range:

  Integer vector. Ranks to evaluate, every entry at least 2 and at most
  128, the GPU solver's rank cap.

- preprocessing:

  String. One of `c("none", "sd", "sqrt_sd")`.

- use_second_layer:

  Boolean. If `TRUE`, runs NMF on normalised counts.

- nmf_hals_params:

  Named list. Contains the NMF parameters.

- nmf_consensus_params:

  Named list. Contains the consensus parameters.

- n_runs:

  Integer. Number of restarts per k. Must be at least 2.

- seed:

  Integer. Base random seed.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list of equal-length vectors, one element per swept k: `k`,
`stability`, `best_error`, `median_error`, `consensus_failed`,
`n_dropped`, `n_empty_clusters` and `n_converged`.

## References

Kotliar et al., eLife, 2019
