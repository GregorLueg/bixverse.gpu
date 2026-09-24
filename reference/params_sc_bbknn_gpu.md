# Wrapper function for the GPU BBKNN parameters

GPU counterpart to
[`bixverse::params_sc_bbknn()`](https://gregorlueg.github.io/bixverse/reference/params_sc_bbknn.html).
Same BBKNN knobs, but the kNN block is the GPU one, see
[`params_knn_gpu_defaults()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_knn_gpu_defaults.md).

## Usage

``` r
params_sc_bbknn_gpu(
  neighbours_within_batch = 3L,
  set_op_mix_ratio = 1,
  local_connectivity = 1,
  trim = NULL,
  knn = list()
)
```

## Arguments

- neighbours_within_batch:

  Integer. Number of neighbours to consider per batch. Defaults to `3L`.

- set_op_mix_ratio:

  Numeric. Mixing ratio between union (1.0) and intersection (0.0).
  Defaults to `1.0`.

- local_connectivity:

  Numeric. UMAP connectivity computation parameter, how many nearest
  neighbours of each cell are assumed to be fully connected. Defaults to
  `1.0`.

- trim:

  Optional integer. Trim the neighbours of each cell to these many top
  connectivities. May help with population independence and improve the
  tidiness of clustering. If `NULL`, it defaults to
  `10 * neighbours_within_batch`.

- knn:

  List. Optional overrides for the kNN block. Validated against
  [`params_knn_gpu_defaults()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_knn_gpu_defaults.md)
  minus `k` and `extract_knn`. Unknown keys are an error, not a silent
  pass-through.

## Value

A flat named list with all GPU BBKNN parameters.

## Details

Two keys of the GPU kNN block do nothing here and are rejected rather
than silently ignored. `k` is set by `neighbours_within_batch`, and
`extract_knn` only applies to a self-query, whereas BBKNN builds one
index per batch and queries each with every cell.

## References

Polański, et al., Bioinformatics, 2020
