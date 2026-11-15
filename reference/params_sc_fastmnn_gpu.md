# Wrapper function for the GPU fastMNN parameters

GPU counterpart to
[`bixverse::params_sc_fastmnn()`](https://gregorlueg.github.io/bixverse/reference/params_sc_fastmnn.html).
Same fastMNN knobs minus the PCA ones, since the GPU path corrects the
PCA already stored in the object. The kNN block is the GPU one, see
[`params_knn_gpu_defaults()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_knn_gpu_defaults.md),
with `k` and `ann_dist` defaulting to the CPU values.

## Usage

``` r
params_sc_fastmnn_gpu(ndist = 3, cos_norm = TRUE, knn = list())
```

## Arguments

- ndist:

  Numeric. Number of median distances for the tricube kernel bandwidth.
  Defaults to `3.0`.

- cos_norm:

  Boolean. Apply cosine normalisation before computing distances.
  Defaults to `TRUE`.

- knn:

  List. Optional overrides for the kNN block. See
  [`params_knn_gpu_defaults()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_knn_gpu_defaults.md)
  for the available elements. Unknown elements are an error. Defaults to
  [`list()`](https://rdrr.io/r/base/list.html).

## Value

A named list with the following elements:

- ndist - Numeric. Number of median distances for the tricube kernel
  bandwidth. Defaults to `3.0`.

- cos_norm - Boolean. Apply cosine normalisation before computing
  distances. Defaults to `TRUE`.

- The elements of
  [`params_knn_gpu_defaults()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_knn_gpu_defaults.md),
  overridden by `knn`, spliced in at this position.

## Details

Every search in fastMNN is a cross-query between two sets of cells, so
`knn_method = "nndescent"` builds and beam searches a CAGRA graph here.
`k = 0L` is rejected, since fastMNN has no data-driven fallback for it.

## References

Haghverdi, et al., Nat Biotechnol, 2018
