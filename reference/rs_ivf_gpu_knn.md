# IVF-GPU-accelerated kNN graph (deprecated)

**\[deprecated\]**

The three GPU kNN searches went behind one wrapper, see
[`rs_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_gpu_knn.md).
Note that Euclidean distances now come back as true L2 rather than
squared.

## Usage

``` r
rs_ivf_gpu_knn(embd, ivf_params, seed, verbose)
```

## Arguments

- embd:

  Numeric matrix of embeddings, cells x features.

- ivf_params:

  Named list, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- seed:

  Integer. Random seed for reproducibility.

- verbose:

  Integer. `0L` quiet, `1L` normal, `2L` detailed.

## Value

A named list with `indices`, `dist` and `dist_metric`.
