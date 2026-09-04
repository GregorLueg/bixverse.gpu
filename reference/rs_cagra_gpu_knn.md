# CAGRA-style GPU-accelerated kNN graph (deprecated)

**\[deprecated\]**

The three GPU kNN searches went behind one wrapper, see
[`rs_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_gpu_knn.md).
Note that Euclidean distances now come back as true L2 rather than
squared.

## Usage

``` r
rs_cagra_gpu_knn(embd, cagra_params, extract_knn, seed, verbose)
```

## Arguments

- embd:

  Numeric matrix of embeddings, cells x features.

- cagra_params:

  Named list, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- extract_knn:

  Logical. Skip the beam search.

- seed:

  Integer. Random seed for reproducibility.

- verbose:

  Integer. `0L` quiet, `1L` normal, `2L` detailed.

## Value

A named list with `indices`, `dist` and `dist_metric`.
