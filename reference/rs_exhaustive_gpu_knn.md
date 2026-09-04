# Exhaustive GPU-accelerated kNN graph (deprecated)

**\[deprecated\]**

The three GPU kNN searches went behind one wrapper, see
[`rs_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_gpu_knn.md).
Note that Euclidean distances now come back as true L2 rather than
squared.

## Usage

``` r
rs_exhaustive_gpu_knn(embd, k, dist_metric, verbose)
```

## Arguments

- embd:

  Numeric matrix of embeddings, cells x features.

- k:

  Integer. Number of neighbours to return.

- dist_metric:

  String. One of `c("euclidean", "cosine")`.

- verbose:

  Integer. `0L` quiet, `1L` normal, `2L` detailed.

## Value

A named list with `indices`, `dist` and `dist_metric`.
