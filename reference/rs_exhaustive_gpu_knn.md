# Generate an GPU-accelerated kNN graph from an exhaustive search

Runs an exhaustive kNN search on the GPU.

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

  String. Distance metric; one of `c("euclidean", "cosine")`.

- verbose:

  Logical. Whether to print progress messages.

## Value

A named list with:

- `indices` - Integer matrix of shape cells x k with 0-based neighbour
  indices.

- `dist` - Numeric matrix of shape cells x k with distances to the
  neighbours.

- `dist_metric` - Character. The distance metric used.
