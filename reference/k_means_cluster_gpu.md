# GPU-accelerated k-means clustering

Performs k-means clustering as a GPU-accelerated form. In its current
form uses the cubecl with wgpu backend.

## Usage

``` r
k_means_cluster_gpu(
  data,
  k,
  kmeans_params = params_kmeans_gpu(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- data:

  Numerical matrix or data frame. The data to cluster, of shape samples
  x features. Will be coerced to a matrix.

- k:

  Integer. Number of clusters to create. Must be \>= 2.

- kmeans_params:

  Named list. GPU-accelerated k-mean parameters, see
  [`params_kmeans_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_kmeans_gpu.md).

- seed:

  Integer. Random seed for reproducibility. Defaults to 42L.

- .verbose:

  Logical. Controls verbosity. Defaults to `TRUE`.

## Value

A `KMeansClusterGPU` class with assignments and centroids.
