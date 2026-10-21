# Shared implementation of the GPU fast clustering

Body behind both
[`fast_cluster_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/fast_cluster_gpu_sc.md)
methods. Pulls the embedding off the object, resolves the `NULL`
defaults of
[`params_sc_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fast_cluster_gpu.md),
hands it to
[`rs_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_fast_cluster_gpu.md)
and wraps the result into a `SingleCellFastClusters` S3 object.

## Usage

``` r
.fast_cluster_gpu(
  object,
  embd_to_use,
  no_embd_to_use,
  resolutions,
  fc_params,
  n_seeds,
  return_kmeans,
  seed,
  backend,
  n_centroids,
  snn,
  grid_search,
  no_seeds,
  .verbose
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- embd_to_use:

  String. Embedding name. Defaults to `"pca"`.

- no_embd_to_use:

  Optional integer. Number of dimensions to keep.

- resolutions:

  Numeric vector. Resolutions of the community detection.

- fc_params:

  List. Output of
  [`params_sc_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fast_cluster_gpu.md).

- n_seeds:

  Integer. Number of seeds per resolution. `1L` runs a single pass
  without stability statistics.

- return_kmeans:

  Boolean. Return the k-means assignments and centroids.

- seed:

  Integer. Seed for reproducibility.

- backend:

  String. One of `c("wgpu", "mlx")`. `"mlx"` runs the k-means on Apple
  Silicon via MLX and needs a build with the MLX backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md).

- n_centroids:

  **\[deprecated\]** Use `params_sc_fast_cluster_gpu(n_centroids = )`.

- snn:

  **\[deprecated\]** Use
  `params_sc_fast_cluster_gpu(centroid_graph = list(knn_sym = ))`.

- grid_search:

  **\[deprecated\]** Use `n_seeds`.

- no_seeds:

  **\[deprecated\]** Use `n_seeds`.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

A `SingleCellFastClusters` S3 object.
