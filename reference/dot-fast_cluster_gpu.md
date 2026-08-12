# Shared implementation of the GPU fast Louvain clustering

Body behind both
[`fast_cluster_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/fast_cluster_gpu_sc.md)
methods. Pulls the embedding off the object, hands it to
[`rs_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_fast_cluster_gpu.md)
or
[`rs_fast_cluster_grid_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_fast_cluster_grid_gpu.md)
and wraps the result into a `SingleCellFastClusters` S3 object.

## Usage

``` r
.fast_cluster_gpu(
  object,
  embd_to_use,
  no_embd_to_use,
  resolutions,
  n_centroids,
  fc_params,
  snn,
  return_kmeans,
  grid_search,
  no_seeds,
  seed,
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

  Numeric vector. Louvain resolutions.

- n_centroids:

  Optional integer. Number of k-means centroids. Defaults to
  `sqrt(n_cells)` Rust-side if `NULL`. Clamped to `n_cells - 1`.

- fc_params:

  List. Output of
  [`params_sc_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fast_cluster_gpu.md).

- snn:

  Boolean. Convert the centroid kNN to an sNN graph.

- return_kmeans:

  Boolean. Return the k-means assignments and centroids.

- grid_search:

  Boolean. Run the multi-seed grid version.

- no_seeds:

  Integer. Number of seeds to vary Louvain over. Must be at least 2.
  Only used when `grid_search = TRUE`.

- seed:

  Integer. Seed for reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

A `SingleCellFastClusters` S3 object.
