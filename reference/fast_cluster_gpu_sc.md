# Run fast clustering on a SingleCells object (GPU)

GPU counterpart of
[`bixverse::fast_cluster_sc()`](https://gregorlueg.github.io/bixverse/reference/fast_cluster_sc.html).
Coarsens the cells with k-means on the chosen embedding, builds a graph
over the centroids (a kNN graph or the cell-mediated graph), runs
Louvain or Leiden on it and propagates the communities back to the
cells. With `n_seeds > 1` every resolution is run once per seed and
stability statistics are returned.

Only the k-means coarsening (and the cell-mediated top-m search) run on
the GPU. The centroid graph and the community detection stay on the CPU,
so the speedup tracks how much of the run k-means owns. That share grows
with the cell count and with `n_centroids`. There is no `km_type`
argument: the GPU k-means is full-batch Lloyd's and has no mini-batch
path.

## Usage

``` r
fast_cluster_gpu_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  resolutions = c(2, 1, 0.5),
  fc_params = params_sc_fast_cluster_gpu(),
  n_seeds = 1L,
  return_kmeans = FALSE,
  seed = 42L,
  backend = c("wgpu", "mlx"),
  n_centroids = lifecycle::deprecated(),
  snn = lifecycle::deprecated(),
  grid_search = lifecycle::deprecated(),
  no_seeds = lifecycle::deprecated(),
  .verbose = TRUE
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

`SingleCellFastClusters` S3 object with:

- memberships:

  data.table with `cell_idx` and one column per resolution
  (`res_<value>`).

- stats:

  data.table of the stability statistics, or `NULL` for `n_seeds = 1L`.

- k_means_cluster:

  Integer vector of k-means assignments, or `NULL`.

- centroids:

  Numeric matrix of centroids, or `NULL`.

- resolutions:

  Resolutions used.

with `cell_indices` stored as an attribute (0-indexed).
