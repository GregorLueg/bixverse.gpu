# Run fast Louvain clustering on a SingleCells object (GPU)

GPU counterpart of
[`bixverse::fast_cluster_sc()`](https://gregorlueg.github.io/bixverse/reference/fast_cluster_sc.html).
Runs k-means on the chosen embedding, builds a kNN graph on the
centroids, applies Louvain clustering and propagates the memberships
back to the cells. Optionally runs a grid over multiple seeds and
returns stability statistics.

Only the k-means coarsening runs on the WGPU backend. The centroid kNN,
the optional sNN pass and the Louvain runs stay on the CPU, so the
speedup tracks how much of the run k-means owns. That share grows with
the cell count and with `n_centroids`. There is no `km_type` argument:
the GPU k-means is full-batch Lloyd's and has no mini-batch path.

## Usage

``` r
fast_cluster_gpu_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  resolutions = c(2, 1, 0.5),
  n_centroids = NULL,
  fc_params = params_sc_fast_cluster_gpu(),
  snn = TRUE,
  return_kmeans = FALSE,
  grid_search = FALSE,
  no_seeds = 10L,
  seed = 42L,
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

`SingleCellFastClusters` S3 object with:

- memberships:

  data.table with `cell_idx` and one column per resolution
  (`res_<value>`).

- stats:

  data.table of grid statistics, or `NULL`.

- k_means_cluster:

  Integer vector of k-means assignments, or `NULL`.

- centroids:

  Numeric matrix of centroids, or `NULL`.

- resolutions:

  Resolutions used.

with `cell_indices` stored as an attribute (0-indexed).
