# GPU: fast Louvain clustering on the data (with multiple seeds)

**\[experimental\]** GPU equivalent of
[`bixverse::rs_fast_cluster_sc_grid`](https://gregorlueg.github.io/bixverse/reference/rs_fast_cluster_sc_grid.html).
Builds the k-means to kNN/sNN graph once, then runs Louvain with several
seeds (derived from the original one) for every resolution. Returns
additional metrics around cluster stability and community conductance.
Only the k-means runs on the GPU.

## Usage

``` r
rs_fast_cluster_grid_gpu(
  embd,
  resolutions,
  n_centroids,
  fc_params,
  snn,
  return_kmeans,
  no_seeds,
  seed,
  verbose
)
```

## Arguments

- embd:

  Numeric matrix. The original embedding.

- resolutions:

  Numeric vector. The Louvain resolutions to iterate through.

- n_centroids:

  Optional integer. The number of clusters to find. If not provided,
  defaults to `sqrt(nrow(embd))`.

- fc_params:

  Named list. See
  [`params_sc_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fast_cluster_gpu.md).

- snn:

  Boolean. Shall the kNN graph be additionally transformed into an sNN
  graph.

- return_kmeans:

  Boolean. Shall the k-means centroids and assignments be returned
  alongside the grid results.

- no_seeds:

  Integer. Number of additional seeds to use. Should be \>= 2.

- seed:

  Integer. For reproducibility.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list with the following elements:

- membership - A list with `memberships` (the labels from the seed with
  the best conductance, per resolution) and `stats` (the metrics per
  resolution).

- k_means_cluster - Optional integer vector of k-means assignments.

- centroids - Optional numeric matrix of k-means centroids.
