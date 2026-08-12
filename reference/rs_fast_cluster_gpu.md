# GPU: fast Louvain clustering on the data

**\[experimental\]** GPU equivalent of
[`bixverse::rs_fast_cluster_sc`](https://gregorlueg.github.io/bixverse/reference/rs_fast_cluster_sc.html).
Runs k-means clustering on the WGPU backend, followed by a kNN detection
on the centroids to then run Louvain clustering on the graph and
propagate the membership back to the original data. Everything after the
k-means stays on the CPU.

## Usage

``` r
rs_fast_cluster_gpu(
  embd,
  resolutions,
  n_centroids,
  fc_params,
  snn,
  return_kmeans,
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
  alongside the memberships.

- seed:

  Integer. For reproducibility.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list with the following elements:

- membership - The memberships across the different resolutions.

- k_means_cluster - Optional integer vector of k-means assignments.

- centroids - Optional numeric matrix of k-means centroids.
