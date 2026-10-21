# GPU: fast clustering on the data

**\[experimental\]** GPU equivalent of
[`bixverse::rs_fast_cluster_sc`](https://gregorlueg.github.io/bixverse/reference/rs_fast_cluster_sc.html).
Coarsens the cells with k-means on the GPU, builds a graph over the
centroids (a kNN graph with one of the symmetrisations, or the
cell-mediated graph) and runs Louvain or Leiden on it. The community of
each centroid is propagated back to its cells. With `n_seeds > 1` every
resolution is run once per seed, the labels come from the seed with the
lowest mean conductance and stability metrics across the seeds are
returned. Everything after stage one stays on the CPU.

## Usage

``` r
rs_fast_cluster_gpu(
  embd,
  resolutions,
  fc_params,
  n_seeds,
  seed,
  verbose,
  backend = "wgpu"
)
```

## Arguments

- embd:

  Numeric matrix. The embedding, cells x features.

- resolutions:

  Numeric vector. The resolutions to iterate through.

- fc_params:

  Named list. See
  [`params_sc_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fast_cluster_gpu.md).
  Carries the GPU k-means keys (`k_means_iter`, `k_means_init`, `fixed`,
  `quantise`) alongside the fast clustering ones.

- n_seeds:

  Integer. Number of seeds per resolution. `1L` runs a single pass
  without stability metrics.

- seed:

  Integer. For reproducibility.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

- backend:

  String. `"wgpu"` or `"mlx"` (Apple Silicon, needs a build with the MLX
  backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).
  MLX ignores `fixed` and `quantise`.

## Value

A list with the following elements:

- membership - List with one integer membership vector per resolution
  (0-indexed).

- k_means_cluster - Integer vector with the k-means centroid per cell
  (0-indexed).

- centroids - Numerical matrix with the k-means centroids.

- stats - `NULL` for `n_seeds = 1L`, otherwise a list with `mean_ari`,
  `median_ari`, `mean_conductance`, `median_conductance` and
  `mean_n_comms`, one value per resolution.
