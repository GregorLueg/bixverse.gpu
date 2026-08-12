# Default parameters for GPU fast Louvain clustering

GPU counterpart to
[`bixverse::params_sc_fast_cluster()`](https://gregorlueg.github.io/bixverse/reference/params_sc_fast_cluster.html).
The mini-batch k-means knobs are gone (the GPU k-means is full-batch
Lloyd's) and the k-means block comes from the GPU parameters instead.
Two knobs the CPU wrapper never exposed, `same_weight` and
`multi_level_louvain`, are available here.

The k-means distance is taken from `knn$ann_dist`, so the coarsening and
the centroid graph agree on the geometry. There is no separate `metric`
argument, and `"manhattan"` is not supported by the GPU k-means.

## Usage

``` r
params_sc_fast_cluster_gpu(
  k_means_iter = 50L,
  k_means_init = NULL,
  fixed = TRUE,
  quantise = FALSE,
  same_weight = FALSE,
  full_snn = FALSE,
  pruning = NULL,
  snn_similarity = c("jaccard", "rank"),
  louvain_iters = 10L,
  multi_level_louvain = TRUE,
  knn = list(k = 5L)
)
```

## Arguments

- k_means_iter:

  Integer. Maximum number of k-means iterations.

- k_means_init:

  Optional character. Initialisation method. One of `"random"`,
  `"parallel"` or `"plusplus"`. If `NULL`, picked on the Rust side based
  on the number of centroids.

- fixed:

  Boolean. Shall k-means run for a fixed number of iterations, without
  checking for convergence.

- quantise:

  Boolean. Shall the data buffer be held at fp16 on the GPU. Halves the
  buffer and helps when the assignment kernels are memory bound.

- same_weight:

  Boolean. If `TRUE`, all kNN edges get weight `1.0`. Otherwise edges
  with a reverse counterpart are double counted.

- full_snn:

  Boolean. Shall the full shared nearest neighbour graph be generated,
  including edges between centroids that are not neighbours.

- pruning:

  Optional numeric. Weights below this threshold are set to 0 when
  generating the sNN graph. If `NULL`, defaults to
  `1 / ceiling(k * 0.8)`.

- snn_similarity:

  String. One of `c("jaccard", "rank")`. Jaccard computes the Jaccard
  index between neighbour sets; rank weights edges by the best combined
  rank of a shared neighbour. Both are normalised to `[0, 1]`.

- louvain_iters:

  Integer. Number of Louvain iterations.

- multi_level_louvain:

  Boolean. Shall multi-level Louvain be applied.

- knn:

  List. Optional overrides for the kNN parameters applied to the
  centroids. See
  [`bixverse::params_knn_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_knn_defaults.html)
  for the available parameters. Defaults to `k = 5L`.

## Value

A named list with the GPU fast clustering parameters.
