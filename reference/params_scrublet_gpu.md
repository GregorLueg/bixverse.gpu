# Wrapper function for GPU Scrublet doublet detection parameters

GPU counterpart to
[`bixverse::params_scrublet()`](https://gregorlueg.github.io/bixverse/reference/params_scrublet.html).
Two differences from the CPU list. The `pca` sub-list is gone: the GPU
SVD is always randomised, so `random_svd` and `sparse` have nothing to
switch and `no_pcs` is a plain argument. And the kNN block is backend
dependent, see `knn_backend`.

## Usage

``` r
params_scrublet_gpu(
  sim_doublet_ratio = 1.5,
  expected_doublet_rate = 0.1,
  stdev_doublet_rate = 0.02,
  n_bins_histogram = 100L,
  manual_threshold = NULL,
  no_pcs = 30L,
  normalisation = list(),
  hvg = list(),
  knn_backend = c("gpu", "cpu"),
  knn = list(k = 0L)
)
```

## Arguments

- sim_doublet_ratio:

  Numeric. Number of doublets to simulate relative to the number of
  observed cells. Defaults to `1.5`.

- expected_doublet_rate:

  Numeric in `[0, 1]`. Expected doublet rate, typically 0.05-0.10
  depending on cell loading. Defaults to `0.1`.

- stdev_doublet_rate:

  Numeric in `[0, 1]`. Uncertainty in the expected doublet rate.
  Defaults to `0.02`.

- n_bins_histogram:

  Integer. Histogram bins for the Otsu threshold search. Defaults to
  `100L`.

- manual_threshold:

  Optional numeric. Fixed doublet score threshold. If `NULL`, Otsu's
  method picks it.

- no_pcs:

  Integer. Number of principal components. Defaults to `30L`.

- normalisation:

  List. Optional overrides. See
  [`bixverse::params_norm_doublets_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_norm_doublets_defaults.html)
  for the available parameters: `log_transform`, `mean_center`,
  `normalise_variance`, `target_size`.

- hvg:

  List. Optional overrides. See
  [`bixverse::params_hvg_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_hvg_defaults.html)
  for the available parameters: `min_gene_var_pctl`, `hvg_method`,
  `loess_span`, `clip_max`, `n_bins`, `binning_strategy`.

- knn_backend:

  String. One of `"gpu"` or `"cpu"`. Picks which nearest neighbour index
  runs over the combined observed-plus-simulated embedding, and with it
  which keys `knn` accepts. `"gpu"` is the fast default; `"cpu"` buys
  the exact CPU indices at the cost of a host round trip on a matrix
  that is `(1 + sim_doublet_ratio) * n_cells` rows tall.

- knn:

  List. Optional overrides for the kNN block. Validated against
  [`params_knn_gpu_defaults()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_knn_gpu_defaults.md)
  when `knn_backend = "gpu"` and against
  [`bixverse::params_knn_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_knn_defaults.html)
  when `knn_backend = "cpu"`. Unknown keys are an error, not a silent
  pass-through. Defaults to `list(k = 0L)`, which asks Rust to pick `k`.

## Value

A flat named list with all GPU Scrublet parameters.

## Details

Leave `knn_method` alone unless you have a reason. `"exhaustive"` is the
default and is the right answer for Scrublet on the GPU arm.

Scrublet queries at a high `k` by construction: the count is taken over
an embedding `(1 + sim_doublet_ratio) * n_cells` rows tall, and `k = 0L`
then scales `k` by the same factor, so a 20k-cell run searches at `k`
around 175. Exhaustive barely notices `k`, since the scan is the cost
and `k` only sizes the top-k selection. NN-descent notices a lot: its
build degree tracks `k`, so the descent does more work per node as `k`
climbs. Measured at 20k cells and 30 PCs, exhaustive took 0.98s against
38.8s for NN-descent at `k = 200`. NN-descent only came out ahead at
`k = 10`.

`"ivf"` is the one worth trying: it was the quickest of the three across
that sweep and holds a Pearson above 0.99 against exhaustive.

Recall matters more here than elsewhere. The doublet score is a
neighbour count, so a backend that drops neighbours biases every score
downwards.

## References

Wolock, et al., Cell Syst, 2020
