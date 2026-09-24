# Default parameters for Harmony v2 GPU batch correction

Default parameters for Harmony v2 GPU batch correction

## Usage

``` r
params_sc_harmony_v2_gpu(
  k = NULL,
  sigma = 0.1,
  theta = 2,
  lambda = 1,
  max_iter_kmeans = 4L,
  max_iter_harmony = 10L,
  epsilon_kmeans = 0.001,
  epsilon_harmony = 0.01,
  window_size = 3L,
  alpha = 0.2,
  tau = 0,
  batch_proportion_cutoff = 1e-05,
  use_dynamic_lambda = FALSE,
  csr_cube_count = 256L,
  k_means_iter = 30L,
  k_means_init = NULL,
  fixed = FALSE,
  quantise = FALSE
)
```

## Arguments

- k:

  Integer or `NULL`. Number of clusters for k-means clustering. If not
  provided, it will be automatically determined as
  `min(round(N / 30), 100)`. Defaults to `NULL`.

- sigma:

  Numeric vector. Per-cluster diversity weights. Either a single value
  (broadcast to all clusters) or a vector of length k. Defaults to
  `0.1`.

- theta:

  Numeric vector. Per-variable diversity penalty. Must be a single
  value; only one batch covariate is supported on the GPU path. Defaults
  to `2.0`.

- lambda:

  Numeric vector. Ridge regression penalty for the linear model.
  Typically a single value. Ignored when `use_dynamic_lambda = TRUE`.
  Defaults to `1.0`.

- max_iter_kmeans:

  Integer. Maximum number of k-means Jacobi sweeps per Harmony round.
  Defaults to `4L`.

- max_iter_harmony:

  Integer. Maximum number of Harmony outer iterations. Defaults to
  `10L`.

- epsilon_kmeans:

  Numeric. Convergence threshold for k-means clustering. Defaults to
  `0.001`.

- epsilon_harmony:

  Numeric. Convergence threshold for Harmony. Defaults to `0.01`.

- window_size:

  Integer. Number of previous iterations to consider when checking
  convergence. Defaults to `3L`.

- alpha:

  Numeric. Scaling factor for dynamic lambda estimation. Must be in (0,
  1). Only relevant when `use_dynamic_lambda = TRUE`. Defaults to `0.2`.

- tau:

  Numeric. Scaling factor for theta based on batch size. A value of 0
  disables batch-size scaling of theta. Defaults to `0.0`.

- batch_proportion_cutoff:

  Numeric. Cutoff for pruning batches with small proportions during
  ridge regression. Defaults to `1e-05`.

- use_dynamic_lambda:

  Boolean. If `TRUE`, lambda is estimated dynamically per cluster
  instead of using the fixed `lambda` value. Defaults to `FALSE`.

- csr_cube_count:

  Integer. Number of parallel thread groups used when building the
  level-CSR index on the GPU. Adjust for your hardware if needed.
  Defaults to `256L`.

- k_means_iter:

  Integer. Maximum number of k-means iterations for the initial centroid
  computation. Defaults to `30L`.

- k_means_init:

  String or `NULL`. Initialisation strategy for k-means. Defaults to
  `NULL`.

- fixed:

  Boolean. If `TRUE`, centroids are fixed after initialisation. Defaults
  to `FALSE`.

- quantise:

  Boolean. If `TRUE`, quantises intermediate values to f16 during
  k-means. Defaults to `FALSE`.

## Value

A named list with the following elements:

- k - Integer or `NULL`. Number of clusters for k-means clustering. If
  not provided, it will be automatically determined as
  `min(round(N / 30), 100)`. Defaults to `NULL`.

- sigma - Numeric vector. Per-cluster diversity weights. Either a single
  value (broadcast to all clusters) or a vector of length k. Defaults to
  `0.1`.

- theta - Numeric vector. Per-variable diversity penalty. Must be a
  single value; only one batch covariate is supported on the GPU path.
  Defaults to `2.0`.

- lambda - Numeric vector. Ridge regression penalty for the linear
  model. Typically a single value. Ignored when
  `use_dynamic_lambda = TRUE`. Defaults to `1.0`.

- max_iter_kmeans - Integer. Maximum number of k-means Jacobi sweeps per
  Harmony round. Defaults to `4L`.

- max_iter_harmony - Integer. Maximum number of Harmony outer
  iterations. Defaults to `10L`.

- epsilon_kmeans - Numeric. Convergence threshold for k-means
  clustering. Defaults to `0.001`.

- epsilon_harmony - Numeric. Convergence threshold for Harmony. Defaults
  to `0.01`.

- window_size - Integer. Number of previous iterations to consider when
  checking convergence. Defaults to `3L`.

- alpha - Numeric. Scaling factor for dynamic lambda estimation. Must be
  in (0, 1). Only relevant when `use_dynamic_lambda = TRUE`. Defaults to
  `0.2`.

- tau - Numeric. Scaling factor for theta based on batch size. A value
  of 0 disables batch-size scaling of theta. Defaults to `0.0`.

- batch_proportion_cutoff - Numeric. Cutoff for pruning batches with
  small proportions during ridge regression. Defaults to `1e-05`.

- use_dynamic_lambda - Boolean. If `TRUE`, lambda is estimated
  dynamically per cluster instead of using the fixed `lambda` value.
  Defaults to `FALSE`.

- csr_cube_count - Integer. Number of parallel thread groups used when
  building the level-CSR index on the GPU. Adjust for your hardware if
  needed. Defaults to `256L`.

- k_means_iter - Integer. Maximum number of k-means iterations for the
  initial centroid computation. Defaults to `30L`.

- k_means_init - String or `NULL`. Initialisation strategy for k-means.
  Defaults to `NULL`.

- fixed - Boolean. If `TRUE`, centroids are fixed after initialisation.
  Defaults to `FALSE`.

- quantise - Boolean. If `TRUE`, quantises intermediate values to f16
  during k-means. Defaults to `FALSE`.
