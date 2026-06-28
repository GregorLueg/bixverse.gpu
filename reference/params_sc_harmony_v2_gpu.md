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

  Optional integer. Number of clusters for k-means clustering. If not
  provided, it will be automatically determined as
  `min(round(N / 30), 100)`.

- sigma:

  Numeric vector. Per-cluster diversity weights. Either a single value
  (broadcast to all clusters) or a vector of length k.

- theta:

  Numeric vector. Per-variable diversity penalty. Must be a single
  value; only one batch covariate is supported on the GPU path.

- lambda:

  Numeric vector. Ridge regression penalty for the linear model.
  Typically a single value. Ignored when `use_dynamic_lambda = TRUE`.

- max_iter_kmeans:

  Integer. Maximum number of k-means Jacobi sweeps per Harmony round.

- max_iter_harmony:

  Integer. Maximum number of Harmony outer iterations.

- epsilon_kmeans:

  Numeric. Convergence threshold for k-means clustering.

- epsilon_harmony:

  Numeric. Convergence threshold for Harmony.

- window_size:

  Integer. Number of previous iterations to consider when checking
  convergence.

- alpha:

  Numeric. Scaling factor for dynamic lambda estimation. Must be in (0,
  1). Only relevant when `use_dynamic_lambda = TRUE`.

- tau:

  Numeric. Scaling factor for theta based on batch size. A value of 0
  disables batch-size scaling of theta.

- batch_proportion_cutoff:

  Numeric. Cutoff for pruning batches with small proportions during
  ridge regression.

- use_dynamic_lambda:

  Boolean. If `TRUE`, lambda is estimated dynamically per cluster
  instead of using the fixed `lambda` value.

- csr_cube_count:

  Integer. Number of parallel thread groups used when building the
  level-CSR index on the GPU. Adjust for your hardware if needed.

- k_means_iter:

  Integer. Maximum number of k-means iterations for the initial centroid
  computation.

- k_means_init:

  Optional string. Initialisation strategy for k-means.

- fixed:

  Boolean. If `TRUE`, centroids are fixed after initialisation.

- quantise:

  Boolean. If `TRUE`, quantises intermediate values to f16 during
  k-means.

## Value

A list with the parameters.
