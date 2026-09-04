# tests of gpu knn searches ----------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

if (!requireNamespace("BiocNeighbors")) {
  exit_file("BiocNeighbors is not available")
}

## synthetic data --------------------------------------------------------------

set.seed(42L)

nrow <- 1000
ncol <- 30
k <- 10L

data <- matrix(rnorm(nrow * ncol), ncol = ncol)

## tests -----------------------------------------------------------------------

### bioc results ---------------------------------------------------------------

bioc_knn <- BiocNeighbors::findKNN(
  X = data,
  k = k
)

## gpu searches ----------------------------------------------------------------

calc_recall_bioc <- function(knn_mat, rs_knn_mat) {
  sum(knn_mat == (rs_knn_mat + 1)) / prod(dim(knn_mat))
}

# no sqrt: since manifolds-rs 0.4.0 the euclidean backends root their own
# squared output before handing it back
calc_dist_bioc <- function(knn_dist, rs_knn_dist) {
  sum(abs(knn_dist - rs_knn_dist)) / prod(dim(knn_dist))
}

### exhaustive gpu -------------------------------------------------------------

gpu_exhaustive_res <- rs_gpu_knn(
  embd = data,
  k = k,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(dist_metric = "euclidean"),
  seed = 42L,
  verbose = 0L
)

recall_exhaustive_gpu <- calc_recall_bioc(
  knn_mat = bioc_knn$index,
  rs_knn_mat = gpu_exhaustive_res$indices
)

dist_diff_exhaustive_gpu <- calc_dist_bioc(
  knn_dist = bioc_knn$distance,
  gpu_exhaustive_res$dist
)

expect_true(
  current = recall_exhaustive_gpu >= 0.98,
  info = "gpu exhaustive index - recall"
)

expect_true(
  current = dist_diff_exhaustive_gpu <= 1e-6,
  info = "gpu exhaustive index - distance"
)

### ivf gpu --------------------------------------------------------------------

gpu_ivf_res <- rs_gpu_knn(
  embd = data,
  k = k,
  knn_method = "ivf",
  nn_params = params_nn_gpu(
    dist_metric = "euclidean",
    # on small data sets IVF basically does not behave...
    n_list = 3L,
    n_probes = 3L
  ),
  seed = 42L,
  verbose = 0L
)

recall_ivf_gpu <- calc_recall_bioc(
  knn_mat = bioc_knn$index,
  rs_knn_mat = gpu_ivf_res$indices
)

dist_diff_ivf_gpu <- calc_dist_bioc(
  knn_dist = bioc_knn$distance,
  gpu_ivf_res$dist
)

expect_true(
  current = recall_ivf_gpu >= 0.98,
  info = "gpu ivf index - recall"
)

expect_true(
  current = dist_diff_ivf_gpu <= 1e-6,
  info = "gpu ivf index - distance"
)

### cagra ----------------------------------------------------------------------

gpu_cagra <- rs_gpu_knn(
  embd = data,
  k = k,
  knn_method = "nndescent",
  nn_params = params_nn_gpu(
    dist_metric = "euclidean",
    extract_knn = FALSE
  ),
  seed = 42L,
  verbose = 0L
)

recall_cagra_gpu <- calc_recall_bioc(
  knn_mat = bioc_knn$index,
  rs_knn_mat = gpu_cagra$indices
)

dist_diff_cagra_gpu <- calc_dist_bioc(
  knn_dist = bioc_knn$distance,
  gpu_cagra$dist
)

expect_true(
  current = recall_cagra_gpu >= 0.98,
  info = "gpu cagra index - recall"
)

# performance worse on small data sets, but captures the overall structure
expect_true(
  current = dist_diff_cagra_gpu <= 1e-3,
  info = "gpu cagra index - distance"
)
