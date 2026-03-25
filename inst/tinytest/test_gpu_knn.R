# tests of gpu knn searches ----------------------------------------------------

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

calc_dist_bioc <- function(knn_dist, rs_knn_dist) {
  sum(abs(
    knn_dist - sqrt(rs_knn_dist)
  )) /
    prod(dim(knn_dist))
}

### exhaustive gpu -------------------------------------------------------------

gpu_exhaustive_res <- rs_exhaustive_gpu_knn(
  embd = data,
  k = k,
  dist_metric = "euclidean",
  verbose = FALSE
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

gpu_ivf_res <- rs_ivf_gpu_knn(
  embd = data,
  ivf_params = params_sc_ivf(
    k = k,
    ann_dist = "euclidean",
    # on small data sets IVF basically does not behave...
    nlist = 3L,
    nprobe = 3L
  ),
  seed = 42L,
  verbose = FALSE
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

gpu_cagra <- rs_cagra_gpu_knn(
  embd = data,
  cagra_params = params_sc_cagra(k_query = k, ann_dist = "euclidean"),
  extract_knn = FALSE,
  seed = 42L,
  verbose = FALSE
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
