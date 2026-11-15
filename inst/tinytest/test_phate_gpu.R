# phate gpu tests --------------------------------------------------------------

## params ----------------------------------------------------------------------

# no GPU needed: the defaults have to follow `PhateParams::default()` in
# manifolds-rs

phate_defaults <- params_phate_gpu()

expect_equal(
  current = phate_defaults,
  target = list(
    decay = 40,
    bandwidth_scale = 1,
    thresh = 1e-4,
    graph_symmetry = "additive",
    n_landmarks = 2048L,
    landmark_method = "spectral",
    n_svd = NULL,
    t_max = 100L,
    t_custom = NULL,
    gamma = 1,
    mds_method = "sgd_dense",
    mds_iter = NULL,
    randomised = TRUE
  ),
  info = "params_phate_gpu defaults match manifolds-rs"
)

expect_true(
  current = bixverse.gpu:::checkPhateGpuParams(params_phate_gpu(decay = NULL)),
  info = "params_phate_gpu - decay = NULL allowed"
)

expect_true(
  current = bixverse.gpu:::checkPhateGpuParams(params_phate_gpu(n_landmarks = NULL)),
  info = "params_phate_gpu - n_landmarks = NULL allowed"
)

expect_error(
  current = params_phate_gpu(gamma = 2),
  info = "params_phate_gpu - gamma outside [-1, 1]"
)

expect_error(
  current = params_phate_gpu(landmark_method = "spectral_mlx"),
  info = "params_phate_gpu - spectral_mlx goes through backend, not here"
)

expect_error(
  current = params_phate_gpu(n_landmarks = 0L),
  info = "params_phate_gpu - n_landmarks must be positive"
)

## gpu guards ------------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

# DX12 compiles shaders with FXC, which rejects barriers it cannot prove
# uniform. Hits kernels in ann-search-rs and bixverse-rs.
if (.Platform$OS.type == "windows") {
  exit_file("kernels do not compile under DX12 / FXC yet")
}

#' Helper function to check cluster separation
#'
#' Base R is unbearable slow, so, Rust...
check_cluster_separation <- function(embd, cluster_membership) {
  manifoldsR::rs_check_cluster_separation(
    embd = embd,
    cluster_membership = as.integer(cluster_membership)
  )
}

#' Helper function to check if clusters are separated
separates <- function(embd) {
  res <- check_cluster_separation(embd, cluster_membership)
  mean(res$within_dists) < mean(res$between_dists)
}

## synthetic data --------------------------------------------------------------

n_samples <- 200L

zeallot::`%<-%`(
  c(cluster_data, cluster_membership),
  manifoldsR::rs_data_clusters(
    n_samples = n_samples,
    dim = 32L,
    n_clusters = 3L,
    seed = 42L
  )
)

cluster_data_df <- as.data.frame(cluster_data)

# below n_samples, so the landmark path actually runs
landmark_params <- params_phate_gpu(n_landmarks = 50L, n_svd = 20L)

# tests ------------------------------------------------------------------------

## defaults --------------------------------------------------------------------

# 2048 landmarks >= 200 samples: the full operator
phate_res <- phate_gpu(data = cluster_data, .verbose = FALSE)

expect_true(
  current = checkmate::testMatrix(
    x = phate_res,
    mode = "numeric",
    any.missing = FALSE,
    ncols = 2L,
    nrows = n_samples
  ),
  info = "phate gpu result correctly returned"
)

expect_true(
  current = separates(phate_res),
  info = "phate gpu correctly separates clusters"
)

## reproducibility and input types ---------------------------------------------

# exhaustive kNN and random landmarks keep the device out of anything that is
# not bit-stable
phate_exact <- phate_gpu(
  data = cluster_data,
  knn_method = "exhaustive",
  phate_params = params_phate_gpu(n_landmarks = 50L, landmark_method = "random"),
  .verbose = FALSE
)

phate_exact_df <- phate_gpu(
  data = cluster_data_df,
  knn_method = "exhaustive",
  phate_params = params_phate_gpu(n_landmarks = 50L, landmark_method = "random"),
  .verbose = FALSE
)

expect_equal(
  current = phate_exact_df,
  target = phate_exact,
  info = "phate gpu df input matches matrix input"
)

## knn methods -----------------------------------------------------------------

for (method in c("nndescent", "exhaustive", "ivf")) {
  res <- phate_gpu(
    data = cluster_data,
    knn_method = method,
    phate_params = landmark_params,
    .verbose = FALSE
  )
  expect_true(
    current = separates(res),
    info = sprintf("phate gpu (%s) correctly separates clusters", method)
  )
}

## landmark methods ------------------------------------------------------------

for (method in c("spectral", "random", "density")) {
  res <- phate_gpu(
    data = cluster_data,
    phate_params = params_phate_gpu(
      n_landmarks = 50L,
      n_svd = 20L,
      landmark_method = method
    ),
    .verbose = FALSE
  )
  expect_true(
    current = checkmate::testMatrix(
      res,
      mode = "numeric",
      any.missing = FALSE,
      nrows = n_samples,
      ncols = 2L
    ) &&
      separates(res),
    info = sprintf("phate gpu (%s landmarks) correctly separates", method)
  )
}

## parameter variants ----------------------------------------------------------

variants <- list(
  full_operator = params_phate_gpu(n_landmarks = NULL),
  fixed_t = params_phate_gpu(n_landmarks = 50L, t_custom = 10L),
  classic_mds = params_phate_gpu(n_landmarks = 50L, mds_method = "classic"),
  binary_kernel = params_phate_gpu(n_landmarks = 50L, decay = NULL),
  multiplicative = params_phate_gpu(
    n_landmarks = 50L,
    graph_symmetry = "multiplicative"
  )
)

for (nm in names(variants)) {
  res <- phate_gpu(
    data = cluster_data,
    phate_params = variants[[nm]],
    .verbose = FALSE
  )
  expect_true(
    current = separates(res),
    info = sprintf("phate gpu (%s) correctly separates clusters", nm)
  )
}

# a binary kernel has to differ from the alpha-decay one: if `decay = NULL`
# fell back to 40 on the Rust side, these would be identical
expect_false(
  current = isTRUE(all.equal(
    phate_gpu(
      data = cluster_data,
      knn_method = "exhaustive",
      phate_params = params_phate_gpu(
        n_landmarks = 50L,
        landmark_method = "random",
        decay = NULL
      ),
      .verbose = FALSE
    ),
    phate_exact
  )),
  info = "phate gpu - decay = NULL reaches Rust as the binary kernel"
)

phate_cosine <- phate_gpu(
  data = cluster_data,
  nn_params = params_nn_gpu(dist_metric = "cosine"),
  phate_params = landmark_params,
  .verbose = FALSE
)

expect_true(
  current = separates(phate_cosine),
  info = "phate gpu (cosine) correctly separates clusters"
)

phate_3d <- phate_gpu(
  data = cluster_data,
  n_dim = 3L,
  phate_params = landmark_params,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    phate_3d,
    mode = "numeric",
    any.missing = FALSE,
    nrows = n_samples,
    ncols = 3L
  ),
  info = "phate gpu returns three dimensions"
)

## precomputed knn -------------------------------------------------------------

# 6 * 5 + 5 = 35 wide, exhaustive: the same graph the internal search builds
knn_wide <- generate_knn_graph_gpu(
  data = cluster_data,
  k = 35L,
  knn_method = "exhaustive",
  .verbose = FALSE
)

phate_from_knn <- phate_gpu(
  data = cluster_data,
  knn = knn_wide,
  knn_method = "exhaustive",
  phate_params = params_phate_gpu(n_landmarks = 50L, landmark_method = "random"),
  .verbose = FALSE
)

expect_equal(
  current = phate_from_knn,
  target = phate_exact,
  tolerance = 1e-5,
  info = "phate gpu - precomputed kNN of the search width matches the search"
)

knn_narrow <- generate_knn_graph_gpu(
  data = cluster_data,
  k = 15L,
  knn_method = "exhaustive",
  .verbose = FALSE
)

expect_warning(
  current = phate_gpu(
    data = cluster_data,
    knn = knn_narrow,
    phate_params = landmark_params,
    .verbose = FALSE
  ),
  pattern = "35 wide",
  info = "phate gpu - narrow precomputed kNN warns"
)

## mlx -------------------------------------------------------------------------

if (mlx_available()) {
  phate_mlx <- phate_gpu(
    data = cluster_data,
    phate_params = landmark_params,
    backend = "mlx",
    .verbose = FALSE
  )

  expect_true(
    current = separates(phate_mlx),
    info = "phate gpu (mlx) correctly separates clusters"
  )
}
