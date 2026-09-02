# tsne gpu tests ---------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
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

# tests ------------------------------------------------------------------------

## tsne gpu general ------------------------------------------------------------

### general wrapper (default: BH + nndescent) ----------------------------------

tsne_gpu_res <- tsne_gpu(
  data = cluster_data,
  perplexity = 15,
  .verbose = FALSE
)

tsne_gpu_res_tests <- check_cluster_separation(
  embd = tsne_gpu_res,
  cluster_membership = cluster_membership
)

expect_true(
  current = checkmate::testMatrix(
    x = tsne_gpu_res,
    mode = "numeric",
    ncols = 2L,
    nrow = n_samples
  ),
  info = "tsne gpu result correctly returned"
)

expect_true(
  current = mean(tsne_gpu_res_tests$within_dists) <
    mean(tsne_gpu_res_tests$between_dists),
  info = "tsne gpu correctly separates clusters"
)

tsne_gpu_res_from_df <- tsne_gpu(
  data = cluster_data_df,
  perplexity = 15,
  .verbose = FALSE
)

expect_equal(
  current = tsne_gpu_res,
  target = tsne_gpu_res_from_df,
  info = "tsne gpu df input matches matrix input"
)

### exhaustive knn method ------------------------------------------------------

tsne_gpu_exhaustive <- tsne_gpu(
  data = cluster_data,
  perplexity = 15,
  knn_method = "exhaustive",
  .verbose = FALSE
)

tsne_gpu_exhaustive_tests <- check_cluster_separation(
  embd = tsne_gpu_exhaustive,
  cluster_membership = cluster_membership
)

expect_true(
  current = mean(tsne_gpu_exhaustive_tests$within_dists) <
    mean(tsne_gpu_exhaustive_tests$between_dists),
  info = "tsne gpu (exhaustive kNN) correctly separates clusters"
)

### pre-computed knn -----------------------------------------------------------

precomputed_knn <- generate_knn_graph_gpu(
  data = cluster_data,
  k = 45L, # ~ 3 * perplexity
  knn_method = "exhaustive",
  .verbose = FALSE
)

tsne_gpu_from_knn <- tsne_gpu(
  data = cluster_data,
  knn = precomputed_knn,
  perplexity = 15,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    x = tsne_gpu_from_knn,
    mode = "numeric",
    ncols = 2L,
    nrow = n_samples
  ),
  info = "tsne gpu with pre-computed kNN returns the expected matrix"
)

tsne_gpu_from_knn_tests <- check_cluster_separation(
  embd = tsne_gpu_from_knn,
  cluster_membership = cluster_membership
)

expect_true(
  current = mean(tsne_gpu_from_knn_tests$within_dists) <
    mean(tsne_gpu_from_knn_tests$between_dists),
  info = "tsne gpu from pre-computed kNN correctly separates clusters"
)

## .prepare_tsne_params_gpu ----------------------------------------------------

### parameter composition ------------------------------------------------------

prep_composed <- .prepare_tsne_params_gpu(
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(dist_metric = "euclidean", node_degree_final = 20L),
  tsne_params = params_tsne_gpu(theta = 0.3, init = "spectral")
)

expect_equal(
  current = prep_composed$knn_method,
  target = "exhaustive",
  info = "knn_method is set correctly in the merged params"
)

expect_equal(
  current = prep_composed$dist_metric,
  target = "euclidean",
  info = "nn_params fields are present in the merged params"
)

expect_equal(
  current = prep_composed$node_degree_final,
  target = 20L,
  info = "nn_params overrides are present in the merged params"
)

expect_equal(
  current = prep_composed$theta,
  target = 0.3,
  info = "tsne_params fields are present in the merged params"
)

expect_equal(
  current = prep_composed$init,
  target = "spectral",
  info = "tsne_params overrides are present in the merged params"
)

### defaults -------------------------------------------------------------------

prep_defaults <- .prepare_tsne_params_gpu(
  knn_method = "nndescent",
  nn_params = params_nn_gpu(),
  tsne_params = params_tsne_gpu()
)

expect_equal(
  current = prep_defaults$n_epochs,
  target = 1000L,
  info = "default n_epochs is 1000"
)

expect_equal(
  current = prep_defaults$early_exag_iter,
  target = 250L,
  info = "default early_exag_iter is 250"
)

expect_equal(
  current = prep_defaults$init,
  target = "pca",
  info = "default init is pca"
)

expect_true(
  current = prep_defaults$randomised,
  info = "default randomised is TRUE"
)

expect_null(
  current = prep_defaults$lr,
  info = "default lr is NULL (resolved on the Rust side)"
)

expect_null(
  current = prep_defaults$late_exag_factor,
  info = "default late_exag_factor is NULL"
)
