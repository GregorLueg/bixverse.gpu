# umap gpu tests ---------------------------------------------------------------

## synthetic data --------------------------------------------------------------

n_samples <- 200L

zeallot::`%<-%`(
  c(cluster_data, cluster_membership),
  rs_data_clusters(
    n_samples = n_samples,
    dim = 32L,
    n_clusters = 3L,
    seed = 42L
  )
)

cluster_data_df <- as.data.frame(cluster_data)

# tests ------------------------------------------------------------------------

## umap gpu general ------------------------------------------------------------

### general wrapper (default optimiser: adam_gpu) ------------------------------

umap_gpu_res <- umap_gpu(data = cluster_data, k = 5L, .verbose = FALSE)

umap_gpu_res_tests <- check_cluster_separation(
  embd = umap_gpu_res,
  cluster_membership = cluster_membership
)

expect_true(
  current = checkmate::testMatrix(
    x = umap_gpu_res,
    mode = "numeric",
    ncols = 2L,
    nrow = n_samples
  ),
  info = "umap gpu result correctly returned"
)

expect_true(
  current = mean(umap_gpu_res_tests$within_dists) <
    mean(umap_gpu_res_tests$between_dists),
  info = "umap gpu correctly separates clusters"
)

umap_gpu_res_from_df <- umap_gpu(
  data = cluster_data_df,
  k = 5L,
  .verbose = FALSE
)

expect_equal(
  current = umap_gpu_res,
  target = umap_gpu_res_from_df,
  info = "umap gpu df is the same as umap gpu matrix return results"
)

## .prepare_umap_params_gpu ----------------------------------------------------

### n_epochs resolution --------------------------------------------------------

prep_adam_gpu_large <- .prepare_umap_params_gpu(
  n = 20000L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(optimiser = "adam_gpu"),
  .verbose = FALSE
)

expect_equal(
  current = prep_adam_gpu_large$n_epochs,
  target = 500L,
  info = "adam_gpu always gets n_epochs = 500"
)

prep_adam_parallel_large <- .prepare_umap_params_gpu(
  n = 20000L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(optimiser = "adam_parallel"),
  .verbose = FALSE
)

expect_equal(
  current = prep_adam_parallel_large$n_epochs,
  target = 500L,
  info = "adam_parallel always gets n_epochs = 500"
)

prep_sgd_small <- .prepare_umap_params_gpu(
  n = 5000L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = "nndescent",
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(optimiser = "sgd"),
  .verbose = FALSE
)

expect_equal(
  current = prep_sgd_small$n_epochs,
  target = 500L,
  info = "sgd with n <= 10000 gets n_epochs = 500"
)

prep_sgd_large <- .prepare_umap_params_gpu(
  n = 20000L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = "ivf",
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(optimiser = "sgd"),
  .verbose = FALSE
)

expect_equal(
  current = prep_sgd_large$n_epochs,
  target = 200L,
  info = "sgd with n > 10000 gets n_epochs = 200"
)

prep_adam_large <- .prepare_umap_params_gpu(
  n = 20000L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = "nndescent",
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(optimiser = "adam"),
  .verbose = FALSE
)

expect_equal(
  current = prep_adam_large$n_epochs,
  target = 200L,
  info = "adam with n > 10000 gets n_epochs = 200"
)

prep_user_epochs <- .prepare_umap_params_gpu(
  n = 20000L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(n_epochs = 750L),
  .verbose = FALSE
)

expect_equal(
  current = prep_user_epochs$n_epochs,
  target = 750L,
  info = "user-defined n_epochs is respected"
)

expect_true(
  current = is.integer(prep_user_epochs$n_epochs),
  info = "user-defined n_epochs is coerced to integer"
)

### parameter composition ------------------------------------------------------

prep_composed <- .prepare_umap_params_gpu(
  n = 100L,
  min_dist = 0.2,
  spread = 1.5,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(dist_metric = "euclidean", node_degree_final = 20L),
  umap_params = params_umap_gpu(lr = 0.5, init = "pca"),
  .verbose = FALSE
)

expect_equal(
  current = prep_composed$min_dist,
  target = 0.2,
  info = "min_dist is overwritten correctly in final params"
)

expect_equal(
  current = prep_composed$spread,
  target = 1.5,
  info = "spread is overwritten correctly in final params"
)

expect_equal(
  current = prep_composed$knn_method,
  target = "exhaustive",
  info = "nn_method is set as knn_method in final params"
)

expect_equal(
  current = prep_composed$dist_metric,
  target = "euclidean",
  info = "nn_params fields are present in final params"
)

expect_equal(
  current = prep_composed$node_degree_final,
  target = 20L,
  info = "nn_params overrides are present in final params"
)

expect_equal(
  current = prep_composed$lr,
  target = 0.5,
  info = "umap_params fields are present in final params"
)

expect_equal(
  current = prep_composed$init,
  target = "pca",
  info = "umap_params overrides are present in final params"
)
