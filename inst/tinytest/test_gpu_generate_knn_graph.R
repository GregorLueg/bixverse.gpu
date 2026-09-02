# test generate_knn_graph (manifoldsR-style embedding kNN) ---------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

n_neighbours <- 10L
n_samples <- 1000L

## synthetic data --------------------------------------------------------------

zeallot::`%<-%`(
  c(cluster_data, cluster_membership),
  manifoldsR::rs_data_clusters(
    n_samples = n_samples,
    dim = 32L,
    n_clusters = 15L,
    seed = 42L
  )
)

### cpu reference --------------------------------------------------------------

# from manifoldsR
exhaustive_cpu <- manifoldsR::generate_knn_graph(
  data = cluster_data,
  k = n_neighbours,
  knn_method = "exhaustive",
  nn_params = manifoldsR::params_nn(dist_metric = "euclidean"),
  .verbose = FALSE
)

## exhaustive: class, getters, ground truth ------------------------------------

exhaustive_gpu <- generate_knn_graph_gpu(
  data = cluster_data,
  k = n_neighbours,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(dist_metric = "euclidean"),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(exhaustive_gpu, "NearestNeighbours"),
  info = "exhaustive GPU returns a NearestNeighbours object"
)

expect_equal(
  current = dim(exhaustive_gpu),
  target = c(n_samples, n_neighbours),
  info = "dim primitive on the NearestNeighbours class works also here"
)

expect_true(
  current = checkmate::testMatrix(
    manifoldsR::get_idx_mat(exhaustive_gpu),
    mode = "integer",
    nrow = n_samples,
    ncol = n_neighbours
  ),
  info = "get_idx_mat behaves also here"
)

expect_equivalent(
  current = manifoldsR::get_idx_mat(exhaustive_gpu),
  target = manifoldsR::get_idx_mat(exhaustive_cpu),
  info = "get_idx_mat returns the same for CPU and GPU"
)

expect_true(
  current = checkmate::testInteger(
    manifoldsR::get_idx_flat(exhaustive_gpu),
    len = n_samples * n_neighbours
  ),
  info = "get_idx_flat behaves also here"
)

expect_equivalent(
  current = manifoldsR::get_idx_flat(exhaustive_gpu),
  target = manifoldsR::get_idx_flat(exhaustive_cpu),
  info = "get_idx_flat returns the same for CPU and GPU"
)

expect_true(
  current = checkmate::testMatrix(
    manifoldsR::get_dist_mat(exhaustive_gpu),
    mode = "numeric",
    nrow = n_samples,
    ncol = n_neighbours
  ),
  info = "get_dist_mat behaves"
)

expect_equivalent(
  current = manifoldsR::get_dist_mat(exhaustive_gpu),
  target = manifoldsR::get_dist_mat(exhaustive_cpu),
  info = "get_dist_mat returns the same for CPU and GPU",
  tolerance = 1e-6 # GPU / CPU delta
)

expect_true(
  current = checkmate::testNumeric(
    manifoldsR::get_dist_flat(exhaustive_gpu),
    len = n_samples * n_neighbours
  ),
  info = "get_dist_flat behaves also here"
)

expect_equivalent(
  current = manifoldsR::get_dist_flat(exhaustive_gpu),
  target = manifoldsR::get_dist_flat(exhaustive_cpu),
  info = "get_dist_flat returns the same for CPU and GPU",
  tolerance = 1e-6 # GPU / CPU delta
)

## 1-indexed indices -----------------------------------------------------------

exhaustive_idx <- manifoldsR::get_idx_flat(exhaustive_gpu)

expect_true(
  current = min(exhaustive_idx) >= 1L && max(exhaustive_idx) <= n_samples,
  info = "indices are 1-indexed and within [1, n]"
)

## approximate backends: class + recall vs exhaustive --------------------------

# IVF on n=1000 needs nlist and nprobe overrides. Defaults (sqrt(n)) are fine
# for larger data but recall dips on small sets. nndescent defaults are OK.
approx_configs <- list(
  ivf = params_nn_gpu(n_list = 3L, n_probes = 3L, dist_metric = "euclidean"),
  nndescent = params_nn_gpu(node_degree_final = 15L, dist_metric = "euclidean")
)

for (method in names(approx_configs)) {
  idx_i <- generate_knn_graph_gpu(
    data = cluster_data,
    k = n_neighbours,
    knn_method = method,
    nn_params = approx_configs[[method]],
    seed = 42L,
    .verbose = FALSE
  )

  expect_true(
    current = checkmate::testClass(idx_i, "NearestNeighbours"),
    info = sprintf("%s returns a NearestNeighbours object", method)
  )

  expect_equal(
    current = dim(idx_i),
    target = c(n_samples, n_neighbours),
    info = sprintf("%s returns the right shape", method)
  )

  recall <- sum(
    manifoldsR::get_idx_flat(idx_i) == manifoldsR::get_idx_flat(exhaustive_gpu)
  ) /
    (n_neighbours * n_samples)

  expect_true(
    current = recall > 0.98,
    info = sprintf(
      "%s has sensible recall vs exhaustive (%.2f)",
      method,
      recall
    )
  )
}

## nndescent with extract_knn = TRUE -------------------------------------------

# Skip beam search, pull kNN straight from the pruned NNDescent graph.
# Lower quality but must still return a valid object with the right shape.
cagra_direct <- generate_knn_graph_gpu(
  data = cluster_data,
  k = n_neighbours,
  knn_method = "nndescent",
  nn_params = params_nn_gpu(extract_knn = TRUE),
  seed = 42L,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(cagra_direct, "NearestNeighbours"),
  info = "nndescent with extract_knn = TRUE returns NearestNeighbours"
)

expect_equal(
  current = dim(cagra_direct),
  target = c(n_samples, n_neighbours),
  info = "nndescent extract_knn returns the right shape"
)

## cosine metric round-trip ---------------------------------------------------

# Sanity check that the cosine branch runs and returns the right shape.
cosine <- generate_knn_graph_gpu(
  data = cluster_data,
  k = n_neighbours,
  knn_method = "exhaustive",
  nn_params = params_nn_gpu(dist_metric = "cosine"),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(cosine, "NearestNeighbours"),
  info = "cosine metric returns a NearestNeighbours object"
)

expect_equal(
  current = dim(cosine),
  target = c(n_samples, n_neighbours),
  info = "cosine metric returns the right shape"
)

expect_true(
  current = checkmate::qtest(cosine$dist, "N[0, 1]"),
  info = "cosine distance in right range"
)
