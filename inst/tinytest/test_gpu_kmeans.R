# k means gpu tests ------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

## synthetic data --------------------------------------------------------------

n_samples <- 200L
n_clusters <- 3L

zeallot::`%<-%`(
  c(cluster_data, cluster_membership),
  manifoldsR::rs_data_clusters(
    n_samples = n_samples,
    dim = 32L,
    n_clusters = n_clusters,
    seed = 42L
  )
)

## gpu k-means -----------------------------------------------------------------

res_gpu <- k_means_cluster_gpu(
  data = cluster_data,
  k = n_clusters,
  kmeans_params = params_kmeans_gpu(fixed = FALSE),
  seed = 42L,
  .verbose = FALSE
)

expect_true(
  current = inherits(res_gpu, "KMeansClusterGPU"),
  info = "gpu kmeans returns KMeansClusterGPU object"
)
expect_true(
  current = checkmate::testList(x = res_gpu, len = 4),
  info = "gpu kmeans returns expected list length"
)
expect_true(
  current = checkmate::testNames(
    x = names(res_gpu),
    must.include = c("centroids", "assignments", "k", "metric")
  ),
  info = "gpu kmeans returns expected names"
)
expect_true(
  current = checkmate::testMatrix(
    x = res_gpu$centroids,
    mode = "numeric",
    nrows = n_clusters,
    ncols = 32L
  ),
  info = "gpu kmeans centroids have correct dimensions"
)
expect_true(
  current = checkmate::qtest(
    x = res_gpu$assignments,
    sprintf("I%d[1,%d]", n_samples, n_clusters)
  ),
  info = "gpu kmeans assignments are valid 1-indexed indices"
)
expect_equal(
  current = length(unique(res_gpu$assignments)),
  target = n_clusters,
  info = "gpu kmeans uses all clusters"
)

ari_gpu <- manifoldsR::calc_ari(
  as.integer(cluster_membership),
  res_gpu$assignments
)
expect_true(
  current = ari_gpu > 0.9,
  info = "gpu kmeans recovers known clusters"
)

## getters ---------------------------------------------------------------------

expect_equal(
  current = manifoldsR::membership(res_gpu),
  target = res_gpu$assignments,
  info = "membership() returns assignments"
)

expect_equal(
  current = manifoldsR::get_centroids(res_gpu),
  target = res_gpu$centroids,
  info = "get_centroids() returns centroids"
)

## reproducibility -------------------------------------------------------------

res_a <- k_means_cluster_gpu(
  data = cluster_data,
  k = n_clusters,
  kmeans_params = params_kmeans_gpu(fixed = FALSE),
  seed = 99L,
  .verbose = FALSE
)

res_b <- k_means_cluster_gpu(
  data = cluster_data,
  k = n_clusters,
  kmeans_params = params_kmeans_gpu(fixed = FALSE),
  seed = 99L,
  .verbose = FALSE
)

expect_equal(
  current = res_a$assignments,
  target = res_b$assignments,
  info = "gpu kmeans is deterministic with same seed"
)

## data frame input ------------------------------------------------------------

res_df <- k_means_cluster_gpu(
  data = as.data.frame(cluster_data),
  k = n_clusters,
  kmeans_params = params_kmeans_gpu(fixed = FALSE),
  seed = 42L,
  .verbose = FALSE
)

expect_equal(
  current = res_df$assignments,
  target = res_gpu$assignments,
  info = "data frame input matches matrix input"
)

## quantised vs full precision -------------------------------------------------

res_quantised <- k_means_cluster_gpu(
  data = cluster_data,
  k = n_clusters,
  kmeans_params = params_kmeans_gpu(fixed = FALSE, quantise = TRUE),
  seed = 42L,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    x = res_quantised$centroids,
    mode = "numeric",
    nrows = n_clusters,
    ncols = 32L
  ),
  info = "quantised gpu kmeans centroids have correct dimensions"
)

ari_quantised <- manifoldsR::calc_ari(
  as.integer(cluster_membership),
  res_quantised$assignments
)

expect_true(
  current = ari_quantised > 0.9,
  info = "quantised gpu kmeans recovers known clusters"
)

expect_true(
  current = manifoldsR::calc_ari(
    res_gpu$assignments,
    res_quantised$assignments
  ) >
    0.9,
  info = "quantised agrees with full precision"
)

## agreement with cpu full k-means ---------------------------------------------

res_cpu_full <- manifoldsR::kmeans_cluster(
  data = cluster_data,
  k = n_clusters,
  method = "full",
  seed = 42L,
  .verbose = FALSE
)

expect_true(
  current = manifoldsR::calc_ari(
    res_cpu_full$assignments,
    res_gpu$assignments
  ) >
    0.9,
  info = "gpu kmeans agrees with cpu full kmeans"
)
