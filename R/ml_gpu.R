# ------------------------------------------------------------------------------
# Contains general GPU-accelerated machine learning algorithms:
# - GPU-accelerated k-means. Useful in low-dimensional, high N/K scenarios (at
#   least on Apple Silicon)
# ------------------------------------------------------------------------------

# clustering -------------------------------------------------------------------

## k-means gpu -----------------------------------------------------------------

#' GPU-accelerated k-means clustering
#'
#' @description
#' Performs k-means clustering as a GPU-accelerated form. In its current form
#' uses the cubecl with wgpu backend.
#'
#' @param data Numerical matrix or data frame. The data to cluster, of shape
#' samples x features. Will be coerced to a matrix.
#' @param k Integer. Number of clusters to create. Must be >= 2.
#' @param kmeans_params Named list. GPU-accelerated k-mean parameters, see
#' [params_kmeans_gpu()].
#' @param seed Integer. Random seed for reproducibility. Defaults to 42L.
#' @param .verbose Logical. Controls verbosity. Defaults to `TRUE`.
#'
#' @returns A `KMeansClusterGPU` class with assignments and centroids.
#'
#' @export
k_means_cluster_gpu <- function(
  data,
  k,
  kmeans_params = params_kmeans_gpu(),
  seed = 42L,
  .verbose = TRUE
) {
  assert_gpu()

  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }

  # checks
  checkmate::assertMatrix(data, mode = "numeric")
  checkmate::qassert(k, "I1[2,)")
  assertKMeansGpuParams(kmeans_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, "B1")

  res <- rs_kmeans_gpu(
    data = data,
    dist = kmeans_params$metric,
    n_centroids = k,
    kmeans_params = kmeans_params,
    seed = seed,
    verbose = .verbose
  )

  new_kmeans_cluster_gpu(
    centroids = res$centroids,
    assignments = res$assignments,
    k = k,
    metric = kmeans_params$metric
  )
}
