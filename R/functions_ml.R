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

# generate nearest neighbour graphs (on gpu) -----------------------------------

## manifoldsR ------------------------------------------------------------------

#' Generate a k-nearest neighbour graph (GPU-accelerated)
#'
#' @description
#' This function generates a kNN graph based on a given numeric matrix. Three
#' different GPU-accelerated versions are available
#' \itemize{
#'   \item `"exhaustive"` - Exact nearest neighbour search via GPU.
#'   \item `"ivf"` - Inverted file index that leverages k-means clustering
#'   and probing a few of the clusters via GPU-accelerated distance
#'   calculations.
#'   \item `"nndescent"` - A CAGRA style nearest neighbour search on the GPU.
#' }
#' @param data Numeric matrix. The embedding or feature matrix to compute
#' neighbours on. Rows are observations, columns are features.
#' @param k Integer. The number of nearest neighbours to compute.
#' @param knn_method Character. The algorithm to use for nearest neighbour
#' search. One of `c("exhaustive", "ivf", "nndescent")`. Defaults to
#' `"nndescent"`
#' @param nn_params List. Output of [params_nn_gpu()].
#' @param seed Integer. For reproducibility. Defaults to `42L`.
#' @param extract_knn `r lifecycle::badge("deprecated")` Use the `extract_knn`
#' field of [params_nn_gpu()] instead.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return A nearest neighbours class object with 1-indexed neighbour indices
#' and distances. Euclidean distances are true L2, not squared.
#'
#' @export
#'
#' @importFrom manifoldsR new_nearest_neighbour
generate_knn_graph_gpu <- function(
  data,
  k,
  knn_method = c(
    "nndescent",
    "exhaustive",
    "ivf"
  ),
  nn_params = params_nn_gpu(),
  seed = 42L,
  extract_knn = lifecycle::deprecated(),
  .verbose = TRUE
) {
  assert_gpu()

  knn_method <- match.arg(knn_method)

  if (lifecycle::is_present(extract_knn)) {
    lifecycle::deprecate_warn(
      when = "0.4.0",
      what = "generate_knn_graph_gpu(extract_knn)",
      with = "params_nn_gpu(extract_knn = )"
    )
    checkmate::qassert(extract_knn, "B1")
    nn_params$extract_knn <- extract_knn
  }

  # checks
  checkmate::assertMatrix(data, mode = "numeric")
  checkmate::qassert(k, "I1")
  checkmate::assertChoice(
    knn_method,
    c(
      "exhaustive",
      "ivf",
      "nndescent"
    )
  )
  assertNnGpuParams(nn_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

  nn_data <- rs_gpu_knn(
    embd = data,
    k = k,
    knn_method = knn_method,
    nn_params = nn_params,
    seed = seed,
    verbose = parse_verbosity(.verbose)
  )

  with(
    nn_data,
    new_nearest_neighbour(
      indices = c(t(indices)) + 1L, # 1-index
      dist = c(t(dist)),
      k = as.integer(k),
      n = nrow(data)
    )
  )
}
