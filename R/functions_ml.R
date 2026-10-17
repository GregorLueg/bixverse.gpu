# ------------------------------------------------------------------------------
# Contains general GPU-accelerated machine learning algorithms:
# - GPU-accelerated k-means. Useful in low-dimensional, high N/K scenarios (at
#   least on Apple Silicon)
# - GPU-accelerated Leiden/Louvain. Level-0 local moving on the GPU, the rest
#   of the multi-level run on the CPU
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

## leiden gpu ------------------------------------------------------------------

#' GPU-accelerated Leiden or Louvain community detection on an igraph
#'
#' @description
#' GPU counterpart of [bixverse::leiden_clustering()]. The level-0 local
#' moving runs on the GPU; the coarser levels, the Leiden refinement and the
#' aggregation stay on the CPU. Graphs below 50,000 nodes, or with a node of
#' degree above 512, run fully on the CPU with coloured local moving, as the
#' kernel launches cost more than they save there. On one device the partition
#' depends on `seed` only, but it is not identical to the CPU result: the GPU
#' accepts moves with an order-free rule instead of the CPU's sequential
#' re-check. Communities are numbered by size, `1` being the largest.
#'
#' @param graph igraph. Undirected. If it has an edge attribute `weight`, it is
#' used, otherwise all edges get weight 1.
#' @param resolution Numeric. Modularity resolution parameter. Higher values
#' give more communities.
#' @param algorithm String. One of `c("leiden", "louvain")`.
#' @param beta Numeric. Randomness of the Leiden refinement merges. Ignored for
#' Louvain.
#' @param n_iterations Optional integer. Number of full iterations, each
#' starting from the previous partition. A negative value iterates until the
#' partition is stable. If `NULL`, `5L` for Leiden and `2L` for Louvain.
#' @param min_gain Numeric. With a negative `n_iterations`, also stop once an
#' iteration gains no more than this in modularity.
#' @param local_moving String. One of `c("queue", "coloured")`. Applies to the
#' coarser levels only; level 0 always runs the coloured moves on the GPU.
#' @param seed Integer. Random seed.
#'
#' @returns Integer vector with the community of each node (1-based, `1` is
#' the largest community).
#'
#' @export
#'
#' @references Traag, Waltman & van Eck, Sci Rep, 2019; Blondel, et al., J Stat
#' Mech, 2008
leiden_clustering_gpu <- function(
  graph,
  resolution = 1.0,
  algorithm = c("leiden", "louvain"),
  beta = 0.01,
  n_iterations = NULL,
  min_gain = 0,
  local_moving = c("queue", "coloured"),
  seed = 42L
) {
  assert_gpu()

  algorithm <- match.arg(algorithm)
  local_moving <- match.arg(local_moving)
  # checks
  checkmate::assertClass(graph, "igraph")
  checkmate::assertFALSE(igraph::is_directed(graph))
  checkmate::qassert(resolution, "N1(0,)")
  checkmate::assertChoice(algorithm, c("leiden", "louvain"))
  checkmate::qassert(beta, "N1(0,)")
  checkmate::qassert(n_iterations, c("X1", "0"))
  checkmate::assertTRUE(is.null(n_iterations) || n_iterations != 0)
  checkmate::qassert(min_gain, "N1[0,)")
  checkmate::assertChoice(local_moving, c("queue", "coloured"))
  checkmate::qassert(seed, "X1[0,)")

  if (is.null(n_iterations)) {
    n_iterations <- bixverse:::.default_community_iters(algorithm)
  }

  edges <- igraph::as_edgelist(graph, names = FALSE)
  weights <- igraph::E(graph)$weight
  if (is.null(weights)) {
    weights <- rep(1, nrow(edges))
  }

  rs_leiden_clustering_gpu(
    from = as.integer(edges[, 1]),
    to = as.integer(edges[, 2]),
    weights = as.numeric(weights),
    n_nodes = igraph::vcount(graph),
    leiden_params = list(
      resolution = resolution,
      beta = beta,
      n_iterations = as.integer(n_iterations),
      min_gain = min_gain,
      refine = algorithm == "leiden",
      local_moving = local_moving,
      seed = as.integer(seed)
    )
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
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the
#' search on Apple Silicon via MLX and needs a build with the MLX backend,
#' see [mlx_available()].
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
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  knn_method <- match.arg(knn_method)
  backend <- match.arg(backend)
  assert_backend(backend)

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
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

  nn_data <- rs_gpu_knn(
    embd = data,
    k = k,
    knn_method = knn_method,
    nn_params = nn_params,
    seed = seed,
    backend = backend,
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
