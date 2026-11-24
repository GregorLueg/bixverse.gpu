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

#' GPU-accelerated Leiden or Louvain community detection
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
#' @param graph `SingleCellConnectivity` or undirected igraph. For an igraph,
#' the edge attribute `weight` is used if present, otherwise all edges get
#' weight 1. An igraph is converted with [bixverse::as_sc_connectivity()]
#' first.
#' @param resolution Numeric. Modularity resolution parameter. Higher values
#' give more communities.
#' @param algorithm String. One of `c("leiden", "louvain")`.
#' @param beta Numeric. Randomness of the Leiden refinement merges. Ignored for
#' Louvain.
#' @param n_iterations Optional integer. Number of full iterations, each
#' starting from the previous partition. A negative value iterates until the
#' partition is stable. If `NULL`, the bixverse default for `algorithm`.
#' @param min_gain Numeric. With a negative `n_iterations`, also stop once an
#' iteration gains no more than this in modularity.
#' @param local_moving Optional string. One of `c("queue", "coloured")`.
#' Applies to the coarser levels only; level 0 always runs the coloured moves on
#' the GPU. `NULL` is `"queue"` for Leiden and `"coloured"` for Louvain.
#' @param node_order Optional string. One of `c("bfs", "shuffle")`. Visit order
#' of the queue on the coarser levels, `"bfs"` (sweep in BFS order from a
#' random start) or `"shuffle"` (as igraph). `NULL` is `"bfs"` for Leiden and
#' `"shuffle"` for Louvain.
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
  local_moving = NULL,
  node_order = NULL,
  seed = 42L
) {
  assert_gpu()

  algorithm <- match.arg(algorithm)
  # checks
  checkmate::assertMultiClass(graph, c("SingleCellConnectivity", "igraph"))
  checkmate::qassert(resolution, "N1(0,)")
  checkmate::assertChoice(algorithm, c("leiden", "louvain"))
  checkmate::qassert(beta, "N1(0,)")
  checkmate::qassert(n_iterations, c("X1", "0"))
  checkmate::assertTRUE(is.null(n_iterations) || n_iterations != 0)
  checkmate::qassert(min_gain, "N1[0,)")
  checkmate::assertChoice(local_moving, c("queue", "coloured"), null.ok = TRUE)
  checkmate::assertChoice(node_order, c("bfs", "shuffle"), null.ok = TRUE)
  checkmate::qassert(seed, "X1[0,)")

  if (is.null(n_iterations)) {
    n_iterations <- bixverse:::.default_community_iters(algorithm)
  }

  # the pointer belongs to bixverse's Rust lib, so the graph crosses over as
  # an edge list
  graph <- bixverse::as_sc_connectivity(graph)
  edges <- bixverse::rs_graph_to_edges(bixverse:::.connectivity_ptr(graph))

  rs_leiden_clustering_gpu(
    edges = edges$edges,
    weights = edges$weights,
    n_nodes = graph$n,
    leiden_params = list(
      resolution = resolution,
      beta = beta,
      n_iterations = as.integer(n_iterations),
      min_gain = min_gain,
      refine = algorithm == "leiden",
      local_moving = local_moving,
      node_order = node_order,
      seed = as.integer(seed)
    )
  )
}

# generate nearest neighbour graphs (on gpu) -----------------------------------

## manifoldsR ------------------------------------------------------------------

#' Generate a k-nearest neighbour graph (GPU-accelerated)
#'
#' @description
#' This function generates a kNN graph based on a given numeric matrix. Four
#' different GPU-accelerated versions are available
#' \itemize{
#'   \item `"nndescent"` - The NN-descent kNN graph itself, no index and no
#'   search. Rows the descent left short are topped up with an exact search.
#'   \item `"cagra"` - The NN-descent graph pruned into a CAGRA graph and beam
#'   searched.
#'   \item `"exhaustive"` - Exact nearest neighbour search via GPU.
#'   \item `"ivf"` - Inverted file index that clusters the data with k-means
#'   and probes a few of the clusters at query time.
#' }
#' @param data Numeric matrix. The embedding or feature matrix to compute
#' neighbours on. Rows are observations, columns are features.
#' @param k Integer. The number of nearest neighbours to compute.
#' @param knn_method Character. The algorithm to use for nearest neighbour
#' search. One of `c("nndescent", "cagra", "exhaustive", "ivf")`. Defaults to
#' `"nndescent"`.
#' @param nn_params List. Output of [params_nn_gpu()].
#' @param seed Integer. For reproducibility. Defaults to `42L`.
#' @param extract_knn `r lifecycle::badge("deprecated")` Use
#' `knn_method = "nndescent"` (graph only) or `knn_method = "cagra"` (beam
#' search) instead.
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
    "cagra",
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
      with = "generate_knn_graph_gpu(knn_method)"
    )
    checkmate::qassert(extract_knn, "B1")
    if (knn_method %in% c("nndescent", "cagra")) {
      knn_method <- if (extract_knn) "nndescent" else "cagra"
    }
  }

  # checks
  checkmate::assertMatrix(data, mode = "numeric")
  checkmate::qassert(k, "I1")
  checkmate::assertChoice(
    knn_method,
    c("nndescent", "cagra", "exhaustive", "ivf")
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
