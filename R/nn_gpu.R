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
  assertNnParamsGpu(nn_params)
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
