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
#' `"exhaustive"`
#' @param nn_params List. Output of [params_nn_gpu()].
#' @param extract_knn Boolean. CAGRA-specific (`knn_method = "nndescent"`).
#' Shall the beam search be skipped and the kNN graph be extracted directly
#' after the NNDescent iteration and optional refine sweeps. Lower quality, but
#' faster.
#' @param seed Integer. For reproducibility. Defaults to `42L`.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return A nearest neighbours class object with 1-indexed neighbour indices
#' and distances.
#'
#' @export
#'
#' @importFrom manifoldsR new_nearest_neighbour
generate_knn_graph <- function(
  data,
  k,
  knn_method = c(
    "exhaustive",
    "ivf",
    "nndescent"
  ),
  nn_params = params_nn_gpu(),
  seed = 42L,
  extract_knn = FALSE,
  .verbose = TRUE
) {
  knn_method <- match.arg(knn_method)

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

  # overwrite this parameter here
  nn_params$k <- k

  # rust
  nn_data <- switch(
    knn_method,
    exhaustive = rs_exhaustive_gpu_knn(
      embd = data,
      k = k,
      dist_metric = nn_params$dist_metric,
      verbose = parse_verbosity(.verbose)
    ),
    ivf = rs_ivf_gpu_knn(
      embd = data,
      ivf_params = nn_params,
      seed = seed,
      verbose = parse_verbosity(.verbose)
    ),
    nndescent = rs_cagra_gpu_knn(
      embd = data,
      ivf_params = nn_params,
      extract_knn = extract_knn,
      seed = seed,
      verbose = parse_verbosity(.verbose)
    )
  )

  res <- with(
    nn_data,
    new_nearest_neighbour(
      indices = indices + 1L, # 1-index
      dist = dist,
      k = as.integer(k),
      n = as.integer(n)
    )
  )

  res
}
