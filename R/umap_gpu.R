# parametric umap --------------------------------------------------------------

## helpers ---------------------------------------------------------------------

#' Internal helper to prepare parametric UMAP parameters
#'
#' @param min_dist Numeric. Minimum distance between embedded points.
#' @param spread Numeric. Effective scale of embedded points.
#' @param knn_method String. Approximate nearest neighbour method.
#' @param nn_params Named list. Nearest neighbour parameters, see
#' [manifoldsR::params_nn()].
#' @param parametric_umap_params Named list. Parametric UMAP parameters.
#'
#' @return Returns the merged list of final parameters.
#'
#' @keywords internal
.prepare_parametric_umap_params <- function(
  min_dist,
  spread,
  knn_method,
  nn_params,
  parametric_umap_params
) {
  checkmate::qassert(min_dist, "N1[0,)")
  checkmate::qassert(spread, "N1(0,)")
  checkmate::assertChoice(
    knn_method,
    c("hnsw", "annoy", "nndescent", "balltree", "exhaustive")
  )
  manifoldsR:::assertNnParams(nn_params)
  assertParametricUmapParams(parametric_umap_params)

  final_params <- c(nn_params, parametric_umap_params)
  final_params[["min_dist"]] <- min_dist
  final_params[["spread"]] <- spread
  final_params[["knn_method"]] <- knn_method

  final_params
}

## main function ---------------------------------------------------------------

#' Parametric UMAP
#'
#' @description Performs parametric UMAP dimensionality reduction using a
#' neural network encoder trained on the GPU via wgpu.
#'
#' @param data Numerical matrix or data frame. The data to embed of shape
#' samples x features. Will be coerced to a matrix.
#' @param n_dim Integer. Number of embedding dimensions. Defaults to `2L`.
#' @param k Integer. Number of nearest neighbours. Defaults to `15L`.
#' @param min_dist Numeric. Minimum distance between embedded points. Defaults
#' to `0.1`.
#' @param spread Numeric. Effective scale of embedded points. Defaults to
#' `1.0`.
#' @param knn_method Character. Approximate nearest neighbour algorithm. One of
#' `"hnsw"`, `"annoy"`, `"nndescent"`, `"balltree"`, or `"exhaustive"`.
#' Defaults to `"hnsw"`.
#' @param nn_params Named list. Nearest neighbour parameters, see
#' [params_nn()].
#' @param parametric_umap_params Named list. Parametric UMAP parameters, see
#' [params_parametric_umap()].
#' @param use_gpu Boolean. Shall the neural net be trained on GPU via the
#' `wgpu` backend. On smaller datasets, the CPU can be faster (via the
#' `ndarray`) backend due to kernel launch overhead.
#' data sets, the CPU will be faster via the Ndarray.
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param .verbose Logical. Controls verbosity. Defaults to `TRUE`.
#'
#' @return A `ParametricUmapModel` object containing the embedding matrix
#' and the trained encoder model.
#'
#' @export
parametric_umap <- function(
  data,
  n_dim = 2L,
  k = 15L,
  min_dist = 0.1,
  spread = 1.0,
  knn_method = c("hnsw", "annoy", "nndescent", "balltree", "exhaustive"),
  nn_params = manifoldsR::params_nn(),
  parametric_umap_params = params_parametric_umap(),
  use_gpu = TRUE,
  seed = 42L,
  .verbose = TRUE
) {
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  knn_method <- match.arg(knn_method)

  checkmate::assert_matrix(
    data,
    mode = "numeric",
    any.missing = FALSE,
    min.rows = 2,
    min.cols = 1
  )
  checkmate::assert_int(n_dim, lower = 1, upper = ncol(data))
  checkmate::qassert(k, "I1[2,)")
  checkmate::qassert(min_dist, "N1[0,)")
  checkmate::qassert(spread, "N1(0,)")
  checkmate::qassert(use_gpu, "B1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, "B1")

  final_params <- .prepare_parametric_umap_params(
    min_dist = min_dist,
    spread = spread,
    knn_method = knn_method,
    nn_params = nn_params,
    parametric_umap_params = parametric_umap_params
  )

  res <- tryCatch(
    {
      rs_parametric_umap(
        data = data,
        n_dim = n_dim,
        min_dist = min_dist,
        spread = spread,
        k = k,
        parametric_params = final_params,
        seed = seed,
        use_gpu = use_gpu,
        verbose = .verbose
      )
    },
    error = function(e) {
      stop("Parametric UMAP computation failed: ", e$message, call. = FALSE)
    }
  )

  # wrap model pointer in environment to prevent GC
  model_env <- new.env(parent = emptyenv())
  model_env$ptr <- res$model

  structure(
    list(
      embedding = res$embedding,
      model = model_env,
      params = list(
        n_dim = n_dim,
        k = k,
        min_dist = min_dist,
        spread = spread,
        knn_method = knn_method,
        n_samples = nrow(data),
        n_features = ncol(data)
      )
    ),
    class = "ParametricUmapModel"
  )
}
