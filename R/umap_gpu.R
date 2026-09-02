## param combination -----------------------------------------------------------

#' Internal helper to prepare the UMAP parameters (GPU version)
#'
#' @param n Integer. Number of samples in the data set
#' @param min_dist Numeric. Minimum distance between embedded points.
#' @param spread Numeric. Effective scale of embedded points.
#' @param knn_method String. Method to use to generate the kNN graph.
#' @param nn_params Named list. The nearest neighbour search parameters (GPU).
#' @param umap_params Named list. The UMAP-specific parameters (GPU).
#' @param .verbose Boolean. Controls verbosity
#'
#' @return Returns the list of final parameters.
#'
#' @export
#'
#' @keywords internal
.prepare_umap_params_gpu <- function(
  n,
  min_dist,
  spread,
  knn_method,
  nn_params,
  umap_params,
  .verbose = TRUE
) {
  # checks
  checkmate::qassert(n, "I1")
  checkmate::qassert(min_dist, "N1")
  checkmate::qassert(spread, "N1")
  checkmate::assertChoice(
    knn_method,
    c(
      "ivf",
      "nndescent",
      "exhaustive"
    )
  )
  assertNnParamsGpu(nn_params)
  assertUmapParamsGpu(umap_params)
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

  final_params <- c(nn_params, umap_params)
  final_params[["min_dist"]] <- min_dist
  final_params[["spread"]] <- spread
  final_params[["knn_method"]] <- knn_method

  # determine n_epochs if not specified
  if (is.null(final_params$n_epochs)) {
    if (
      final_params$optimiser %in% c("adam_parallel", "adam_gpu") | n <= 10000L
    ) {
      if (.verbose) {
        message(
          paste(
            "Using n_epochs = 500",
            "(dataset <10k samples or 'adam_parallel'/'adam_gpu' optimiser)"
          )
        )
      }
      final_params$n_epochs <- 500L
    } else {
      if (.verbose) {
        message(
          "Using n_epochs = 200 (dataset <=10k samples with sgd/adam optimiser)"
        )
      }
      final_params$n_epochs <- 200L
    }
  } else {
    if (.verbose) {
      message(
        "Using user defined n_epochs"
      )
    }
    final_params$n_epochs <- as.integer(final_params$n_epochs)
  }

  final_params
}


## main function ---------------------------------------------------------------

#' Rust-based UMAP (GPU)
#'
#' @description Performs UMAP dimensionality reduction on the input data.
#' This function provides a user-friendly interface with input validation
#' before calling the Rust implementation. Leverages GPU-accelerated kNN
#' searches and in the default setting also uses a GPU-accelerated Adam
#' optimiser for the embedding.
#'
#' @param data Numerical matrix or data frame. The data to embed of shape
#' samples x features. Will be coerced to a matrix.
#' @param knn Optional `NearestNeighbours` class. If provided, UMAP will skip
#' the k-nearest neighbour graph generation and use this one. Defaults to
#' `NULL`. See [manifoldsR::new_nearest_neighbour()] for details.
#' @param n_dim Integer. Number of dimensions in the embedding space.
#' Defaults to `2L`.
#' @param k Integer. Number of nearest neighbours to consider for manifold
#' approximation. Larger values result in more global structure being
#' preserved. Defaults to `15L`.
#' @param min_dist Numeric. Minimum distance between points in the embedding.
#' Controls how tightly points are packed. Smaller values result in more
#' clustered embeddings. Must be >= 0. Defaults to `0.5`. If you use SGD,
#' consider reducing this!
#' @param spread Numeric. Effective scale of embedded points. Determines the
#' scale at which embedded points will be spread out. Defaults to `1.0`.
#' @param knn_method Character. (Approximate) Nearest neighbour method to use.
#' One of `"exhaustive"`, `"ivf"` or `"nndescent"`. These are
#' GPU-accelerated methods.
#' @param nn_params_gpu Named list. Nearest neighbour search parameters, see
#' [params_nn_gpu()].
#' @param umap_params Named list. UMAP (GPU) algorithm parameters, see
#' [params_umap_gpu()].
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param use_high_precision Optional boolean. Gives fine-grained control over
#' `fp32` vs `fp64` usage. The GPU calculations will be forced into `fp32`.
#' @param .verbose Logical. Controls verbosity. Defaults to `TRUE`.
#'
#' @return A numerical matrix with dimensions samples x n_dim containing
#' the UMAP embedding.
#'
#' @export
umap_gpu <- function(
  data,
  knn = NULL,
  n_dim = 2L,
  k = 15L,
  min_dist = 0.5,
  spread = 1.0,
  knn_method = c(
    "nndescent",
    "exhaustive",
    "ivf"
  ),
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
) {
  # transformation
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
  checkmate::assert(
    checkmate::testNull(knn),
    checkmate::testClass(knn, "NearestNeighbours")
  )
  checkmate::assert_int(n_dim, lower = 1, upper = ncol(data))
  checkmate::qassert(k, "I1[2,)")
  checkmate::qassert(min_dist, "N1[0,)")
  checkmate::qassert(spread, "N1[0,)")
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(seed, "I1")

  final_umap_params <- .prepare_umap_params_gpu(
    n = nrow(data),
    min_dist = min_dist,
    spread = spread,
    knn_method = knn_method,
    nn_params = nn_params,
    umap_params = umap_params,
    .verbose = .verbose
  )

  # check if knn was provided
  res <- if (!is.null(knn)) {
    if (.verbose) {
      message("Using provided kNN graph.")
    }
    tryCatch(
      {
        rs_umap_from_knn_gpu(
          embd = data,
          knn_data = knn,
          n_dim = n_dim,
          min_dist = min_dist,
          spread = spread,
          k = k,
          umap_params = final_umap_params,
          seed = seed,
          use_high_precision = use_high_precision,
          verbose = parse_verbosity(.verbose)
        )
      },
      error = function(e) {
        stop("UMAP computation failed: ", e$message, call. = FALSE)
      }
    )
  } else {
    tryCatch(
      {
        rs_umap_gpu(
          embd = data,
          n_dim = n_dim,
          min_dist = min_dist,
          spread = spread,
          k = k,
          umap_params = final_umap_params,
          seed = seed,
          use_high_precision = use_high_precision,
          verbose = parse_verbosity(.verbose)
        )
      },
      error = function(e) {
        stop("UMAP computation failed: ", e$message, call. = FALSE)
      }
    )
  }

  res
}
