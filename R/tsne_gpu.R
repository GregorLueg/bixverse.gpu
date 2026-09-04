## param combination -----------------------------------------------------------

#' Internal helper to prepare the t-SNE parameters (GPU version)
#'
#' @param knn_method String. Method to use to generate the kNN graph.
#' @param nn_params Named list. The nearest neighbour search parameters (GPU).
#' @param tsne_params Named list. The t-SNE-specific parameters (GPU).
#'
#' @return Returns the list of final parameters.
#'
#' @export
#'
#' @keywords internal
.prepare_tsne_params_gpu <- function(
  knn_method,
  nn_params,
  tsne_params
) {
  # checks
  checkmate::assertChoice(
    knn_method,
    c(
      "nndescent",
      "exhaustive",
      "ivf"
    )
  )
  assertNnParamsGpu(nn_params)
  assertTsneParamsGpu(tsne_params)

  final_params <- c(nn_params, tsne_params)
  final_params[["knn_method"]] <- knn_method

  final_params
}


## main function ---------------------------------------------------------------

#' Rust-based t-SNE (GPU)
#'
#' @description Performs t-SNE dimensionality reduction on the input data.
#' This function provides a user-friendly interface with input validation
#' before calling the Rust implementation. Leverages GPU-accelerated kNN
#' searches. The optimisation itself still runs on the CPU (a GPU optimiser
#' is on the roadmap).
#'
#' @details The number of neighbours is derived from `perplexity` on the Rust
#' side following the usual `3 * perplexity` convention.
#'
#' @param data Numerical matrix or data frame. The data to embed of shape
#' samples x features. Will be coerced to a matrix.
#' @param knn Optional `NearestNeighbours` class. If provided, t-SNE will skip
#' the k-nearest neighbour graph generation and use this one. Defaults to
#' `NULL`. See [manifoldsR::new_nearest_neighbour()] for details.
#' @param n_dim Integer. Number of dimensions in the embedding space.
#' Currently only `2L` is supported. Defaults to `2L`.
#' @param perplexity Numeric. Perplexity parameter, related to the number of
#' nearest neighbours used in manifold learning. Typical values are between
#' 5 and 50. Defaults to `20.0`.
#' @param approx_type Character. Approximation method for computing repulsive
#' forces. One of `"bh"` for Barnes-Hut or `"fft"` for FFT-accelerated
#' interpolation. Defaults to `"bh"`. The FFT variant is only available on
#' Unix systems.
#' @param knn_method Character. GPU-accelerated (approximate) nearest
#' neighbour method to use. One of `"nndescent"`, `"exhaustive"`, or `"ivf"`.
#' @param nn_params Named list. Nearest neighbour search parameters, see
#' [params_nn_gpu()].
#' @param tsne_params Named list. t-SNE (GPU) algorithm parameters, see
#' [params_tsne_gpu()].
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param use_high_precision Optional boolean. Gives fine-grained control over
#' `fp32` vs `fp64` usage. The GPU kNN calculations will be forced into `fp32`.
#' @param .verbose Logical. Controls verbosity. Defaults to `TRUE`.
#'
#' @return A numerical matrix with dimensions samples x n_dim containing
#' the t-SNE embedding.
#'
#' @export
tsne_gpu <- function(
  data,
  knn = NULL,
  n_dim = 2L,
  perplexity = 20.0,
  approx_type = c("bh", "fft"),
  knn_method = c(
    "nndescent",
    "exhaustive",
    "ivf"
  ),
  nn_params = params_nn_gpu(),
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
) {
  assert_gpu()

  # transformation
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  approx_type <- match.arg(approx_type)
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
  checkmate::qassert(n_dim, "I1[2,2]")
  checkmate::qassert(perplexity, "N1[1,)")
  checkmate::assertChoice(approx_type, c("bh", "fft"))
  checkmate::qassert(seed, "I1")
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

  # FFT only supported on Unix
  if (approx_type == "fft" && .Platform$OS.type != "unix") {
    stop(
      "The FFT approximation is not supported on non-Unix systems.",
      call. = FALSE
    )
  }

  final_tsne_params <- .prepare_tsne_params_gpu(
    knn_method = knn_method,
    nn_params = nn_params,
    tsne_params = tsne_params
  )

  # check if knn was provided
  res <- if (!is.null(knn)) {
    if (.verbose) {
      message("Using provided kNN graph.")
    }
    tryCatch(
      {
        rs_tsne_from_knn_gpu(
          embd = data,
          knn_data = knn,
          n_dim = as.integer(n_dim),
          perplexity = perplexity,
          approx_type = approx_type,
          tsne_params = final_tsne_params,
          seed = seed,
          use_high_precision = use_high_precision,
          verbose = parse_verbosity(.verbose)
        )
      },
      error = function(e) {
        stop("t-SNE computation failed: ", e$message, call. = FALSE)
      }
    )
  } else {
    tryCatch(
      {
        rs_tsne_gpu(
          embd = data,
          n_dim = as.integer(n_dim),
          perplexity = perplexity,
          approx_type = approx_type,
          tsne_params = final_tsne_params,
          seed = seed,
          use_high_precision = use_high_precision,
          verbose = parse_verbosity(.verbose)
        )
      },
      error = function(e) {
        stop("t-SNE computation failed: ", e$message, call. = FALSE)
      }
    )
  }

  res
}
