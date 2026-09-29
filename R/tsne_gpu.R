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
  assertNnGpuParams(nn_params)
  assertTsneGpuParams(tsne_params)

  final_params <- c(nn_params, tsne_params)
  final_params[["knn_method"]] <- knn_method

  final_params
}


## main function ---------------------------------------------------------------

#' Rust-based t-SNE (GPU)
#'
#' @description Performs t-SNE dimensionality reduction on the input data.
#' The kNN search runs on the GPU. With the default `"fft_3k_gpu"` the
#' optimiser runs on the GPU as well, so the whole embedding stays on the
#' device; the other approximations optimise on the CPU.
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
#' forces. One of `"fft_3k_gpu"` (three-kernel FFT interpolation on the GPU,
#' the default), `"bh"` for Barnes-Hut, `"fft"` for FFT-accelerated
#' interpolation or `"fft_3k"` for its three-kernel variant (one forward and
#' three inverse FFTs per epoch instead of four each). The last three run on
#' the CPU; `"fft"` and `"fft_3k"` are only available on Unix systems.
#' @param knn_method Character. GPU-accelerated (approximate) nearest
#' neighbour method to use. One of `"nndescent"`, `"exhaustive"`, or `"ivf"`.
#' @param nn_params Named list. Nearest neighbour search parameters, see
#' [params_nn_gpu()]. tSNE uses higher k usually, hence, `"ivf"`` is the default
#' here.
#' @param tsne_params Named list. t-SNE (GPU) algorithm parameters, see
#' [params_tsne_gpu()].
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param use_high_precision Optional boolean. Gives fine-grained control over
#' `fp32` vs `fp64` usage. The GPU kNN calculations will be forced into `fp32`.
#' Ignored with a warning for `"fft_3k_gpu"`, which always runs in `fp32`.
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
  approx_type = c("fft_3k_gpu", "bh", "fft", "fft_3k"),
  knn_method = c(
    "ivf",
    "exhaustive",
    "nndescent"
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
  checkmate::assertChoice(approx_type, c("fft_3k_gpu", "bh", "fft", "fft_3k"))
  checkmate::qassert(seed, "I1")
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

  # CPU FFT needs FFTW, which is only built on Unix
  if (approx_type %in% c("fft", "fft_3k") && .Platform$OS.type != "unix") {
    stop(
      "The CPU FFT approximations are not supported on non-Unix systems. ",
      "Use `approx_type = \"fft_3k_gpu\"` instead.",
      call. = FALSE
    )
  }
  if (approx_type == "fft_3k_gpu" && isTRUE(use_high_precision)) {
    warning(
      "`use_high_precision` is ignored for \"fft_3k_gpu\", which runs in fp32.",
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
