# ------------------------------------------------------------------------------
# GPU-accelerated embeddings on plain matrices: UMAP, t-SNE and parametric UMAP.
# ------------------------------------------------------------------------------

# umap -------------------------------------------------------------------------

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
  assertNnGpuParams(nn_params)
  assertUmapGpuParams(umap_params)
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
#' @param nn_params Named list. Nearest neighbour search parameters, see
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
  assert_gpu()

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

# tsne -------------------------------------------------------------------------

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
#' the default), `"bh"` for Barnes-Hut, `"bh_qd"` for the quick-and-dirty
#' Barnes-Hut of qdtsne (tree depth capped at `max_depth` in
#' [params_tsne_gpu()], faster and coarser), `"fft"` for FFT-accelerated
#' interpolation or `"fft_3k"` for its three-kernel variant (one forward and
#' three inverse FFTs per epoch instead of four each). The last four run on
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
  approx_type = c("fft_3k_gpu", "bh", "bh_qd", "fft", "fft_3k"),
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
  checkmate::assertChoice(
    approx_type,
    c("fft_3k_gpu", "bh", "bh_qd", "fft", "fft_3k")
  )
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
    c("kmknn", "hnsw", "annoy", "nndescent", "balltree", "exhaustive")
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
#' `"kmknn"`, `"hnsw"`, `"annoy"`, `"nndescent"`, `"balltree"`, or
#' `"exhaustive"`. Defaults to `"kmknn"`.
#' @param nn_params Named list. Nearest neighbour parameters, see
#' [params_nn()].
#' @param parametric_umap_params Named list. Parametric UMAP parameters, see
#' [params_parametric_umap()].
#' @param use_gpu Boolean. Shall the neural net be trained on GPU via the
#' `wgpu` backend. On smaller data sets the CPU backend (`flex`) can be faster,
#' since kernel launch overhead dominates.
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
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
  knn_method = c(
    "kmknn",
    "hnsw",
    "annoy",
    "nndescent",
    "balltree",
    "exhaustive"
  ),
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

  # the flex CPU backend is a legitimate path here, so this is conditional
  if (use_gpu) {
    assert_gpu()
  }

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
        verbose = parse_verbosity(.verbose)
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

## save and load ---------------------------------------------------------------

#' Save a parametric UMAP as a qs2 file
#'
#' @param model `ParametricUmapModel` you want to save.
#' @param path String. The path to the file. Needs to end with `".qs"` file
#' extension.
#'
#' @returns Saves the file to the provided path.
#'
#' @export
save_parametric_umap <- function(model, path) {
  checkmate::assertClass(model, "ParametricUmapModel")
  checkmate::assertString(path)
  bytes <- rs_serialise_parametric_umap(model$model$ptr)
  qs2::qs_save(
    object = list(
      version = 1L,
      bytes = bytes,
      embedding = model$embedding,
      params = model$params
    ),
    file = path
  )
  invisible(path)
}

#' Load a parametric UMAP as a qs2 file
#'
#' @param path String. The path to the serialised trained parametric UMAP model
#' on disk.
#'
#' @returns The `ParametricUmapModel`.
#'
#' @export
load_parametric_umap <- function(path) {
  checkmate::assertFileExists(path)
  obj <- qs2::qs_read(path)
  if (!identical(obj$version, 1L)) {
    stop(
      "Unsupported parametric UMAP file version: ",
      obj$version,
      call. = FALSE
    )
  }
  ptr <- rs_deserialise_parametric_umap(obj$bytes)
  model_env <- new.env(parent = emptyenv())
  model_env$ptr <- ptr
  structure(
    list(embedding = obj$embedding, model = model_env, params = obj$params),
    class = "ParametricUmapModel"
  )
}
