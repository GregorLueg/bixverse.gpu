# ------------------------------------------------------------------------------
# GPU-accelerated embeddings on plain matrices: UMAP, t-SNE, PHATE and
# parametric UMAP.
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

  # determine n_epochs if not specified; SGD drops to 200 on large data like
  # umap-learn, Adam always gets 500
  if (is.null(final_params$n_epochs)) {
    sgd <- final_params$optimiser %in% c("sgd", "sgd_parallel")
    if (sgd && n > 10000L) {
      if (.verbose) {
        message(
          "Using n_epochs = 200 (dataset >10k samples with an SGD optimiser)"
        )
      }
      final_params$n_epochs <- 200L
    } else {
      if (.verbose) {
        message(
          "Using n_epochs = 500 (Adam optimiser or dataset <=10k samples)"
        )
      }
      final_params$n_epochs <- 500L
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
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the kNN
#' search and the `"adam_gpu"` optimiser on MLX (Apple Silicon, needs a build
#' with the MLX backend, see [mlx_available()]). CPU optimisers stay on the
#' CPU.
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
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  backend <- match.arg(backend)
  assert_backend(backend)

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
  checkmate::qassert(seed, "I1")
  checkmate::assertChoice(backend, c("wgpu", "mlx"))

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
    rs_umap_from_knn_gpu(
      embd = data,
      knn_data = knn,
      n_dim = n_dim,
      min_dist = min_dist,
      spread = spread,
      k = k,
      umap_params = final_umap_params,
      seed = seed,
      backend = backend,
      verbose = parse_verbosity(.verbose)
    )
  } else {
    rs_umap_gpu(
      embd = data,
      n_dim = n_dim,
      min_dist = min_dist,
      spread = spread,
      k = k,
      umap_params = final_umap_params,
      seed = seed,
      backend = backend,
      verbose = parse_verbosity(.verbose)
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
#' the CPU.
#' @param knn_method Character. GPU-accelerated (approximate) nearest
#' neighbour method to use. One of `"nndescent"`, `"exhaustive"`, or `"ivf"`.
#' @param nn_params Named list. Nearest neighbour search parameters, see
#' [params_nn_gpu()]. tSNE usually needs a higher k, hence `"ivf"` is the
#' default here.
#' @param tsne_params Named list. t-SNE (GPU) algorithm parameters, see
#' [params_tsne_gpu()].
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the kNN
#' search and the `"fft_3k_gpu"` optimiser on MLX (Apple Silicon, needs a
#' build with the MLX backend, see [mlx_available()]). CPU optimisers stay on
#' the CPU.
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
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  # transformation
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  approx_type <- match.arg(approx_type)
  knn_method <- match.arg(knn_method)
  backend <- match.arg(backend)
  assert_backend(backend)

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
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

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
    rs_tsne_from_knn_gpu(
      embd = data,
      knn_data = knn,
      n_dim = as.integer(n_dim),
      perplexity = perplexity,
      approx_type = approx_type,
      tsne_params = final_tsne_params,
      seed = seed,
      backend = backend,
      verbose = parse_verbosity(.verbose)
    )
  } else {
    rs_tsne_gpu(
      embd = data,
      n_dim = as.integer(n_dim),
      perplexity = perplexity,
      approx_type = approx_type,
      tsne_params = final_tsne_params,
      seed = seed,
      backend = backend,
      verbose = parse_verbosity(.verbose)
    )
  }

  res
}

# phate ------------------------------------------------------------------------

## param combination -----------------------------------------------------------

#' Internal helper to prepare the PHATE parameters (GPU version)
#'
#' @param knn_method String. Method to use to generate the kNN graph.
#' @param nn_params Named list. The nearest neighbour search parameters (GPU).
#' @param phate_params Named list. The PHATE-specific parameters (GPU).
#'
#' @return Returns the list of final parameters.
#'
#' @export
#'
#' @keywords internal
.prepare_phate_params_gpu <- function(
  knn_method,
  nn_params,
  phate_params
) {
  # checks
  checkmate::assertChoice(knn_method, c("nndescent", "exhaustive", "ivf"))
  assertNnGpuParams(nn_params)
  assertPhateGpuParams(phate_params)

  final_params <- c(nn_params, phate_params)
  final_params[["knn_method"]] <- knn_method

  final_params
}

## main function ---------------------------------------------------------------

#' Rust-based PHATE (GPU)
#'
#' @description Performs PHATE dimensionality reduction on the input data. The
#' kNN search runs on the GPU, and so do the randomised SVD and k-means of the
#' default spectral landmark selection. The diffusion time, the powering of
#' the operator, MDS and the landmark interpolation stay on the CPU.
#'
#' @details The kNN search runs `6 * k + 5` wide: neighbours past `k` join the
#' kernel while their affinity clears `thresh` (see [params_phate_gpu()]). A
#' precomputed graph should be that wide as well.
#'
#' @param data Numerical matrix or data frame. The data to embed of shape
#' samples x features. Will be coerced to a matrix.
#' @param knn Optional `NearestNeighbours` class. If provided, PHATE will skip
#' the k-nearest neighbour graph generation and use this one. Should be
#' `6 * k + 5` wide; a narrower graph gives a warning. Defaults to `NULL`. See
#' [manifoldsR::new_nearest_neighbour()] for details.
#' @param n_dim Integer. Number of dimensions in the embedding space. Defaults
#' to `2L`.
#' @param k Integer. Number of nearest neighbours that set the kernel
#' bandwidth. Defaults to `5L`.
#' @param knn_method Character. GPU-accelerated (approximate) nearest
#' neighbour method to use. One of `"ivf"`, `"exhaustive"` or `"nndescent"`.
#' Defaults to `"ivf"`: the `6 * k + 5` wide search is past the width where
#' NN-descent keeps up.
#' @param nn_params Named list. Nearest neighbour search parameters, see
#' [params_nn_gpu()].
#' @param phate_params Named list. PHATE (GPU) algorithm parameters, see
#' [params_phate_gpu()].
#' @param seed Integer. Random seed for reproducibility. Defaults to `42L`.
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the kNN
#' search and the spectral landmark selection on MLX (Apple Silicon, needs a
#' build with the MLX backend, see [mlx_available()]).
#' @param .verbose Logical. Controls verbosity. Defaults to `TRUE`.
#'
#' @return A numerical matrix with dimensions samples x n_dim containing
#' the PHATE embedding.
#'
#' @references Moon, et al., Nat. Biotechnol., 2019
#'
#' @export
phate_gpu <- function(
  data,
  knn = NULL,
  n_dim = 2L,
  k = 5L,
  knn_method = c("ivf", "exhaustive", "nndescent"),
  nn_params = params_nn_gpu(),
  phate_params = params_phate_gpu(),
  seed = 42L,
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  # transformation
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  knn_method <- match.arg(knn_method)
  backend <- match.arg(backend)
  assert_backend(backend)

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
  checkmate::qassert(n_dim, "I1[1,)")
  checkmate::qassert(k, "I1[1,)")
  checkmate::assertChoice(knn_method, c("nndescent", "exhaustive", "ivf"))
  checkmate::qassert(seed, "I1")
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
  checkmate::qassert(.verbose, c("B1", "I1[0, 2]"))

  final_phate_params <- .prepare_phate_params_gpu(
    knn_method = knn_method,
    nn_params = nn_params,
    phate_params = phate_params
  )

  search_width <- 6L * k + 5L
  if (!is.null(knn) && knn$k < search_width) {
    warning(sprintf(
      paste(
        "The provided kNN graph has %d neighbours per sample; PHATE with",
        "k = %d searches %d wide. Neighbours past the graph are lost from the",
        "kernel."
      ),
      knn$k,
      k,
      search_width
    ))
  }
  if (is.null(final_phate_params$n_landmarks) && nrow(data) >= 10000L) {
    warning(sprintf(
      paste(
        "n_landmarks = NULL on %d samples runs the full N x N diffusion",
        "operator. Expect it to be slow and memory hungry."
      ),
      nrow(data)
    ))
  }

  # check if knn was provided
  res <- if (!is.null(knn)) {
    if (.verbose) {
      message("Using provided kNN graph.")
    }
    rs_phate_from_knn_gpu(
      embd = data,
      knn_data = knn,
      n_dim = as.integer(n_dim),
      k = as.integer(k),
      phate_params = final_phate_params,
      seed = seed,
      backend = backend,
      verbose = parse_verbosity(.verbose)
    )
  } else {
    rs_phate_gpu(
      embd = data,
      n_dim = as.integer(n_dim),
      k = as.integer(k),
      phate_params = final_phate_params,
      seed = seed,
      backend = backend,
      verbose = parse_verbosity(.verbose)
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
#' to `0.5`, the manifolds-rs default.
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
  min_dist = 0.5,
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

  res <- rs_parametric_umap(
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
