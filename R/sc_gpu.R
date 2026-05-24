# knn searches -----------------------------------------------------------------

## to knn objects --------------------------------------------------------------

### ivf / exhaustive -----------------------------------------------------------

#' Generate GPU kNN data for single cells (exhaustive / IVF)
#'
#' @description
#' This function generates a `SingleCellNearestNeighbour` object using
#' GPU-accelerated kNN algorithms via the `bixverse.gpu` package. Two methods
#' are available: `"exhaustive"` performs an exact brute-force search on the
#' GPU; `"ivf"` builds an inverted file index that partitions the embedding
#' space into Voronoi cells and probes only a subset at query time, trading a
#' small amount of precision for considerably faster search on larger data sets.
#' This function is the GPU counterpart of [generate_knn_sc()].
#'
#' @param object `SingleCells` class.
#' @param embd_to_use String. The embedding to use. Whichever you choose, it
#' needs to be part of the object.
#' @param cells_to_use Optional string vector. Cell names to include. If `NULL`
#' all cells in the object will be used.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param gpu_method String. One of `c("exhaustive", "ivf")`. `"exhaustive"`
#' computes exact nearest neighbours via brute-force on the GPU. `"ivf"` builds
#' an inverted file index for approximate search.
#' @param ivf_params List. Output of [bixverse.gpu::params_sc_ivf()]. Only
#' used when `gpu_method = "ivf"`. A list with the following items:
#' \itemize{
#'   \item k - Integer. Number of nearest neighbours to identify.
#'   \item ann_dist - String. Distance metric; one of
#'   `c("euclidean", "cosine")`.
#'   \item nlist - Optional integer. Number of clusters to partition the index
#'   into. Controls the granularity of the Voronoi partitioning. If `NULL`,
#'   defaults to `sqrt(n)` on the Rust side.
#'   \item nprobe - Optional integer. Number of clusters to probe at query
#'   time. Higher values improve recall at the cost of speed. If `NULL`,
#'   defaults to `sqrt(nlist)`.
#'   \item nquery - Optional integer. Number of query vectors processed per
#'   GPU batch. If `NULL`, defaults to 100,000.
#'   \item max_iters - Optional integer. Maximum k-means iterations during
#'   index construction. If `NULL`, defaults to 30.
#'   \item seed - Integer. Seed for k-means initialisation.
#' }
#' @param k Integer. Number of neighbours. Only used when
#' `gpu_method = "exhaustive"`.
#' @param dist_metric String. One of `c("euclidean", "cosine")`. Only used
#' when `gpu_method = "exhaustive"`.
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @return Initialised `sc_knn` with the kNN data.
#'
#' @export
generate_gpu_knn_sc <- S7::new_generic(
  name = "generate_gpu_knn_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    cells_to_use = NULL,
    no_embd_to_use = NULL,
    gpu_method = c("exhaustive", "ivf"),
    ivf_params = params_sc_ivf(),
    k = 15L,
    dist_metric = "cosine",
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method generate_gpu_knn_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(generate_gpu_knn_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  gpu_method = c("exhaustive", "ivf"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "cosine",
  seed = 42L,
  .verbose = TRUE
) {
  gpu_method <- match.arg(gpu_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(cells_to_use, c("S+", "0"))
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(dist_metric, "S1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (!embd_to_use %in% get_available_embeddings(object)) {
    warning("The desired embedding was not found. Returning NULL.")
    return(NULL)
  }

  embd <- get_embedding(x = object, embd_name = embd_to_use)

  if (!is.null(cells_to_use)) {
    embd <- embd[which(rownames(embd) %in% cells_to_use), ]
  }

  if (!is.null(no_embd_to_use)) {
    embd <- embd[, 1:min(no_embd_to_use, ncol(embd))]
  }

  if (.verbose) {
    message(sprintf("Generating GPU kNN data with %s method.", gpu_method))
  }

  knn_raw <- switch(
    gpu_method,
    exhaustive = rs_exhaustive_gpu_knn(
      embd = embd,
      k = k,
      dist_metric = dist_metric,
      verbose = parse_verbosity(.verbose)
    ),
    ivf = rs_ivf_gpu_knn(
      embd = embd,
      ivf_params = ivf_params,
      seed = seed,
      verbose = parse_verbosity(.verbose)
    )
  )

  new_sc_knn(knn_data = knn_raw, used_cells = row.names(embd))
}

### cagra ----------------------------------------------------------------------

#' Generate CAGRA GPU kNN data for single cells
#'
#' @description
#' This function generates a `SingleCellNearestNeighbour` object using the
#' CAGRA (CUDA-Accelerated Graph Retrieval Approximation) algorithm via the
#' `bixverse.gpu` package. CAGRA first builds a dense NNDescent graph, then
#' prunes it into a sparser navigational graph optimised for beam-search
#' traversal. Two retrieval modes are available: direct extraction from the
#' NNDescent graph (`extract_knn = TRUE`), which is faster but slightly less
#' precise, or beam search over the pruned CAGRA graph (`extract_knn = FALSE`),
#' which is slower but yields higher recall. This function is the CAGRA
#' counterpart of [generate_knn_sc()].
#'
#' @param object `SingleCells` class.
#' @param embd_to_use String. The embedding to use. Whichever you choose, it
#' needs to be part of the object.
#' @param cells_to_use Optional string vector. Cell names to include. If `NULL`
#' all cells in the object will be used.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param cagra_params List. Output of [bixverse.gpu::params_sc_cagra()]. A
#' list with the following items:
#' \itemize{
#'   \item k_query - Integer. Number of nearest neighbours to return in the
#'   final result.
#'   \item ann_dist - String. Distance metric; one of
#'   `c("euclidean", "cosine")`.
#'   \item k - Optional integer. Final node degree of the pruned CAGRA
#'   navigational graph. Controls the sparsity of the search graph; higher
#'   values improve recall but increase memory usage. If `NULL`, defaults to
#'   `30`.
#'   \item k_build - Optional integer. Number of neighbours during the
#'   NNDescent build phase before CAGRA pruning. If `NULL`, defaults to
#'   `1.5 * k`.
#'   \item refine_sweeps - Integer. Number of refinement sweeps during graph
#'   construction. More sweeps improve graph quality at the cost of build time.
#'   \item max_iters - Optional integer. Maximum iterations for the NNDescent
#'   rounds. If `NULL`, determined automatically.
#'   \item n_trees - Optional integer. Number of trees in the initial
#'   GPU-accelerated random projection forest used to seed NNDescent. If
#'   `NULL`, determined automatically.
#'   \item delta - Numeric. Early-stopping criterion for NNDescent; iterations
#'   terminate when fewer than `delta` fraction of neighbours change.
#'   \item rho - Optional numeric. Sampling rate during NNDescent iterations.
#'   Lower values speed up construction at the cost of graph quality. If
#'   `NULL`, determined automatically.
#'   \item beam_width - Optional integer. Beam width during graph search.
#'   Larger beams improve recall but slow down querying. If `NULL`, determined
#'   automatically.
#'   \item max_beam_iters - Optional integer. Maximum beam search iterations.
#'   If `NULL`, determined automatically.
#'   \item n_entry_points - Optional integer. Number of entry points into the
#'   CAGRA graph. If `NULL`, determined automatically.
#' }
#' @param extract_knn Logical. If `TRUE`, extracts the kNN graph directly from
#' the NNDescent result (faster, slightly lower precision). If `FALSE`, runs
#' beam search over the pruned CAGRA graph (slower, higher precision).
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @return Initialised `sc_knn` with the kNN data.
#'
#' @export
generate_cagra_knn_sc <- S7::new_generic(
  name = "generate_cagra_knn_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    cells_to_use = NULL,
    no_embd_to_use = NULL,
    cagra_params = params_sc_cagra(),
    extract_knn = TRUE,
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method generate_cagra_knn_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(generate_cagra_knn_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  cagra_params = params_sc_cagra(),
  extract_knn = TRUE,
  seed = 42L,
  .verbose = TRUE
) {
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(cells_to_use, c("S+", "0"))
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(extract_knn, "B1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (!embd_to_use %in% get_available_embeddings(object)) {
    warning("The desired embedding was not found. Returning NULL.")
    return(NULL)
  }

  embd <- get_embedding(x = object, embd_name = embd_to_use)

  if (!is.null(cells_to_use)) {
    embd <- embd[which(rownames(embd) %in% cells_to_use), ]
  }

  if (!is.null(no_embd_to_use)) {
    embd <- embd[, 1:min(no_embd_to_use, ncol(embd))]
  }

  if (.verbose) {
    message("Generating GPU kNN data with CAGRA method.")
  }

  knn_raw <- rs_cagra_gpu_knn(
    embd = embd,
    cagra_params = cagra_params,
    extract_knn = extract_knn,
    seed = seed,
    verbose = parse_verbosity(.verbose)
  )

  new_sc_knn(knn_data = knn_raw, used_cells = row.names(embd))
}

## find neighbours (GPU) -------------------------------------------------------

### exhaustive / ivf -----------------------------------------------------------

#' Find GPU-accelerated neighbours for single cells (exhaustive / IVF)
#'
#' @description
#' This function generates kNN data using GPU-accelerated algorithms via the
#' `bixverse.gpu` package. Two methods are available: `"exhaustive"` performs
#' an exact brute-force search on the GPU, which is precise but scales
#' quadratically; `"ivf"` builds an inverted file index that partitions the
#' embedding space into Voronoi cells and probes only a subset at query time,
#' trading a small amount of precision for considerably faster search on larger
#' data sets. Subsequently, the kNN data is used to generate an sNN igraph for
#' downstream clustering. This function lives in a separate package from the
#' CPU-based [find_neighbours_sc()] so that users without GPU hardware do not
#' need to install the GPU dependencies.
#'
#' @param object `SingleCells` class.
#' @param embd_to_use String. The embedding to use. Whichever you choose, it
#' needs to be part of the object.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param gpu_method String. One of `c("exhaustive", "ivf")`. `"exhaustive"`
#' computes exact nearest neighbours via brute-force on the GPU. `"ivf"` builds
#' an inverted file index for approximate search.
#' @param ivf_params List. Output of [bixverse.gpu::params_sc_ivf()]. Only
#' used when `gpu_method = "ivf"`. A list with the following items:
#' \itemize{
#'   \item k - Integer. Number of nearest neighbours to identify.
#'   \item ann_dist - String. Distance metric; one of
#'   `c("euclidean", "cosine")`.
#'   \item nlist - Optional integer. Number of clusters to partition the index
#'   into. Controls the granularity of the Voronoi partitioning. If `NULL`,
#'   defaults to `sqrt(n)` on the Rust side.
#'   \item nprobe - Optional integer. Number of clusters to probe at query
#'   time. Higher values improve recall at the cost of speed. If `NULL`,
#'   defaults to `sqrt(nlist)`.
#'   \item nquery - Optional integer. Number of query vectors processed per
#'   GPU batch. If `NULL`, defaults to 100,000.
#'   \item max_iters - Optional integer. Maximum k-means iterations during
#'   index construction. If `NULL`, defaults to 30.
#'   \item seed - Integer. Seed for k-means initialisation.
#' }
#' @param k Integer. Number of neighbours. Only used when
#' `gpu_method = "exhaustive"`.
#' @param dist_metric String. One of `c("euclidean", "cosine")`. Only used
#' when `gpu_method = "exhaustive"`.
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()].
#' Controls sNN graph construction. The relevant items are:
#' \itemize{
#'   \item full_snn - Boolean. Whether to generate edges between all cells
#'   rather than only between neighbours.
#'   \item pruning - Numeric. Weights below this threshold are set to 0 in
#'   the sNN graph.
#'   \item snn_similarity - String. One of `c("rank", "jaccard")`. Defines
#'   how the sNN edge weights are calculated.
#' }
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return The object with added kNN matrix and sNN graph.
#'
#' @export
find_neighbours_gpu_sc <- S7::new_generic(
  name = "find_neighbours_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    gpu_method = c("exhaustive", "ivf"),
    ivf_params = params_sc_ivf(),
    k = 15L,
    dist_metric = "cosine",
    snn_params = params_sc_neighbours(),
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method find_neighbours_gpu_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(find_neighbours_gpu_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  gpu_method = c("exhaustive", "ivf"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "cosine",
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
) {
  gpu_method <- match.arg(gpu_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(dist_metric, "S1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (!embd_to_use %in% get_available_embeddings(object)) {
    warning("The desired embedding was not found. Returning class as is.")
    return(object)
  }

  knn_data <- generate_gpu_knn_sc(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    gpu_method = gpu_method,
    ivf_params = ivf_params,
    k = k,
    dist_metric = dist_metric,
    seed = seed,
    .verbose = .verbose
  )
  object <- set_knn(object, knn_data)

  if (.verbose) {
    message(sprintf("Generating sNN graph (full: %s).", snn_params$full_snn))
  }
  snn_graph_rs <- with(
    snn_params,
    rs_sc_snn(
      knn_mat = get_knn_mat(knn_data),
      snn_method = snn_similarity,
      pruning = pruning,
      limited_graph = !full_snn,
      verbose = parse_verbosity(.verbose)
    )
  )

  if (.verbose) {
    message("Transforming sNN data to igraph.")
  }
  snn_g <- igraph::make_empty_graph(
    n = nrow(get_knn_mat(knn_data)),
    directed = FALSE
  )
  snn_g <- igraph::add_edges(
    snn_g,
    snn_graph_rs$edges,
    attr = list(weight = snn_graph_rs$weights)
  )

  object <- set_snn_graph(object, snn_graph = snn_g)

  return(object)
}

# find_neighbours_cagra_sc -----------------------------------------------------

#' Find CAGRA GPU-accelerated neighbours for single cells
#'
#' @description
#' This function generates kNN data using the CAGRA (CUDA-Accelerated Graph
#' Retrieval Approximation) algorithm on the wgpu backend via the `bixverse.gpu`
#' package. CAGRA first builds a dense NNDescent graph, then prunes it into a
#' sparser navigational graph optimised for beam-search traversal. Two retrieval
#' modes are available: direct extraction from the NNDescent graph
#' (`extract_knn = TRUE`), which is faster but slightly less precise, or beam
#' search over the pruned CAGRA graph (`extract_knn = FALSE`), which is slower
#' but yields higher recall. CAGRA tends to perform well on high-dimensional
#' embeddings and very large data sets. Subsequently, the kNN data is used to
#' generate an sNN igraph for downstream clustering. As with
#' [find_neighbours_gpu_sc()], this function lives in a separate
#' package so that users without GPU hardware are not required to install the
#' GPU dependencies.
#'
#' @param object `SingleCells` class.
#' @param embd_to_use String. The embedding to use. Whichever you choose, it
#' needs to be part of the object.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param cagra_params List. Output of [bixverse.gpu::params_sc_cagra()]. A
#' list with the following items:
#' \itemize{
#'   \item k_query - Integer. Number of nearest neighbours to return in the
#'   final result.
#'   \item ann_dist - String. Distance metric; one of
#'   `c("euclidean", "cosine")`.
#'   \item k - Optional integer. Final node degree of the pruned CAGRA
#'   navigational graph. Controls the sparsity of the search graph; higher
#'   values improve recall but increase memory usage. If `NULL`, defaults to
#'   `30`.
#'   \item k_build - Optional integer. Number of neighbours during the
#'   NNDescent build phase before CAGRA pruning. If `NULL`, defaults to
#'   `1.5 * k`.
#'   \item refine_sweeps - Integer. Number of refinement sweeps during graph
#'   construction. More sweeps improve graph quality at the cost of build time.
#'   \item max_iters - Optional integer. Maximum iterations for the NNDescent
#'   rounds. If `NULL`, determined automatically.
#'   \item n_trees - Optional integer. Number of trees in the initial
#'   GPU-accelerated random projection forest used to seed NNDescent. If
#'   `NULL`, determined automatically.
#'   \item delta - Numeric. Early-stopping criterion for NNDescent; iterations
#'   terminate when fewer than `delta` fraction of neighbours change.
#'   \item rho - Optional numeric. Sampling rate during NNDescent iterations.
#'   Lower values speed up construction at the cost of graph quality. If
#'   `NULL`, determined automatically.
#'   \item beam_width - Optional integer. Beam width during graph search.
#'   Larger beams improve recall but slow down querying. If `NULL`, determined
#'   automatically.
#'   \item max_beam_iters - Optional integer. Maximum beam search iterations.
#'   If `NULL`, determined automatically.
#'   \item n_entry_points - Optional integer. Number of entry points into the
#'   CAGRA graph. If `NULL`, determined automatically.
#' }
#' @param extract_knn Logical. If `TRUE`, extracts the kNN graph directly from
#' the NNDescent result (faster, slightly lower precision). If `FALSE`, runs
#' beam search over the pruned CAGRA graph (slower, higher precision).
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()].
#' Controls sNN graph construction. The relevant items are:
#' \itemize{
#'   \item full_snn - Boolean. Whether to generate edges between all cells
#'   rather than only between neighbours.
#'   \item pruning - Numeric. Weights below this threshold are set to 0 in
#'   the sNN graph.
#'   \item snn_similarity - String. One of `c("rank", "jaccard")`. Defines
#'   how the sNN edge weights are calculated.
#' }
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return The object with added kNN matrix and sNN graph.
#'
#' @export
find_neighbours_cagra_sc <- S7::new_generic(
  name = "find_neighbours_cagra_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    cagra_params = params_sc_cagra(),
    extract_knn = TRUE,
    snn_params = params_sc_neighbours(),
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method find_neighbours_cagra_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(find_neighbours_cagra_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  cagra_params = params_sc_cagra(),
  extract_knn = TRUE,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
) {
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(extract_knn, "B1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (!embd_to_use %in% get_available_embeddings(object)) {
    warning("The desired embedding was not found. Returning class as is.")
    return(object)
  }

  knn_data <- generate_cagra_knn_sc(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    cagra_params = cagra_params,
    extract_knn = extract_knn,
    seed = seed,
    .verbose = .verbose
  )
  object <- set_knn(object, knn_data)

  if (.verbose) {
    message(sprintf("Generating sNN graph (full: %s).", snn_params$full_snn))
  }
  snn_graph_rs <- with(
    snn_params,
    rs_sc_snn(
      knn_mat = get_knn_mat(knn_data),
      snn_method = snn_similarity,
      pruning = pruning,
      limited_graph = !full_snn,
      verbose = parse_verbosity(.verbose)
    )
  )

  if (.verbose) {
    message("Transforming sNN data to igraph.")
  }
  snn_g <- igraph::make_empty_graph(
    n = nrow(get_knn_mat(knn_data)),
    directed = FALSE
  )
  snn_g <- igraph::add_edges(
    snn_g,
    snn_graph_rs$edges,
    attr = list(weight = snn_graph_rs$weights)
  )

  object <- set_snn_graph(object, snn_graph = snn_g)

  return(object)
}
