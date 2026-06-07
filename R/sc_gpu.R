# ------------------------------------------------------------------------------
# GPU-accelerated single cell workflows:
# - Has GPU-accelerated kNN searches which are split into two core versions:
#   CAGRA and the IVF/exhaustive versions
# - GPU-accelerated sparse, randomised SVD. Leverages GPU-accelerated math
#   multiplications to accelerate that part.
# - GPU-accelerated version of Harmony (version 2 with Arrowhead)
# ------------------------------------------------------------------------------

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
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use. Whichever you choose, it
#' needs to be part of the object for the selected modality.
#' @param cells_to_use Optional string vector. Cell names to include. If `NULL`
#' all cells in the object will be used.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param modality String. One of `c("rna", "adt")`. You can only use `"adt"`
#' on `SingleCellsMultiModal` class.
#' @param gpu_method String. One of `c("exhaustive", "ivf")`.
#' @param ivf_params List. Output of [bixverse.gpu::params_sc_ivf()]. Only
#' used when `gpu_method = "ivf"`.
#' @param k Integer. Number of neighbours. Only used when
#' `gpu_method = "exhaustive"`.
#' @param dist_metric String. One of `c("euclidean", "cosine")`. Only used
#' when `gpu_method = "exhaustive"`.
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity.
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
    modality = c("rna", "adt"),
    gpu_method = c("ivf", "exhaustive"),
    ivf_params = params_sc_ivf(),
    k = 15L,
    dist_metric = "euclidean",
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
  modality = c("rna", "adt"),
  gpu_method = c("ivf", "exhaustive"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "euclidean",
  seed = 42L,
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  gpu_method <- match.arg(gpu_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(cells_to_use, c("S+", "0"))
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(dist_metric, "S1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  if (!embd_to_use %in% get_available_embeddings(object, modality = modality)) {
    warning("The desired embedding was not found. Returning NULL.")
    return(NULL)
  }

  embd <- get_embedding(
    x = object,
    embd_name = embd_to_use,
    modality = modality
  )

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
      verbose = bixverse:::parse_verbosity(.verbose)
    ),
    ivf = rs_ivf_gpu_knn(
      embd = embd,
      ivf_params = ivf_params,
      seed = seed,
      verbose = bixverse:::parse_verbosity(.verbose)
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
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use. Whichever you choose, it
#' needs to be part of the object for the selected modality.
#' @param cells_to_use Optional string vector. Cell names to include. If `NULL`
#' all cells in the object will be used.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param modality String. One of `c("rna", "adt")`. You can only use `"adt"`
#' on `SingleCellsMultiModal` class.
#' @param cagra_params List. Output of [bixverse.gpu::params_sc_cagra()].
#' @param extract_knn Logical. If `TRUE`, extracts the kNN graph directly from
#' the NNDescent result (faster, slightly lower precision). If `FALSE`, runs
#' beam search over the pruned CAGRA graph (slower, higher precision).
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity.
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
    modality = c("rna", "adt"),
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
  modality = c("rna", "adt"),
  cagra_params = params_sc_cagra(),
  extract_knn = TRUE,
  seed = 42L,
  .verbose = TRUE
) {
  modality <- match.arg(modality)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(cells_to_use, c("S+", "0"))
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(extract_knn, "B1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  if (!embd_to_use %in% get_available_embeddings(object, modality = modality)) {
    warning("The desired embedding was not found. Returning NULL.")
    return(NULL)
  }

  embd <- get_embedding(
    x = object,
    embd_name = embd_to_use,
    modality = modality
  )

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
    verbose = bixverse:::parse_verbosity(.verbose)
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
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param modality String. One of `c("rna", "adt")`. You can only use `"adt"`
#' on `SingleCellsMultiModal` class.
#' @param gpu_method String. One of `c("exhaustive", "ivf")`.
#' @param ivf_params List. Output of [bixverse.gpu::params_sc_ivf()]. Only
#' used when `gpu_method = "ivf"`.
#' @param k Integer. Number of neighbours. Only used when
#' `gpu_method = "exhaustive"`.
#' @param dist_metric String. One of `c("euclidean", "cosine")`. Only used
#' when `gpu_method = "exhaustive"`.
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()].
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return The object with added kNN matrix and sNN graph in the selected
#' modality slot.
#'
#' @export
find_neighbours_gpu_sc <- S7::new_generic(
  name = "find_neighbours_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    modality = c("rna", "adt"),
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
  modality = c("rna", "adt"),
  gpu_method = c("exhaustive", "ivf"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "cosine",
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  gpu_method <- match.arg(gpu_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(dist_metric, "S1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  if (!embd_to_use %in% get_available_embeddings(object, modality = modality)) {
    warning("The desired embedding was not found. Returning class as is.")
    return(object)
  }

  knn_data <- generate_gpu_knn_sc(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    modality = modality,
    gpu_method = gpu_method,
    ivf_params = ivf_params,
    k = k,
    dist_metric = dist_metric,
    seed = seed,
    .verbose = .verbose
  )
  object <- set_knn(object, knn_data, modality = modality)

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
      verbose = bixverse:::parse_verbosity(.verbose)
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

  object <- set_snn_graph(object, snn_graph = snn_g, modality = modality)

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
#' but yields higher recall. Subsequently, the kNN data is used to generate an
#' sNN igraph for downstream clustering.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param modality String. One of `c("rna", "adt")`. You can only use `"adt"`
#' on `SingleCellsMultiModal` class.
#' @param cagra_params List. Output of [bixverse.gpu::params_sc_cagra()].
#' @param extract_knn Logical. If `TRUE`, extracts the kNN graph directly from
#' the NNDescent result. If `FALSE`, runs beam search over the pruned CAGRA
#' graph.
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()].
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return The object with added kNN matrix and sNN graph in the selected
#' modality slot.
#'
#' @export
find_neighbours_cagra_sc <- S7::new_generic(
  name = "find_neighbours_cagra_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    modality = c("rna", "adt"),
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
  modality = c("rna", "adt"),
  cagra_params = params_sc_cagra(),
  extract_knn = TRUE,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
) {
  modality <- match.arg(modality)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(extract_knn, "B1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  if (!embd_to_use %in% get_available_embeddings(object, modality = modality)) {
    warning("The desired embedding was not found. Returning class as is.")
    return(object)
  }

  knn_data <- generate_cagra_knn_sc(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    modality = modality,
    cagra_params = cagra_params,
    extract_knn = extract_knn,
    seed = seed,
    .verbose = .verbose
  )
  object <- set_knn(object, knn_data, modality = modality)

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
      verbose = bixverse:::parse_verbosity(.verbose)
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

  object <- set_snn_graph(object, snn_graph = snn_g, modality = modality)

  return(object)
}

# pca --------------------------------------------------------------------------

## gpu-accelerated sparse, randomised svd --------------------------------------

#' GPU-accelerated PCA for single cell
#'
#' @description
#' This function will run sparse, randomised SVD while running several of the
#' large matrix multiplications on GPU for improved speed. This also means you
#' will have to provide the necessary VRAM for your data set. This version only
#' works on the `"rna"` modality.
#'
#' @param object `SingleCells` class
#' @param no_pcs Integer. Number of PCs to calculate.
#' @param hvg Optional integer. If you want to provide your own HVG genes.
#' Otherwise, the function will default to what is found in
#' [bixverse::get_hvg()]. Please provide 1-indexed genes here! If you provide
#' these, the internal HVG will be overwritten.
#' @param seed Integer. Controls reproducibility. Only relevant if
#' `randomised_svd = TRUE`.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @return The function will add the PCA factors, loadings and singular values
#' to the object cache in memory.
#'
#' @export
calculate_pca_gpu_sc <- S7::new_generic(
  name = "calculate_pca_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    no_pcs,
    hvg = NULL,
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method calculate_pca_gpu_sc SingleCells
#'
#' @importFrom zeallot `%<-%`
#' @importFrom magrittr `%>%`
S7::method(calculate_pca_gpu_sc, SingleCells) <- function(
  object,
  no_pcs,
  hvg = NULL,
  seed = 42L,
  .verbose = TRUE
) {
  checkmate::assertClass(object, "bixverse::SingleCells")
  checkmate::qassert(no_pcs, "I1")
  checkmate::qassert(hvg, c("I+", "0"))
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if ((length(get_hvg(object)) == 0) && is.null(hvg)) {
    warning(paste(
      "No HVGs identified in the object nor provided.",
      "Please run find_hvg_sc() or provide the indices of the HVG",
      "Returning object as is."
    ))
    return(object)
  }

  selected_hvg <- if (!is.null(hvg)) {
    if (.verbose) {
      message(
        paste(
          "HVGs provided.",
          "Will use these ones and set the internal HVG to the provided genes."
        )
      )
    }
    object <- set_hvg(object, hvg) # this one deals with zero/one indexing internally
    hvg - 1L
  } else {
    get_hvg(object)
  }

  if (.verbose) {
    message(
      sprintf(
        "Using GPU-accelerated, randomised sparse SVD data with %i HVG.",
        length(selected_hvg)
      )
    )
  }

  zeallot::`%<-%`(
    c(pca_factors, pca_loadings, singular_values),
    rs_sc_pca_sparse_gpu(
      f_path_gene = bixverse:::get_rust_count_gene_f_path(object),
      no_pcs = no_pcs,
      cell_indices = get_cells_to_keep(object),
      gene_indices = selected_hvg,
      seed = seed,
      verbose = bixverse:::parse_verbosity(.verbose)
    )
  )

  object <- set_pca_factors(object, pca_factors)
  object <- set_pca_loadings(object, pca_loadings)
  object <- set_pca_singular_vals(object, singular_values[1:no_pcs])

  return(object)
}

# gpu harmony ------------------------------------------------------------------

#' Run Harmony v2 (GPU)
#'
#' @description
#' A GPU-accelerated version of Harmony v2 by Patikas et al., 2026,
#' implemented in Rust. Performs batch correction on PCA embeddings and stores
#' the result as a `"harmony_gpu"` embedding in the object. Only a single
#' batch covariate is supported on the GPU path.
#'
#' @param object `SingleCells` class.
#' @param batch_column String. Column name in the object containing the batch
#' labels.
#' @param modality String. One of `c("rna", "adt")`. You can only use `"adt"`
#' on `SingleCellsMultiModal` class.
#' @param harmony_params List. Output of [params_sc_harmony_v2_gpu()].
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @return The object with a `"harmony_gpu"` embedding added. If no PCA
#' embeddings are found, returns the object unchanged with a warning.
#'
#' @export
harmony_v2_gpu_sc <- S7::new_generic(
  name = "harmony_v2_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    batch_column,
    modality = c("rna", "adt"),
    harmony_params = params_sc_harmony_v2_gpu(),
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method harmony_v2_gpu_sc SingleCells
#'
#' @export
S7::method(harmony_v2_gpu_sc, SingleCells) <- function(
  object,
  batch_column,
  modality = c("rna", "adt"),
  harmony_params = params_sc_harmony_v2_gpu(),
  seed = 42L,
  .verbose = TRUE
) {
  modality <- match.arg(modality)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(batch_column, "S1")
  assertScHarmonyParamsV2Gpu(harmony_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  if (is.null(get_pca_factors(object, modality = modality))) {
    warning("No PCA embeddings found in the object. Returning class as is")
    return(object)
  } else {
    pca_data <- get_pca_factors(object, modality = modality)
  }

  batch_indices <- unlist(object[[batch_column]])
  batch_factor <- factor(batch_indices)
  batch_indices <- as.integer(batch_factor) - 1L

  checkmate::assertTRUE(length(batch_indices) == nrow(pca_data))

  if (is.null(harmony_params$k)) {
    harmony_params$k <- as.integer(min(round(nrow(pca_data) / 30), 100L))
    if (.verbose) {
      message(sprintf(
        " Auto-determined number of Harmony clusters: %d",
        harmony_params$k
      ))
    }
  }

  harmony_embd <- rs_harmony_v2_gpu(
    pca = pca_data,
    harmony_params = harmony_params,
    batch_labels = list(batch_indices),
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  colnames(harmony_embd) <- sprintf("harmony_gpu_%s", 1:ncol(harmony_embd))

  object <- set_embedding(
    x = object,
    embd = harmony_embd,
    name = "harmony_gpu",
    modality = modality
  )

  return(object)
}
