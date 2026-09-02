# ------------------------------------------------------------------------------
# GPU-accelerated single cell workflows:
# - GPU-accelerated kNN searches (exhaustive, IVF, CAGRA) behind one generic
# - GPU-accelerated sparse, randomised SVD. Leverages GPU-accelerated math
#   multiplications to accelerate that part.
# - GPU-accelerated version of Harmony (version 2 with Arrowhead)
# - GPU-accelerated UMAP (kNN + Adam optimiser) and t-SNE (kNN only)
# ------------------------------------------------------------------------------

# knn searches -----------------------------------------------------------------

## helpers ---------------------------------------------------------------------

#' Pull an embedding out of a single cell object for a GPU kNN search
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use.
#' @param cells_to_use Optional string vector. Cell names to include.
#' @param no_embd_to_use Optional integer. Number of dimensions to keep.
#' @param modality String. One of `c("rna", "adt")`.
#'
#' @return The embedding matrix, or `NULL` if the embedding is not in the
#' object.
#'
#' @keywords internal
.gpu_knn_embedding <- function(
  object,
  embd_to_use,
  cells_to_use,
  no_embd_to_use,
  modality
) {
  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  if (!embd_to_use %in% get_available_embeddings(object, modality = modality)) {
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

  embd
}

#' Build an sNN igraph from kNN data and attach it to a single cell object
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param knn_data Initialised `sc_knn` with the kNN data.
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()].
#' @param modality String. One of `c("rna", "adt")`.
#' @param .verbose Boolean or integer. Controls verbosity.
#'
#' @return The object with the sNN graph in the selected modality slot.
#'
#' @keywords internal
.set_snn_graph_gpu <- function(
  object,
  knn_data,
  snn_params,
  modality,
  .verbose
) {
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

  set_snn_graph(
    object,
    snn_graph = snn_g,
    modality = modality,
    from = "knn"
  )
}

#' Fold the deprecated kNN arguments into the current ones
#'
#' @param knn_method String. Current argument.
#' @param nn_params List. Current argument.
#' @param k Integer. Current argument.
#' @param gpu_method Deprecated. Superseded by `knn_method`.
#' @param ivf_params Deprecated. Superseded by `nn_params`.
#' @param dist_metric Deprecated. Superseded by `params_nn_gpu(dist_metric)`.
#' @param fn String. Name of the calling function, for the warning text.
#'
#' @return A list with the resolved `knn_method`, `nn_params` and `k`.
#'
#' @keywords internal
#' @importFrom lifecycle deprecate_warn
.resolve_deprecated_knn_args <- function(
  knn_method,
  nn_params,
  k,
  gpu_method,
  ivf_params,
  dist_metric,
  fn
) {
  if (lifecycle::is_present(gpu_method)) {
    lifecycle::deprecate_warn(
      when = "0.4.0",
      what = sprintf("%s(gpu_method)", fn),
      with = sprintf("%s(knn_method)", fn)
    )
    knn_method <- gpu_method
  }

  if (lifecycle::is_present(ivf_params)) {
    lifecycle::deprecate_warn(
      when = "0.4.0",
      what = sprintf("%s(ivf_params)", fn),
      with = sprintf("%s(nn_params)", fn)
    )
    # the old IVF wrapper carried k on the list, the new one takes it as an
    # argument
    if (!is.null(ivf_params$k)) {
      k <- as.integer(ivf_params$k)
      ivf_params$k <- NULL
    }
    nn_params <- ivf_params
  }

  if (lifecycle::is_present(dist_metric)) {
    lifecycle::deprecate_warn(
      when = "0.4.0",
      what = sprintf("%s(dist_metric)", fn),
      with = "params_nn_gpu(dist_metric = )"
    )
    nn_params$dist_metric <- dist_metric
  }

  list(knn_method = knn_method, nn_params = nn_params, k = k)
}

## to knn objects --------------------------------------------------------------

#' Generate GPU kNN data for single cells
#'
#' @description
#' This function generates a `SingleCellNearestNeighbour` object using
#' GPU-accelerated kNN algorithms via the `bixverse.gpu` package. Three methods
#' are available: `"exhaustive"` performs an exact brute-force search on the
#' GPU; `"ivf"` builds an inverted file index that partitions the embedding
#' space into Voronoi cells and probes only a subset at query time; and
#' `"nndescent"` builds a dense NNDescent graph and prunes it into a CAGRA
#' navigational graph, which is then either beam searched or handed back as the
#' descent left it (`params_nn_gpu(extract_knn = TRUE)`, faster, lower recall).
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
#' @param knn_method String. One of `c("nndescent", "exhaustive", "ivf")`.
#' @param nn_params List. Output of [bixverse.gpu::params_nn_gpu()].
#' @param k Integer. Number of neighbours.
#' @param seed Integer. For reproducibility.
#' @param gpu_method `r lifecycle::badge("deprecated")` Use `knn_method`.
#' @param ivf_params `r lifecycle::badge("deprecated")` Use `nn_params`.
#' @param dist_metric `r lifecycle::badge("deprecated")` Use
#' `params_nn_gpu(dist_metric = )`.
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
    knn_method = c("nndescent", "exhaustive", "ivf"),
    nn_params = params_nn_gpu(),
    k = 15L,
    seed = 42L,
    gpu_method = lifecycle::deprecated(),
    ivf_params = lifecycle::deprecated(),
    dist_metric = lifecycle::deprecated(),
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
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  k = 15L,
  seed = 42L,
  gpu_method = lifecycle::deprecated(),
  ivf_params = lifecycle::deprecated(),
  dist_metric = lifecycle::deprecated(),
  .verbose = TRUE
) {
  modality <- match.arg(modality)

  resolved <- .resolve_deprecated_knn_args(
    knn_method = knn_method,
    nn_params = nn_params,
    k = k,
    gpu_method = gpu_method,
    ivf_params = ivf_params,
    dist_metric = dist_metric,
    fn = "generate_gpu_knn_sc"
  )
  knn_method <- match.arg(
    resolved$knn_method,
    c("nndescent", "exhaustive", "ivf")
  )
  nn_params <- resolved$nn_params
  k <- resolved$k

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(cells_to_use, c("S+", "0"))
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::assertChoice(modality, c("rna", "adt"))
  checkmate::assertChoice(knn_method, c("nndescent", "exhaustive", "ivf"))
  assertNnParamsGpu(nn_params)
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  embd <- .gpu_knn_embedding(
    object = object,
    embd_to_use = embd_to_use,
    cells_to_use = cells_to_use,
    no_embd_to_use = no_embd_to_use,
    modality = modality
  )

  if (is.null(embd)) {
    warning("The desired embedding was not found. Returning NULL.")
    return(NULL)
  }

  if (.verbose) {
    message(sprintf("Generating GPU kNN data with %s method.", knn_method))
  }

  knn_raw <- rs_gpu_knn(
    embd = embd,
    k = k,
    knn_method = knn_method,
    nn_params = nn_params,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  new_sc_knn(knn_data = knn_raw, used_cells = row.names(embd))
}

## find neighbours (GPU) -------------------------------------------------------

#' Find GPU-accelerated neighbours for single cells
#'
#' @description
#' This function generates kNN data using GPU-accelerated algorithms via the
#' `bixverse.gpu` package, then turns it into an sNN igraph for downstream
#' clustering. See [generate_gpu_knn_sc()] for the three searches on offer.
#' This function lives in a separate package from the CPU-based
#' [find_neighbours_sc()] so that users without GPU hardware do not need to
#' install the GPU dependencies.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param modality String. One of `c("rna", "adt")`. You can only use `"adt"`
#' on `SingleCellsMultiModal` class.
#' @param knn_method String. One of `c("nndescent", "exhaustive", "ivf")`.
#' @param nn_params List. Output of [bixverse.gpu::params_nn_gpu()].
#' @param k Integer. Number of neighbours.
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()]. The
#' kNN graph-related parameters will be ignored in favour of `nn_params`.
#' @param seed Integer. For reproducibility.
#' @param gpu_method `r lifecycle::badge("deprecated")` Use `knn_method`.
#' @param ivf_params `r lifecycle::badge("deprecated")` Use `nn_params`.
#' @param dist_metric `r lifecycle::badge("deprecated")` Use
#' `params_nn_gpu(dist_metric = )`.
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
    knn_method = c("nndescent", "exhaustive", "ivf"),
    nn_params = params_nn_gpu(),
    k = 15L,
    snn_params = params_sc_neighbours(),
    seed = 42L,
    gpu_method = lifecycle::deprecated(),
    ivf_params = lifecycle::deprecated(),
    dist_metric = lifecycle::deprecated(),
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
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  k = 15L,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  gpu_method = lifecycle::deprecated(),
  ivf_params = lifecycle::deprecated(),
  dist_metric = lifecycle::deprecated(),
  .verbose = TRUE
) {
  modality <- match.arg(modality)

  resolved <- .resolve_deprecated_knn_args(
    knn_method = knn_method,
    nn_params = nn_params,
    k = k,
    gpu_method = gpu_method,
    ivf_params = ivf_params,
    dist_metric = dist_metric,
    fn = "find_neighbours_gpu_sc"
  )
  knn_method <- match.arg(
    resolved$knn_method,
    c("nndescent", "exhaustive", "ivf")
  )
  nn_params <- resolved$nn_params
  k <- resolved$k

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::assertChoice(modality, c("rna", "adt"))
  checkmate::assertChoice(knn_method, c("nndescent", "exhaustive", "ivf"))
  assertNnParamsGpu(nn_params)
  checkmate::qassert(k, "I1[1,)")
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

  # hard tier: the kNN indices built here go straight to Rust downstream
  assert_sc_state(object, artefacts = embd_to_use, modality = modality)

  knn_data <- generate_gpu_knn_sc(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    modality = modality,
    knn_method = knn_method,
    nn_params = nn_params,
    k = k,
    seed = seed,
    .verbose = .verbose
  )
  object <- set_knn(
    object,
    knn_data,
    modality = modality,
    from = embd_to_use
  )

  .set_snn_graph_gpu(
    object = object,
    knn_data = knn_data,
    snn_params = snn_params,
    modality = modality,
    .verbose = .verbose
  )
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
#' @param pca_params Named list. Controls the parameters to be used for the
#' PCA calculation which is single cell-specific, see [params_sc_pca()].
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
    pca_params = bixverse::params_sc_pca(),
    hvg = NULL,
    seed = 42L,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method calculate_pca_gpu_sc SingleCells
#'
#' @importFrom zeallot %<-%
#' @importFrom magrittr %>%
S7::method(calculate_pca_gpu_sc, SingleCells) <- function(
  object,
  no_pcs,
  pca_params = bixverse::params_sc_pca(),
  hvg = NULL,
  seed = 42L,
  .verbose = TRUE
) {
  checkmate::assertClass(object, "bixverse::SingleCells")
  checkmate::qassert(no_pcs, "I1")
  checkmate::qassert(hvg, c("I+", "0"))
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if ((length(suppressWarnings(get_hvg(object))) == 0) && is.null(hvg)) {
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
    # this one deals with zero/one indexing internally
    object <- set_hvg(object, hvg)
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
      f_path_cell = bixverse:::get_rust_count_cell_f_path(object),
      no_pcs = no_pcs,
      pca_params = pca_params,
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

  # hard tier: the corrected embedding is written back onto the object
  assert_sc_state(object, artefacts = "pca", modality = modality)

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
    modality = modality,
    from = "pca"
  )

  return(object)
}

# gpu umap ---------------------------------------------------------------------

#' Run UMAP on a SingleCells object (GPU)
#'
#' @description
#' GPU-accelerated counterpart to [bixverse::umap_sc()]. Pulls an embedding
#' (defaulting to PCA) off the object, runs [umap_gpu()] on it (GPU kNN plus
#' GPU Adam optimiser by default), and writes the resulting embedding back
#' into `sc_cache$other_embeddings[[slot_name]]`.
#'
#' When `use_knn = TRUE` (the default), the kNN graph already stored on the
#' object is reused via [bixverse::sc_knn_to_nearest_neighbours()]. Otherwise
#' a fresh GPU kNN is built from the chosen embedding.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param use_knn Boolean. Use the kNN graph found in the object. Defaults to
#' `TRUE`. Only reused if the modality lines up; otherwise a fresh GPU kNN is
#' generated.
#' @param embd_to_use String. The embedding to use for UMAP. Must be available
#' in the object for the chosen modality.
#' @param slot_name String. The name of this embedding within the object.
#' Defaults to `"umap"`.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL`, all will be used.
#' @param modality String. On which modality to run UMAP. One of
#' `c("rna", "adt", "wnn")`. The two latter options are only available on
#' `SingleCellsMultiModal`.
#' @param n_dim Integer. Number of UMAP dimensions. Defaults to `2L`.
#' @param k Integer. Number of nearest neighbours. Defaults to `15L`.
#' @param min_dist Numeric. Minimum distance between embedded points. Defaults
#' to `0.5`.
#' @param spread Numeric. Effective scale of embedded points. Defaults to
#' `1.0`.
#' @param knn_method String. GPU (approximate) nearest neighbour method. One
#' of `c("nndescent", "exhaustive", "ivf")`.
#' @param nn_params Named list. GPU kNN parameters, see [params_nn_gpu()].
#' @param umap_params Named list. UMAP (GPU) parameters, see
#' [params_umap_gpu()].
#' @param seed Integer. For reproducibility.
#' @param use_high_precision Optional boolean. Fine-grained fp32 vs fp64
#' control for the optimiser. GPU kNN is always fp32.
#' @param .verbose Boolean or integer. Controls verbosity.
#'
#' @return The object with a `"umap"` embedding added. If the requested
#' embedding is missing, returns the object unchanged with a warning.
#'
#' @seealso [umap_gpu()], [bixverse::umap_sc()], [tsne_gpu_sc()]
#'
#' @export
#'
#' @import bixverse
umap_gpu_sc <- S7::new_generic(
  name = "umap_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    use_knn = TRUE,
    embd_to_use = "pca",
    slot_name = "umap",
    no_embd_to_use = NULL,
    modality = c("rna", "adt", "wnn"),
    n_dim = 2L,
    k = 15L,
    min_dist = 0.5,
    spread = 1.0,
    knn_method = c("nndescent", "exhaustive", "ivf"),
    nn_params = params_nn_gpu(),
    umap_params = params_umap_gpu(),
    seed = 42L,
    use_high_precision = NULL,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method umap_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(umap_gpu_sc, SingleCells) <- function(
  object,
  use_knn = TRUE,
  embd_to_use = "pca",
  slot_name = "umap",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  k = 15L,
  min_dist = 0.5,
  spread = 1.0,
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  knn_method <- match.arg(knn_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(use_knn, "B1")
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(slot_name, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(n_dim, "I1[1,)")
  checkmate::qassert(k, "I1[2,)")
  checkmate::qassert(min_dist, "N1[0,)")
  checkmate::qassert(spread, "N1[0,)")
  assertNnParamsGpu(nn_params)
  assertUmapParamsGpu(umap_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  cache_modality <- if (modality == "wnn") "rna" else modality

  # embedding
  available <- get_available_embeddings(object, modality = cache_modality)
  if (!(embd_to_use %in% available)) {
    warning(sprintf(
      "Embedding '%s' not found on the object. Returning object as is.",
      embd_to_use
    ))
    return(object)
  }

  # hard tier: the manifold is written back onto the object, and it is read
  # from `cache_modality` while the kNN comes from `modality`
  assert_sc_state(object, artefacts = embd_to_use, modality = cache_modality)
  if (modality == "wnn" || use_knn) {
    assert_sc_state(object, artefacts = "knn", modality = modality)
  }
  embd <- get_embedding(
    x = object,
    embd_name = embd_to_use,
    modality = cache_modality
  )
  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take]
  }

  # knn
  knn <- if (modality == "wnn") {
    bixverse:::.get_manifoldsr_knn_from_wnn(x = object)
  } else if (use_knn) {
    bixverse:::.get_manifoldsr_knn(x = object, modality = modality)
  } else {
    NULL
  }

  if (.verbose) {
    message("Running GPU UMAP.")
  }

  umap_embd <- umap_gpu(
    data = embd,
    knn = knn,
    n_dim = n_dim,
    k = k,
    min_dist = min_dist,
    spread = spread,
    knn_method = knn_method,
    nn_params = nn_params,
    umap_params = umap_params,
    seed = seed,
    use_high_precision = use_high_precision,
    .verbose = .verbose
  )

  rownames(umap_embd) <- rownames(embd)
  colnames(umap_embd) <- sprintf("umap_%s", seq_len(ncol(umap_embd)))

  object <- set_embedding(
    x = object,
    embd = umap_embd,
    name = slot_name,
    modality = modality,
    from = .manifold_from_gpu(
      embd_to_use = embd_to_use,
      cache_modality = cache_modality,
      modality = modality,
      has_knn = !is.null(knn)
    )
  )

  return(object)
}

# gpu tsne ---------------------------------------------------------------------

#' Run t-SNE on a SingleCells object (GPU)
#'
#' @description
#' GPU-accelerated counterpart to [bixverse::tsne_sc()]. Runs [tsne_gpu()] on
#' an embedding pulled from the object; only the kNN step is GPU-accelerated,
#' the optimiser still runs on CPU (a GPU optimiser is on the roadmap).
#'
#' t-SNE derives the number of neighbours from `perplexity` on the Rust side
#' (the usual `3 * perplexity` convention). To avoid a silent mismatch with
#' the cached kNN, `use_knn` defaults to `FALSE`: every call generates a
#' fresh GPU kNN sized to the requested perplexity. Handy for sweeping
#' perplexities since the kNN is cheap on GPU.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param use_knn Boolean. Defaults to `FALSE`. Set to `TRUE` to reuse the
#' cached kNN; only sensible when the stored `k` is at least
#' `3 * perplexity`.
#' @param embd_to_use String. The embedding to use for t-SNE. Must be
#' available in the object for the chosen modality.
#' @param slot_name String. The name of this embedding within the object.
#' Defaults to `"tsne"`.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL`, all will be used.
#' @param modality String. On which modality to run t-SNE. One of
#' `c("rna", "adt", "wnn")`. The two latter options are only available on
#' `SingleCellsMultiModal`.
#' @param n_dim Integer. Number of t-SNE dimensions. Currently only `2L` is
#' supported. Defaults to `2L`.
#' @param perplexity Numeric. Perplexity parameter. Typical values between 5
#' and 50. Defaults to `20.0`.
#' @param approx_type String. Approximation method. One of `"bh"`
#' (Barnes-Hut) or `"fft"`. Defaults to `"bh"`. `"fft"` is Unix-only.
#' @param knn_method String. GPU (approximate) nearest neighbour method. One
#' of `c("nndescent", "exhaustive", "ivf")`.
#' @param nn_params Named list. GPU kNN parameters, see [params_nn_gpu()].
#' @param tsne_params Named list. t-SNE (GPU) parameters, see
#' [params_tsne_gpu()].
#' @param seed Integer. For reproducibility.
#' @param use_high_precision Optional boolean. Fine-grained fp32 vs fp64
#' control. GPU kNN is always fp32.
#' @param .verbose Boolean or integer. Controls verbosity.
#'
#' @return The object with a `"tsne"` embedding added. If the requested
#' embedding is missing, returns the object unchanged with a warning.
#'
#' @seealso [tsne_gpu()], [bixverse::tsne_sc()], [umap_gpu_sc()]
#'
#' @export
#'
#' @import bixverse
tsne_gpu_sc <- S7::new_generic(
  name = "tsne_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    use_knn = FALSE,
    embd_to_use = "pca",
    slot_name = "tsne",
    no_embd_to_use = NULL,
    modality = c("rna", "adt", "wnn"),
    n_dim = 2L,
    perplexity = 20.0,
    approx_type = c("bh", "fft"),
    knn_method = c("nndescent", "exhaustive", "ivf"),
    nn_params = params_nn_gpu(),
    tsne_params = params_tsne_gpu(),
    seed = 42L,
    use_high_precision = NULL,
    .verbose = TRUE
  ) {
    S7::S7_dispatch()
  }
)

#' @method tsne_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(tsne_gpu_sc, SingleCells) <- function(
  object,
  use_knn = FALSE,
  embd_to_use = "pca",
  slot_name = "tsne",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  perplexity = 20.0,
  approx_type = c("bh", "fft"),
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  approx_type <- match.arg(approx_type)
  knn_method <- match.arg(knn_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(use_knn, "B1")
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(slot_name, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(n_dim, "I1[2,2]")
  checkmate::qassert(perplexity, "N1[1,)")
  assertNnParamsGpu(nn_params)
  assertTsneParamsGpu(tsne_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  cache_modality <- if (modality == "wnn") "rna" else modality

  # embedding
  available <- get_available_embeddings(object, modality = cache_modality)
  if (!(embd_to_use %in% available)) {
    warning(sprintf(
      "Embedding '%s' not found on the object. Returning object as is.",
      embd_to_use
    ))
    return(object)
  }

  # hard tier: the manifold is written back onto the object, and it is read
  # from `cache_modality` while the kNN comes from `modality`
  assert_sc_state(object, artefacts = embd_to_use, modality = cache_modality)
  if (modality == "wnn" || use_knn) {
    assert_sc_state(object, artefacts = "knn", modality = modality)
  }
  embd <- get_embedding(
    x = object,
    embd_name = embd_to_use,
    modality = cache_modality
  )
  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take]
  }

  # knn - default is regenerate on GPU so perplexity drives k
  knn <- if (modality == "wnn") {
    bixverse:::.get_manifoldsr_knn_from_wnn(x = object)
  } else if (use_knn) {
    bixverse:::.get_manifoldsr_knn(x = object, modality = modality)
  } else {
    NULL
  }

  if (.verbose) {
    message("Running GPU t-SNE.")
  }

  tsne_embd <- tsne_gpu(
    data = embd,
    knn = knn,
    n_dim = n_dim,
    perplexity = perplexity,
    approx_type = approx_type,
    knn_method = knn_method,
    nn_params = nn_params,
    tsne_params = tsne_params,
    seed = seed,
    use_high_precision = use_high_precision,
    .verbose = .verbose
  )

  rownames(tsne_embd) <- rownames(embd)
  colnames(tsne_embd) <- sprintf("tsne_%s", seq_len(ncol(tsne_embd)))

  object <- set_embedding(
    x = object,
    embd = tsne_embd,
    name = slot_name,
    modality = modality,
    from = .manifold_from_gpu(
      embd_to_use = embd_to_use,
      cache_modality = cache_modality,
      modality = modality,
      has_knn = !is.null(knn)
    )
  )

  return(object)
}
