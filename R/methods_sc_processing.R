# ------------------------------------------------------------------------------
# GPU-accelerated single cell processing: kNN searches, sparse randomised PCA,
# fast Louvain clustering and Scrublet doublet detection.
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
    assert_gpu()

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
  assertNnGpuParams(nn_params)
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
    assert_gpu()

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
  assertNnGpuParams(nn_params)
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
#' `svd_solver` is ignored: the GPU path is always randomised.
#' @param hvg Optional integer. If you want to provide your own HVG genes.
#' Otherwise, the function will default to what is found in
#' [bixverse::get_hvg()]. Please provide 1-indexed genes here! If you provide
#' these, the internal HVG will be overwritten.
#' @param seed Integer. Seed for the randomised SVD.
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
    assert_gpu()

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
  bixverse:::assertScPcaParams(pca_params)
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


# ------------------------------------------------------------------------------
# GPU-accelerated fast Louvain clustering:
# - Wraps `rs_fast_cluster_gpu` and `rs_fast_cluster_grid_gpu`.
# - Returns the same `SingleCellFastClusters` S3 object as the CPU
#   `bixverse::fast_cluster_sc()`, so all its getters and `add_sc_new_obs()`
#   work unchanged.
# - bixverse dispatches on its internal `ScOrScSubset` union. That union is not
#   exported, so the two exported classes get a method each and both delegate to
#   `.fast_cluster_gpu()`.
# ------------------------------------------------------------------------------

# fast clustering (gpu) --------------------------------------------------------

#' Run fast Louvain clustering on a SingleCells object (GPU)
#'
#' @description
#' GPU counterpart of [bixverse::fast_cluster_sc()]. Runs k-means on the chosen
#' embedding, builds a kNN graph on the centroids, applies Louvain clustering
#' and propagates the memberships back to the cells. Optionally runs a grid over
#' multiple seeds and returns stability statistics.
#'
#' Only the k-means coarsening runs on the WGPU backend. The centroid kNN, the
#' optional sNN pass and the Louvain runs stay on the CPU, so the speedup tracks
#' how much of the run k-means owns. That share grows with the cell count and
#' with `n_centroids`. There is no `km_type` argument: the GPU k-means is
#' full-batch Lloyd's and has no mini-batch path.
#'
#' @param object `SingleCells` or `SingleCellsSubset` class from `bixverse`.
#' @param embd_to_use String. Embedding name. Defaults to `"pca"`.
#' @param no_embd_to_use Optional integer. Number of dimensions to keep.
#' @param resolutions Numeric vector. Louvain resolutions.
#' @param n_centroids Optional integer. Number of k-means centroids. Defaults
#' to `sqrt(n_cells)` Rust-side if `NULL`. Clamped to `n_cells - 1`.
#' @param fc_params List. Output of [params_sc_fast_cluster_gpu()].
#' @param snn Boolean. Convert the centroid kNN to an sNN graph.
#' @param return_kmeans Boolean. Return the k-means assignments and centroids.
#' @param grid_search Boolean. Run the multi-seed grid version.
#' @param no_seeds Integer. Number of seeds to vary Louvain over. Must be at
#' least 2. Only used when `grid_search = TRUE`.
#' @param seed Integer. Seed for reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @returns `SingleCellFastClusters` S3 object with:
#' \describe{
#'   \item{memberships}{data.table with `cell_idx` and one column per
#'   resolution (`res_<value>`).}
#'   \item{stats}{data.table of grid statistics, or `NULL`.}
#'   \item{k_means_cluster}{Integer vector of k-means assignments, or `NULL`.}
#'   \item{centroids}{Numeric matrix of centroids, or `NULL`.}
#'   \item{resolutions}{Resolutions used.}
#' }
#' with `cell_indices` stored as an attribute (0-indexed).
#'
#' @export
fast_cluster_gpu_sc <- S7::new_generic(
  name = "fast_cluster_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    resolutions = c(2.0, 1.0, 0.5),
    n_centroids = NULL,
    fc_params = params_sc_fast_cluster_gpu(),
    snn = TRUE,
    return_kmeans = FALSE,
    grid_search = FALSE,
    no_seeds = 10L,
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method fast_cluster_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(fast_cluster_gpu_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  resolutions = c(2.0, 1.0, 0.5),
  n_centroids = NULL,
  fc_params = params_sc_fast_cluster_gpu(),
  snn = TRUE,
  return_kmeans = FALSE,
  grid_search = FALSE,
  no_seeds = 10L,
  seed = 42L,
  .verbose = TRUE
) {
  .fast_cluster_gpu(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    resolutions = resolutions,
    n_centroids = n_centroids,
    fc_params = fc_params,
    snn = snn,
    return_kmeans = return_kmeans,
    grid_search = grid_search,
    no_seeds = no_seeds,
    seed = seed,
    .verbose = .verbose
  )
}

## SingleCellsSubset -----------------------------------------------------------

#' @method fast_cluster_gpu_sc SingleCellsSubset
#'
#' @export
#'
#' @import bixverse
S7::method(fast_cluster_gpu_sc, SingleCellsSubset) <- function(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  resolutions = c(2.0, 1.0, 0.5),
  n_centroids = NULL,
  fc_params = params_sc_fast_cluster_gpu(),
  snn = TRUE,
  return_kmeans = FALSE,
  grid_search = FALSE,
  no_seeds = 10L,
  seed = 42L,
  .verbose = TRUE
) {
  .fast_cluster_gpu(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    resolutions = resolutions,
    n_centroids = n_centroids,
    fc_params = fc_params,
    snn = snn,
    return_kmeans = return_kmeans,
    grid_search = grid_search,
    no_seeds = no_seeds,
    seed = seed,
    .verbose = .verbose
  )
}

## implementation --------------------------------------------------------------

#' Shared implementation of the GPU fast Louvain clustering
#'
#' @description
#' Body behind both `fast_cluster_gpu_sc()` methods. Pulls the embedding off the
#' object, hands it to [rs_fast_cluster_gpu()] or [rs_fast_cluster_grid_gpu()]
#' and wraps the result into a `SingleCellFastClusters` S3 object.
#'
#' @inheritParams fast_cluster_gpu_sc
#'
#' @returns A `SingleCellFastClusters` S3 object.
#'
#' @keywords internal
.fast_cluster_gpu <- function(
  object,
  embd_to_use,
  no_embd_to_use,
  resolutions,
  n_centroids,
  fc_params,
  snn,
  return_kmeans,
  grid_search,
  no_seeds,
  seed,
  .verbose
) {
  # checks
  checkmate::assertTRUE(
    S7::S7_inherits(object, bixverse::SingleCells) ||
      S7::S7_inherits(object, bixverse::SingleCellsSubset)
  )
  assertScFastClusterGpuParams(fc_params)
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(resolutions, "N+")
  checkmate::qassert(n_centroids, c("I1", "0"))
  checkmate::qassert(snn, "B1")
  checkmate::qassert(return_kmeans, "B1")
  checkmate::qassert(grid_search, "B1")
  # the grid needs at least two seeds to produce an ARI at all
  checkmate::qassert(no_seeds, if (grid_search) "I1[2,)" else "I1")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  if (!embd_to_use %in% bixverse::get_available_embeddings(object)) {
    stop(sprintf("Embedding '%s' was not found.", embd_to_use))
  }

  embd <- bixverse::get_embedding(x = object, embd_name = embd_to_use)

  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take]
  }

  cells_to_use <- bixverse::get_cells_to_keep(object)

  if (grid_search) {
    res <- rs_fast_cluster_grid_gpu(
      embd = embd,
      resolutions = resolutions,
      n_centroids = n_centroids,
      fc_params = fc_params,
      snn = snn,
      return_kmeans = return_kmeans,
      no_seeds = no_seeds,
      seed = seed,
      verbose = parse_verbosity(.verbose)
    )
    memberships <- res$membership$memberships
    stats <- data.table::as.data.table(res$membership$stats)
    data.table::set(stats, j = "resolution", value = resolutions)
    data.table::setcolorder(stats, "resolution")
  } else {
    res <- rs_fast_cluster_gpu(
      embd = embd,
      resolutions = resolutions,
      n_centroids = n_centroids,
      fc_params = fc_params,
      snn = snn,
      return_kmeans = return_kmeans,
      seed = seed,
      verbose = parse_verbosity(.verbose)
    )
    memberships <- res$membership
    stats <- NULL
  }

  # cell_idx is 1-indexed ORIGINAL positions; matches obs_table$cell_idx
  names(memberships) <- paste0("res_", resolutions)
  membership_dt <- data.table::as.data.table(
    c(list(cell_idx = cells_to_use + 1L), memberships)
  )

  structure(
    list(
      memberships = membership_dt,
      stats = stats,
      k_means_cluster = res$k_means_cluster,
      centroids = res$centroids,
      resolutions = resolutions
    ),
    cell_indices = cells_to_use,
    class = "SingleCellFastClusters"
  )
}

# ------------------------------------------------------------------------------
# GPU-accelerated Scrublet:
# - Wraps `rs_sc_scrublet_gpu`.
# - Returns the same `ScrubletRes` S3 object as the CPU
#   `bixverse::scrublet_sc()`, so its print, plot, get_data and
#   call_doublets_manual methods work unchanged.
# - Full parity with the CPU method, `group_by` included, by reusing bixverse's
#   grouping internals rather than keeping a second copy of the cell-count
#   thresholds and the result reordering.
# - Only `SingleCells` gets a method. `bixverse::scrublet_sc()` has no
#   `SingleCellsSubset` method either, so adding one would be a superset.
# ------------------------------------------------------------------------------

# scrublet (gpu) ---------------------------------------------------------------

#' Doublet detection with Scrublet on the GPU
#'
#' @description GPU counterpart of [bixverse::scrublet_sc()]. Three stages run
#' on the WGPU backend: the randomised sparse SVD of the observed cells, the
#' projection of the simulated doublets into that PC space, and the kNN over
#' the combined embedding. HVG selection, doublet simulation, the kNN
#' classifier and the Otsu threshold stay on the CPU, so the speedup tracks how
#' much of the run the SVD and the kNN own. That share grows with cell count:
#' the combined embedding is `(1 + sim_doublet_ratio) * n_cells` rows tall and
#' an exhaustive kNN over it is quadratic.
#'
#' @details Scores do not match the CPU bit for bit. The SVD is randomised on
#' both sides but draws a different sketch, and the GPU indices break neighbour
#' ties differently. Expect a correlation around 0.99 rather than equality, and
#' a handful of borderline calls to flip because Otsu's threshold is a step
#' function of the histogram bins.
#'
#' @param object `SingleCells` class from `bixverse`.
#' @param scrublet_params List. Output of [params_scrublet_gpu()].
#' @param seed Integer. Random seed.
#' @param streaming Optional boolean. Shall the counts be streamed during HVG
#' selection. If `NULL`, resolved from the cell count.
#' @param cells_to_use Optional character vector. Names of the cells to run on.
#' The returned object covers exactly these cells.
#' @param group_by Optional string. Column in the obs table to run the method
#' per level of, typically a sample identifier.
#' @param return_combined_pca Boolean. Shall the combined PCA of observed cells
#' and simulated doublets be returned.
#' @param return_pairs Boolean. Shall the parent indices of the simulated
#' doublets be returned.
#' @param .verbose Boolean or integer. Controls verbosity and returns run
#' times. `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` ->
#' detailed verbosity.
#'
#' @returns A `ScrubletRes` S3 object, identical in shape to the CPU one, with
#' the following items:
#' \itemize{
#'   \item predicted_doublets - Boolean vector indicating which observed cells
#'   were predicted as doublets (TRUE = doublet, FALSE = singlet).
#'   \item doublet_scores_obs - Numerical vector with the likelihood of being
#'   a doublet for the observed cells.
#'   \item doublet_scores_sim - Numerical vector with the likelihood of being
#'   a doublet for the simulated cells.
#'   \item doublet_errors_obs - Numerical vector with the standard errors of
#'   the scores for the observed cells.
#'   \item z_scores - Z-scores for the observed cells. Represents:
#'   `score - threshold / error`.
#'   \item threshold - Used threshold.
#'   \item detected_doublet_rate - Fraction of cells that are called as
#'   doublet.
#'   \item detectable_doublet_fraction - Fraction of simulated doublets with
#'   scores above the threshold.
#'   \item overall_doublet_rate - Estimated overall doublet rate.
#'   \item pca - Optional PCA embeddings across the original cells and
#'   simulated doublets.
#'   \item pair_1 - Optional index of the parent cell 1 of the simulated
#'   doublets.
#'   \item pair_2 - Optional index of the parent cell 2 of the simulated
#'   doublets.
#' }
#' The 0-indexed cell indices are attached as the `cell_indices` attribute.
#' Grouped runs additionally carry `grouped` and `group_by_col` attributes and
#' a `cell_groups` element.
#'
#' @export
#'
#' @references Wolock, et al., Cell Syst, 2020
scrublet_gpu_sc <- S7::new_generic(
  name = "scrublet_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    scrublet_params = params_scrublet_gpu(),
    seed = 42L,
    streaming = NULL,
    cells_to_use = NULL,
    group_by = NULL,
    return_combined_pca = FALSE,
    return_pairs = FALSE,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method scrublet_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(scrublet_gpu_sc, SingleCells) <- function(
  object,
  scrublet_params = params_scrublet_gpu(),
  seed = 42L,
  streaming = NULL,
  cells_to_use = NULL,
  group_by = NULL,
  return_combined_pca = FALSE,
  return_pairs = FALSE,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, bixverse::SingleCells))
  assertScrubletGpu(scrublet_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(streaming, c("B1", "0"))
  checkmate::qassert(cells_to_use, c("S+", "0"))
  checkmate::qassert(group_by, c("S1", "0"))
  checkmate::qassert(return_combined_pca, "B1")
  checkmate::qassert(return_pairs, "B1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  cells_to_use <- if (!is.null(cells_to_use)) {
    bixverse::get_cell_indices(
      object,
      cell_ids = cells_to_use,
      rust_index = TRUE
    )
  } else {
    bixverse::get_cells_to_keep(object)
  }

  if (is.null(group_by)) {
    return(.scrublet_gpu_run(
      object = object,
      cells_to_use = cells_to_use,
      scrublet_params = scrublet_params,
      seed = seed,
      streaming = streaming,
      return_combined_pca = return_combined_pca,
      return_pairs = return_pairs,
      .verbose = .verbose
    ))
  }

  # the grouping machinery is bixverse's. Reimplementing it here would mean two
  # copies of the group size thresholds and of the reordering in
  # `.concat_scrublet`, kept in step by hand.
  bixverse:::.assert_group_by(object, group_by)
  groups <- bixverse:::.split_cells_by_group(object, group_by, cells_to_use)
  bixverse:::.validate_group_sizes(groups)

  group_results <- bixverse:::.run_per_group(
    groups = groups,
    per_group_fn = function(cells, name, inner_v) {
      .scrublet_gpu_run(
        object = object,
        cells_to_use = cells,
        scrublet_params = scrublet_params,
        seed = seed,
        streaming = streaming,
        return_combined_pca = return_combined_pca,
        return_pairs = return_pairs,
        .verbose = inner_v
      )
    },
    .verbose = .verbose,
    label = "Running Scrublet (GPU) per group"
  )

  bixverse:::.concat_scrublet(
    group_results,
    group_by,
    return_combined_pca,
    return_pairs
  )
}

## implementation --------------------------------------------------------------

#' Run GPU Scrublet on a set of cells
#'
#' @description GPU sibling of `bixverse:::.scrublet_run()`. Resolves
#' streaming, calls [rs_sc_scrublet_gpu()] and stamps the result with the
#' `ScrubletRes` class plus the `cell_indices` attribute that every downstream
#' method reads.
#'
#' @inheritParams scrublet_gpu_sc
#'
#' @param cells_to_use Integer vector of 0-indexed cell indices.
#'
#' @returns A `ScrubletRes` S3 object.
#'
#' @keywords internal
.scrublet_gpu_run <- function(
  object,
  cells_to_use,
  scrublet_params,
  seed,
  streaming,
  return_combined_pca,
  return_pairs,
  .verbose
) {
  streaming <- bixverse:::auto_streaming(
    n_cells = length(cells_to_use),
    streaming = streaming,
    .verbose = .verbose
  )

  scrublet_res <- rs_sc_scrublet_gpu(
    f_path_gene = bixverse:::get_rust_count_gene_f_path(object),
    f_path_cell = bixverse:::get_rust_count_cell_f_path(object),
    cells_to_keep = cells_to_use,
    scrublet_params = scrublet_params,
    seed = seed,
    verbose = parse_verbosity(.verbose),
    streaming = streaming,
    return_combined_pca = return_combined_pca,
    return_pairs = return_pairs
  )

  attr(scrublet_res, "cell_indices") <- cells_to_use
  class(scrublet_res) <- "ScrubletRes"
  scrublet_res
}
