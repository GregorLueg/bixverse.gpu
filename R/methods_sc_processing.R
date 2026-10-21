# ------------------------------------------------------------------------------
# GPU-accelerated single cell processing: kNN searches, sparse randomised PCA,
# fast clustering and Scrublet doublet detection.
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
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the
#' search on Apple Silicon via MLX and needs a build with the MLX backend,
#' see [mlx_available()].
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
    backend = c("wgpu", "mlx"),
    .verbose = TRUE
  ) {
    assert_backend(match.arg(backend))

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
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  backend <- match.arg(backend)

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
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
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
    backend = backend,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  new_sc_knn(knn_data = knn_raw, used_cells = row.names(embd))
}

## find neighbours (GPU) -------------------------------------------------------

#' Find GPU-accelerated neighbours for single cells
#'
#' @description
#' This function generates kNN data using GPU-accelerated algorithms via the
#' `bixverse.gpu` package, then turns it into a connectivities igraph for
#' downstream clustering. See [generate_gpu_knn_sc()] for the three searches on
#' offer. This function lives in a separate package from the CPU-based
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
#' @param graph_params List. Output of
#' [bixverse::params_knn_graph_defaults()]. Controls how the kNN is turned into
#' the connectivities graph.
#' @param seed Integer. For reproducibility.
#' @param gpu_method `r lifecycle::badge("deprecated")` Use `knn_method`.
#' @param ivf_params `r lifecycle::badge("deprecated")` Use `nn_params`.
#' @param dist_metric `r lifecycle::badge("deprecated")` Use
#' `params_nn_gpu(dist_metric = )`.
#' @param snn_params `r lifecycle::badge("deprecated")` Use `graph_params`.
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the
#' search on Apple Silicon via MLX and needs a build with the MLX backend,
#' see [mlx_available()].
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return The object with added kNN matrix and connectivities graph in the
#' selected modality slot.
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
    graph_params = bixverse::params_knn_graph_defaults(),
    seed = 42L,
    gpu_method = lifecycle::deprecated(),
    ivf_params = lifecycle::deprecated(),
    dist_metric = lifecycle::deprecated(),
    snn_params = lifecycle::deprecated(),
    backend = c("wgpu", "mlx"),
    .verbose = TRUE
  ) {
    assert_backend(match.arg(backend))

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
  graph_params = bixverse::params_knn_graph_defaults(),
  seed = 42L,
  gpu_method = lifecycle::deprecated(),
  ivf_params = lifecycle::deprecated(),
  dist_metric = lifecycle::deprecated(),
  snn_params = lifecycle::deprecated(),
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  backend <- match.arg(backend)

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

  if (lifecycle::is_present(snn_params)) {
    lifecycle::deprecate_warn(
      when = "0.6.0",
      what = "find_neighbours_gpu_sc(snn_params)",
      with = "find_neighbours_gpu_sc(graph_params)"
    )
    # params_sc_neighbours() carries the graph keys next to the kNN ones
    graph_params <- snn_params
  }

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::assertChoice(modality, c("rna", "adt"))
  checkmate::assertChoice(knn_method, c("nndescent", "exhaustive", "ivf"))
  assertNnGpuParams(nn_params)
  checkmate::qassert(k, "I1[1,)")
  bixverse:::assertKnnGraphParams(graph_params)
  checkmate::qassert(seed, "I1")
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
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
    backend = backend,
    .verbose = .verbose
  )
  object <- set_knn(
    object,
    knn_data,
    modality = modality,
    from = embd_to_use
  )

  set_connectivities(
    object,
    connectivities = bixverse:::.knn_to_graph(
      knn = knn_data,
      graph_params = graph_params,
      .verbose = .verbose
    ),
    modality = modality,
    from = "knn"
  )
}

# pca --------------------------------------------------------------------------

## gpu-accelerated sparse svd -------------------------------------------------

#' GPU-accelerated PCA for single cell
#'
#' @description
#' This function will run a sparse SVD while running the large sparse matrix
#' multiplications on GPU for improved speed. This also means you will have to
#' provide the necessary VRAM for your data set. This version only works on the
#' `"rna"` modality.
#'
#' @details `svd_solver = "randomised"` runs the randomised SVD: fast, but
#' approximate in the trailing PCs. Every other solver (`"auto"`, `"exact"`,
#' `"covariance"`) runs an exact block Lanczos, as there is no GPU covariance
#' path. The default from [bixverse::params_sc_pca()] is `"auto"`, so the
#' default is exact and slower than the randomised solver.
#'
#' @param object `SingleCells` class
#' @param no_pcs Integer. Number of PCs to calculate.
#' @param pca_params Named list. Controls the parameters to be used for the
#' PCA calculation which is single cell-specific, see [params_sc_pca()].
#' `svd_solver = "randomised"` picks the randomised SVD, anything else the
#' exact block Lanczos.
#' @param hvg Optional integer. If you want to provide your own HVG genes.
#' Otherwise, the function will default to what is found in
#' [bixverse::get_hvg()]. Please provide 1-indexed genes here! If you provide
#' these, the internal HVG will be overwritten.
#' @param seed Integer. Seed for the sketch or the Lanczos starting block.
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the SVD on
#' Apple Silicon via MLX and needs a build with the MLX backend, see
#' [mlx_available()].
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
    backend = c("wgpu", "mlx"),
    .verbose = TRUE
  ) {
    assert_backend(match.arg(backend))

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
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
) {
  backend <- match.arg(backend)

  checkmate::assertClass(object, "bixverse::SingleCells")
  checkmate::qassert(no_pcs, "I1")
  bixverse:::assertScPcaParams(pca_params)
  checkmate::qassert(hvg, c("I+", "0"))
  checkmate::qassert(seed, "I1")
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
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
        "Using GPU-accelerated sparse SVD (%s) with %i HVG.",
        if (identical(pca_params$svd_solver, "randomised")) {
          "randomised"
        } else {
          "block Lanczos"
        },
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
      backend = backend,
      verbose = bixverse:::parse_verbosity(.verbose)
    )
  )

  object <- set_pca_factors(object, pca_factors)
  object <- set_pca_loadings(object, pca_loadings)
  object <- set_pca_singular_vals(object, singular_values[1:no_pcs])

  return(object)
}


# ------------------------------------------------------------------------------
# GPU-accelerated graph clustering:
# - Wraps `leiden_clustering_gpu()` on the cached connectivities, mirroring
#   `bixverse::find_clusters_sc()`.
# ------------------------------------------------------------------------------

# graph clustering (gpu) -------------------------------------------------------

#' Graph-based clustering of cells on the neighbour graph (GPU)
#'
#' @description
#' GPU counterpart of [bixverse::find_clusters_sc()]. Runs Leiden or Louvain
#' via [leiden_clustering_gpu()] on the neighbour graph and adds a column to
#' the obs table. Only the level-0 local moving runs on the GPU, and only on
#' graphs of at least 50,000 cells; smaller graphs run on the CPU. `0` is the
#' largest cluster.
#'
#' @param object `SingleCells` or `SingleCellsSubset` class from `bixverse`.
#' @param cluster_algorithm String. One of `c("leiden", "louvain")`.
#' @param res Numeric. The modularity resolution parameter, see
#' [leiden_clustering_gpu()].
#' @param name String. The name to add to the obs table.
#' @param modality String. Which modality's neighbour graph to use. One of
#' `c("rna", "adt", "wnn")`. The two latter options are only available for
#' multi-modal objects.
#' @param seed Integer. For reproducibility.
#' @param ... Further parameters for [leiden_clustering_gpu()]: `n_iterations`,
#' `beta`, `min_gain` and `local_moving`.
#'
#' @returns The object with the added clustering in the obs table.
#'
#' @export
find_clusters_gpu_sc <- S7::new_generic(
  name = "find_clusters_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    cluster_algorithm = c("leiden", "louvain"),
    res = 1.0,
    name = "leiden_clustering",
    modality = c("rna", "adt", "wnn"),
    seed = 42L,
    ...
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method find_clusters_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(find_clusters_gpu_sc, SingleCells) <- function(
  object,
  cluster_algorithm = c("leiden", "louvain"),
  res = 1.0,
  name = "leiden_clustering",
  modality = c("rna", "adt", "wnn"),
  seed = 42L,
  ...
) {
  .find_clusters_gpu(
    object = object,
    cluster_algorithm = match.arg(cluster_algorithm),
    res = res,
    name = name,
    modality = match.arg(modality),
    seed = seed,
    ...
  )
}

## SingleCellsSubset -----------------------------------------------------------

#' @method find_clusters_gpu_sc SingleCellsSubset
#'
#' @export
#'
#' @import bixverse
S7::method(find_clusters_gpu_sc, SingleCellsSubset) <- function(
  object,
  cluster_algorithm = c("leiden", "louvain"),
  res = 1.0,
  name = "leiden_clustering",
  modality = c("rna", "adt", "wnn"),
  seed = 42L,
  ...
) {
  .find_clusters_gpu(
    object = object,
    cluster_algorithm = match.arg(cluster_algorithm),
    res = res,
    name = name,
    modality = match.arg(modality),
    seed = seed,
    ...
  )
}

## implementation --------------------------------------------------------------

#' Run the GPU graph clustering and write it to the obs table
#'
#' @param object `SingleCells` or `SingleCellsSubset` class.
#' @param cluster_algorithm String. One of `c("leiden", "louvain")`.
#' @param res Numeric. The modularity resolution parameter.
#' @param name String. The name of the obs column.
#' @param modality String. One of `c("rna", "adt", "wnn")`.
#' @param seed Integer. For reproducibility.
#' @param ... Passed to [leiden_clustering_gpu()].
#'
#' @return The object with the added clustering in the obs table.
#'
#' @keywords internal
.find_clusters_gpu <- function(
  object,
  cluster_algorithm,
  res,
  name,
  modality,
  seed,
  ...
) {
  # checks
  checkmate::assertChoice(cluster_algorithm, c("leiden", "louvain"))
  checkmate::qassert(res, "N1")
  checkmate::qassert(name, "S1")
  checkmate::assertChoice(modality, c("rna", "adt", "wnn"))
  checkmate::qassert(seed, "I1")

  if (
    modality != "rna" &&
      !S7::S7_inherits(object, bixverse::SingleCellsMultiModal)
  ) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  # the labels are written back aligned to the kept cells, so a graph over a
  # different cell set silently mislabels
  assert_sc_state(object, artefacts = "connectivities", modality = modality)

  connectivities <- get_connectivities(object, modality = modality)
  if (is.null(connectivities)) {
    warning(
      paste(
        "No neighbour graph found. Did you run find_neighbours_gpu_sc().",
        "Returning class as is."
      )
    )
    return(object)
  }

  clusters <- leiden_clustering_gpu(
    graph = connectivities,
    resolution = res,
    algorithm = cluster_algorithm,
    seed = seed,
    ...
  )

  # Rust returns 1-based labels ordered by size; the column is 0-based
  object[[name]] <- clusters - 1L

  object
}

# ------------------------------------------------------------------------------
# GPU-accelerated fast clustering:
# - Wraps `rs_fast_cluster_gpu`.
# - Returns the same `SingleCellFastClusters` S3 object as the CPU
#   `bixverse::fast_cluster_sc()`, so all its getters and `add_sc_new_obs()`
#   work unchanged.
# - bixverse dispatches on its internal `ScOrScSubset` union. That union is not
#   exported, so the two exported classes get a method each and both delegate to
#   `.fast_cluster_gpu()`.
# ------------------------------------------------------------------------------

# fast clustering (gpu) --------------------------------------------------------

#' Run fast clustering on a SingleCells object (GPU)
#'
#' @description
#' GPU counterpart of [bixverse::fast_cluster_sc()]. Coarsens the cells with
#' k-means on the chosen embedding, builds a graph over the centroids (a kNN
#' graph or the cell-mediated graph), runs Louvain or Leiden on it and
#' propagates the communities back to the cells. With `n_seeds > 1` every
#' resolution is run once per seed and stability statistics are returned.
#'
#' Only the k-means coarsening (and the cell-mediated top-m search) run on the
#' GPU. The centroid graph and the community detection stay on the CPU, so the
#' speedup tracks how much of the run k-means owns. That share grows with the
#' cell count and with `n_centroids`. There is no `km_type` argument: the GPU
#' k-means is full-batch Lloyd's and has no mini-batch path.
#'
#' @param object `SingleCells` or `SingleCellsSubset` class from `bixverse`.
#' @param embd_to_use String. Embedding name. Defaults to `"pca"`.
#' @param no_embd_to_use Optional integer. Number of dimensions to keep.
#' @param resolutions Numeric vector. Resolutions of the community detection.
#' @param fc_params List. Output of [params_sc_fast_cluster_gpu()].
#' @param n_seeds Integer. Number of seeds per resolution. `1L` runs a single
#' pass without stability statistics.
#' @param return_kmeans Boolean. Return the k-means assignments and centroids.
#' @param seed Integer. Seed for reproducibility.
#' @param backend String. One of `c("wgpu", "mlx")`. `"mlx"` runs the k-means
#' on Apple Silicon via MLX and needs a build with the MLX backend, see
#' [mlx_available()].
#' @param n_centroids `r lifecycle::badge("deprecated")` Use
#' `params_sc_fast_cluster_gpu(n_centroids = )`.
#' @param snn `r lifecycle::badge("deprecated")` Use
#' `params_sc_fast_cluster_gpu(centroid_graph = list(knn_sym = ))`.
#' @param grid_search `r lifecycle::badge("deprecated")` Use `n_seeds`.
#' @param no_seeds `r lifecycle::badge("deprecated")` Use `n_seeds`.
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @returns `SingleCellFastClusters` S3 object with:
#' \describe{
#'   \item{memberships}{data.table with `cell_idx` and one column per
#'   resolution (`res_<value>`).}
#'   \item{stats}{data.table of the stability statistics, or `NULL` for
#'   `n_seeds = 1L`.}
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
    fc_params = params_sc_fast_cluster_gpu(),
    n_seeds = 1L,
    return_kmeans = FALSE,
    seed = 42L,
    backend = c("wgpu", "mlx"),
    n_centroids = lifecycle::deprecated(),
    snn = lifecycle::deprecated(),
    grid_search = lifecycle::deprecated(),
    no_seeds = lifecycle::deprecated(),
    .verbose = TRUE
  ) {
    assert_backend(match.arg(backend))

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
  fc_params = params_sc_fast_cluster_gpu(),
  n_seeds = 1L,
  return_kmeans = FALSE,
  seed = 42L,
  backend = c("wgpu", "mlx"),
  n_centroids = lifecycle::deprecated(),
  snn = lifecycle::deprecated(),
  grid_search = lifecycle::deprecated(),
  no_seeds = lifecycle::deprecated(),
  .verbose = TRUE
) {
  .fast_cluster_gpu(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    resolutions = resolutions,
    fc_params = fc_params,
    n_seeds = n_seeds,
    return_kmeans = return_kmeans,
    seed = seed,
    backend = match.arg(backend),
    n_centroids = n_centroids,
    snn = snn,
    grid_search = grid_search,
    no_seeds = no_seeds,
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
  fc_params = params_sc_fast_cluster_gpu(),
  n_seeds = 1L,
  return_kmeans = FALSE,
  seed = 42L,
  backend = c("wgpu", "mlx"),
  n_centroids = lifecycle::deprecated(),
  snn = lifecycle::deprecated(),
  grid_search = lifecycle::deprecated(),
  no_seeds = lifecycle::deprecated(),
  .verbose = TRUE
) {
  .fast_cluster_gpu(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    resolutions = resolutions,
    fc_params = fc_params,
    n_seeds = n_seeds,
    return_kmeans = return_kmeans,
    seed = seed,
    backend = match.arg(backend),
    n_centroids = n_centroids,
    snn = snn,
    grid_search = grid_search,
    no_seeds = no_seeds,
    .verbose = .verbose
  )
}

## implementation --------------------------------------------------------------

#' Fold the deprecated fast clustering arguments into the current ones
#'
#' @param fc_params List. Current argument.
#' @param n_seeds Integer. Current argument.
#' @param n_centroids Deprecated. Superseded by `fc_params$n_centroids`.
#' @param snn Deprecated. Superseded by `fc_params$knn_sym`.
#' @param grid_search Deprecated. Superseded by `n_seeds`.
#' @param no_seeds Deprecated. Superseded by `n_seeds`.
#'
#' @return A list with the resolved `fc_params` and `n_seeds`.
#'
#' @keywords internal
.resolve_deprecated_fc_args <- function(
  fc_params,
  n_seeds,
  n_centroids,
  snn,
  grid_search,
  no_seeds
) {
  fn <- "fast_cluster_gpu_sc"

  if (lifecycle::is_present(n_centroids)) {
    lifecycle::deprecate_warn(
      when = "0.6.0",
      what = sprintf("%s(n_centroids)", fn),
      with = "params_sc_fast_cluster_gpu(n_centroids = )"
    )
    fc_params$n_centroids <- n_centroids
  }

  if (lifecycle::is_present(snn)) {
    lifecycle::deprecate_warn(
      when = "0.6.0",
      what = sprintf("%s(snn)", fn),
      with = "params_sc_fast_cluster_gpu(centroid_graph = list(knn_sym = ))"
    )
    checkmate::qassert(snn, "B1")
    fc_params$knn_sym <- if (snn) "snn" else "knn"
  }

  if (lifecycle::is_present(no_seeds)) {
    lifecycle::deprecate_warn(
      when = "0.6.0",
      what = sprintf("%s(no_seeds)", fn),
      with = sprintf("%s(n_seeds)", fn)
    )
  }

  if (lifecycle::is_present(grid_search)) {
    lifecycle::deprecate_warn(
      when = "0.6.0",
      what = sprintf("%s(grid_search)", fn),
      with = sprintf("%s(n_seeds)", fn)
    )
    checkmate::qassert(grid_search, "B1")
    # the old grid defaulted to 10 seeds
    if (grid_search) {
      n_seeds <- if (lifecycle::is_present(no_seeds)) no_seeds else 10L
    }
  }

  list(fc_params = fc_params, n_seeds = n_seeds)
}

#' Shared implementation of the GPU fast clustering
#'
#' @description
#' Body behind both `fast_cluster_gpu_sc()` methods. Pulls the embedding off the
#' object, resolves the `NULL` defaults of [params_sc_fast_cluster_gpu()], hands
#' it to [rs_fast_cluster_gpu()] and wraps the result into a
#' `SingleCellFastClusters` S3 object.
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
  fc_params,
  n_seeds,
  return_kmeans,
  seed,
  backend,
  n_centroids,
  snn,
  grid_search,
  no_seeds,
  .verbose
) {
  resolved <- .resolve_deprecated_fc_args(
    fc_params = fc_params,
    n_seeds = n_seeds,
    n_centroids = n_centroids,
    snn = snn,
    grid_search = grid_search,
    no_seeds = no_seeds
  )
  fc_params <- resolved$fc_params
  n_seeds <- resolved$n_seeds

  # checks
  checkmate::assertTRUE(
    S7::S7_inherits(object, bixverse::SingleCells) ||
      S7::S7_inherits(object, bixverse::SingleCellsSubset)
  )
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(resolutions, "N+")
  assertScFastClusterGpuParams(fc_params)
  checkmate::qassert(n_seeds, "I1[1,)")
  checkmate::qassert(return_kmeans, "B1")
  checkmate::qassert(seed, "I1")
  checkmate::assertChoice(backend, c("wgpu", "mlx"))
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  if (!embd_to_use %in% bixverse::get_available_embeddings(object)) {
    stop(sprintf("Embedding '%s' was not found.", embd_to_use))
  }

  embd <- bixverse::get_embedding(x = object, embd_name = embd_to_use)
  if (!is.null(no_embd_to_use)) {
    embd <- embd[, 1:min(no_embd_to_use, ncol(embd))]
  }

  if (is.null(fc_params$n_centroids)) {
    fc_params$n_centroids <- as.integer(floor(4 * sqrt(nrow(embd))))
  }
  if (is.null(fc_params$n_iterations)) {
    fc_params$n_iterations <- bixverse:::.default_community_iters(
      fc_params$algorithm
    )
  }

  cells_to_use <- bixverse::get_cells_to_keep(object)

  res <- rs_fast_cluster_gpu(
    embd = embd,
    resolutions = resolutions,
    fc_params = fc_params,
    n_seeds = n_seeds,
    seed = seed,
    verbose = parse_verbosity(.verbose),
    backend = backend
  )

  # cell_idx is 1-indexed ORIGINAL positions; matches obs_table$cell_idx
  membership_dt <- data.table::data.table(cell_idx = cells_to_use + 1L)
  data.table::set(
    membership_dt,
    j = paste0("res_", resolutions),
    value = res$membership
  )

  stats <- NULL
  if (!is.null(res$stats)) {
    stats <- data.table::as.data.table(res$stats)
    data.table::set(stats, j = "resolution", value = resolutions)
    data.table::setcolorder(stats, "resolution")
  }

  structure(
    list(
      memberships = membership_dt,
      stats = stats,
      k_means_cluster = if (return_kmeans) res$k_means_cluster else NULL,
      centroids = if (return_kmeans) res$centroids else NULL,
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
