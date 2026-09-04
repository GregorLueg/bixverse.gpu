# deprecated calls -------------------------------------------------------------

## param wrappers --------------------------------------------------------------

#' @title Default parameters for CAGRA-style kNN search (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' The CAGRA, IVF and exhaustive GPU searches share one parameter wrapper now,
#' see [bixverse.gpu::params_nn_gpu()].
#'
#' @param k Integer. Number of neighbours. Carried on the returned list so the
#' deprecated generics can still read it.
#' @param ann_dist Character. One of `"euclidean"` or `"cosine"`.
#' @param node_degree_final Optional integer. Final node degree of the CAGRA
#' navigational graph.
#' @param k_build Optional integer. Node degree during the NNDescent build
#' phase before CAGRA pruning.
#' @param refine_sweeps Integer. Ignored, the knob is gone.
#' @param max_iters Optional integer. Ignored, the knob is gone.
#' @param n_trees Optional integer. Number of trees in the initial forest.
#' @param delta Numeric. Termination criterion for the NNDescent iterations.
#' @param rho Optional numeric. Sampling rate during NNDescent iterations.
#' @param beam_width Optional integer. Beam width during querying.
#' @param max_beam_iters Optional integer. Maximum beam iterations.
#' @param n_entry_points Optional integer. Number of entry points.
#'
#' @return A list with the parameters, as returned by [params_nn_gpu()].
#'
#' @keywords internal
#' @importFrom lifecycle deprecate_warn
#' @export
params_sc_cagra <- function(
  k = 15L,
  ann_dist = "euclidean",
  node_degree_final = NULL,
  k_build = NULL,
  refine_sweeps = 0L,
  max_iters = NULL,
  n_trees = NULL,
  delta = 0.001,
  rho = NULL,
  beam_width = NULL,
  max_beam_iters = NULL,
  n_entry_points = NULL
) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "params_sc_cagra()",
    with = "params_nn_gpu()"
  )
  c(
    params_nn_gpu(
      dist_metric = ann_dist,
      node_degree_final = node_degree_final,
      k_build = k_build,
      n_tree = n_trees,
      delta = delta,
      rho = rho,
      beam_width = beam_width,
      max_beam_iters = max_beam_iters,
      n_entry_points = n_entry_points
    ),
    list(k = k)
  )
}

#' @title Default parameters for IVF-GPU kNN search (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' The CAGRA, IVF and exhaustive GPU searches share one parameter wrapper now,
#' see [bixverse.gpu::params_nn_gpu()].
#'
#' @param k Integer. Number of neighbours. Carried on the returned list so the
#' deprecated generics can still read it.
#' @param ann_dist Character. One of `"euclidean"` or `"cosine"`.
#' @param nlist Optional integer. Number of clusters to partition the index
#' into.
#' @param nprobe Optional integer. Number of clusters to probe at query time.
#' @param nquery Optional integer. Ignored, the knob is gone.
#' @param max_iters Optional integer. Ignored, the knob is gone.
#' @param seed Integer. Ignored. `seed` is an argument of the calling function.
#'
#' @return A list with the parameters, as returned by [params_nn_gpu()].
#'
#' @keywords internal
#' @importFrom lifecycle deprecate_warn
#' @export
params_sc_ivf <- function(
  k = 15L,
  ann_dist = "euclidean",
  nlist = NULL,
  nprobe = NULL,
  nquery = NULL,
  max_iters = NULL,
  seed = 42L
) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "params_sc_ivf()",
    with = "params_nn_gpu()"
  )
  c(
    params_nn_gpu(
      dist_metric = ann_dist,
      n_list = nlist,
      n_probes = nprobe
    ),
    list(k = k)
  )
}

## knn generics ----------------------------------------------------------------

#' @title Generate CAGRA GPU kNN data for single cells (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' CAGRA now sits behind [bixverse.gpu::generate_gpu_knn_sc()] as
#' `knn_method = "nndescent"`.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use.
#' @param cells_to_use Optional string vector. Cell names to include.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions.
#' @param modality String. One of `c("rna", "adt")`.
#' @param cagra_params List. Output of the deprecated [params_sc_cagra()], or
#' [params_nn_gpu()].
#' @param extract_knn Logical. Skip the beam search.
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity.
#'
#' @return Initialised `sc_knn` with the kNN data.
#'
#' @keywords internal
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
    cagra_params = params_nn_gpu(),
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
#' @importFrom lifecycle deprecate_warn
#'
#' @export
S7::method(generate_cagra_knn_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  cagra_params = params_nn_gpu(),
  extract_knn = TRUE,
  seed = 42L,
  .verbose = TRUE
) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "generate_cagra_knn_sc()",
    with = "generate_gpu_knn_sc()"
  )
  checkmate::qassert(extract_knn, "B1")
  k <- if (is.null(cagra_params$k)) 15L else as.integer(cagra_params$k)
  cagra_params$k <- NULL
  cagra_params$extract_knn <- extract_knn

  generate_gpu_knn_sc(
    object = object,
    embd_to_use = embd_to_use,
    cells_to_use = cells_to_use,
    no_embd_to_use = no_embd_to_use,
    modality = match.arg(modality),
    knn_method = "nndescent",
    nn_params = cagra_params,
    k = k,
    seed = seed,
    .verbose = .verbose
  )
}

#' @title Find neighbours via CAGRA GPU-acceleration for single cells
#' (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' CAGRA now sits behind [bixverse.gpu::find_neighbours_gpu_sc()] as
#' `knn_method = "nndescent"`.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param embd_to_use String. The embedding to use.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions.
#' @param modality String. One of `c("rna", "adt")`.
#' @param cagra_params List. Output of the deprecated [params_sc_cagra()], or
#' [params_nn_gpu()].
#' @param extract_knn Logical. Skip the beam search.
#' @param snn_params List. Output of [bixverse::params_sc_neighbours()].
#' @param seed Integer. For reproducibility.
#' @param .verbose Boolean. Controls verbosity.
#'
#' @return The object with added kNN matrix and sNN graph.
#'
#' @keywords internal
#' @export
find_neighbours_cagra_sc <- S7::new_generic(
  name = "find_neighbours_cagra_sc",
  dispatch_args = "object",
  fun = function(
    object,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    modality = c("rna", "adt"),
    cagra_params = params_nn_gpu(),
    extract_knn = FALSE,
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
#' @importFrom lifecycle deprecate_warn
#'
#' @export
S7::method(find_neighbours_cagra_sc, SingleCells) <- function(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  cagra_params = params_nn_gpu(),
  extract_knn = FALSE,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "find_neighbours_cagra_sc()",
    with = "find_neighbours_gpu_sc()"
  )
  checkmate::qassert(extract_knn, "B1")
  k <- if (is.null(cagra_params$k)) 15L else as.integer(cagra_params$k)
  cagra_params$k <- NULL
  cagra_params$extract_knn <- extract_knn

  find_neighbours_gpu_sc(
    object = object,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    modality = match.arg(modality),
    knn_method = "nndescent",
    nn_params = cagra_params,
    k = k,
    snn_params = snn_params,
    seed = seed,
    .verbose = .verbose
  )
}

## rust wrappers ---------------------------------------------------------------

#' @title CAGRA-style GPU-accelerated kNN graph (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' The three GPU kNN searches went behind one wrapper, see
#' [bixverse.gpu::rs_gpu_knn()]. Note that Euclidean distances now come back
#' as true L2 rather than squared.
#'
#' @param embd Numeric matrix of embeddings, cells x features.
#' @param cagra_params Named list, see [params_nn_gpu()].
#' @param extract_knn Logical. Skip the beam search.
#' @param seed Integer. Random seed for reproducibility.
#' @param verbose Integer. `0L` quiet, `1L` normal, `2L` detailed.
#'
#' @return A named list with `indices`, `dist` and `dist_metric`.
#'
#' @keywords internal
#' @importFrom lifecycle deprecate_warn
#' @export
rs_cagra_gpu_knn <- function(
  embd,
  cagra_params,
  extract_knn,
  seed,
  verbose
) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "rs_cagra_gpu_knn()",
    with = "rs_gpu_knn()"
  )
  k <- if (is.null(cagra_params$k)) 15L else as.integer(cagra_params$k)
  cagra_params$k <- NULL
  cagra_params$extract_knn <- extract_knn
  rs_gpu_knn(
    embd = embd,
    k = k,
    knn_method = "nndescent",
    nn_params = cagra_params,
    seed = seed,
    verbose = verbose
  )
}

#' @title IVF-GPU-accelerated kNN graph (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' The three GPU kNN searches went behind one wrapper, see
#' [bixverse.gpu::rs_gpu_knn()]. Note that Euclidean distances now come back
#' as true L2 rather than squared.
#'
#' @param embd Numeric matrix of embeddings, cells x features.
#' @param ivf_params Named list, see [params_nn_gpu()].
#' @param seed Integer. Random seed for reproducibility.
#' @param verbose Integer. `0L` quiet, `1L` normal, `2L` detailed.
#'
#' @return A named list with `indices`, `dist` and `dist_metric`.
#'
#' @keywords internal
#' @importFrom lifecycle deprecate_warn
#' @export
rs_ivf_gpu_knn <- function(embd, ivf_params, seed, verbose) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "rs_ivf_gpu_knn()",
    with = "rs_gpu_knn()"
  )
  k <- if (is.null(ivf_params$k)) 15L else as.integer(ivf_params$k)
  ivf_params$k <- NULL
  rs_gpu_knn(
    embd = embd,
    k = k,
    knn_method = "ivf",
    nn_params = ivf_params,
    seed = seed,
    verbose = verbose
  )
}

#' @title Exhaustive GPU-accelerated kNN graph (deprecated)
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' The three GPU kNN searches went behind one wrapper, see
#' [bixverse.gpu::rs_gpu_knn()]. Note that Euclidean distances now come back
#' as true L2 rather than squared.
#'
#' @param embd Numeric matrix of embeddings, cells x features.
#' @param k Integer. Number of neighbours to return.
#' @param dist_metric String. One of `c("euclidean", "cosine")`.
#' @param verbose Integer. `0L` quiet, `1L` normal, `2L` detailed.
#'
#' @return A named list with `indices`, `dist` and `dist_metric`.
#'
#' @keywords internal
#' @importFrom lifecycle deprecate_warn
#' @export
rs_exhaustive_gpu_knn <- function(embd, k, dist_metric, verbose) {
  lifecycle::deprecate_warn(
    when = "0.4.0",
    what = "rs_exhaustive_gpu_knn()",
    with = "rs_gpu_knn()"
  )
  rs_gpu_knn(
    embd = embd,
    k = k,
    knn_method = "exhaustive",
    nn_params = params_nn_gpu(dist_metric = dist_metric),
    seed = 42L,
    verbose = verbose
  )
}
