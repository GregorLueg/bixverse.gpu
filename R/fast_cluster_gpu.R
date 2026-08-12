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
  assertScFastClusterGpu(fc_params)
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
