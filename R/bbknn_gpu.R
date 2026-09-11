# ------------------------------------------------------------------------------
# GPU-accelerated BBKNN:
# - Wraps `rs_bbknn_gpu`.
# - Writes back the same artefacts as `bixverse::bbknn_sc()`: a kNN matrix
#   filtered down from the BBKNN distances, and an sNN graph built from the
#   BBKNN connectivities.
# - The graph does NOT go through `.set_snn_graph_gpu()`. BBKNN produces
#   connectivities directly, so there is no shared nearest neighbour step and
#   no `snn_params`. Weights are the connectivities themselves.
# - Reuses `bixverse::rs_bbknn_filtering()` rather than keeping a second copy
#   of the CSR-to-matrix filter. It is identical work on both paths.
# ------------------------------------------------------------------------------

# bbknn (gpu) ------------------------------------------------------------------

#' Run BBKNN on the GPU
#'
#' @description GPU counterpart of [bixverse::bbknn_sc()], implementing the
#' batch-balanced k-nearest neighbour algorithm from Polański, et al. One
#' nearest neighbour index is built per batch and queried by every cell, so
#' each cell gets `neighbours_within_batch` neighbours from every batch. The
#' UMAP connectivity calculations that reduce spurious connections then run on
#' the CPU, shared with the CPU implementation.
#'
#' @details Only the per-batch searches move to the device, and they are the
#' part that scales with batch count: BBKNN builds one index per batch and
#' queries each with all cells, so the work grows as `n_cells * n_batches`.
#' With a handful of batches on a small object the CPU is fine. The GPU starts
#' to matter once you have many samples.
#'
#' Results match the CPU path exactly with `knn_method = "exhaustive"`, since
#' both are exact and recompute distances against the same embedding. The
#' approximate backends break ties differently and will not.
#'
#' @param object `SingleCells` or `SingleCellsSubset` class from `bixverse`.
#' @param batch_column String. The column with the batch information in the
#' obs data of the class.
#' @param no_neighbours_to_keep Integer. Maximum number of neighbours to keep
#' from the BBKNN algorithm. Generating neighbours per batch can produce a lot
#' of them, so this keeps the top `no_neighbours_to_keep`. Defaults to `5L`.
#' @param embd_to_use String. The embedding to use. Atm, the only option is
#' `"pca"`.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL` all will be used.
#' @param bbknn_params List. Output of [params_sc_bbknn_gpu()].
#' @param seed Integer. Random seed.
#' @param .verbose Boolean or integer. Controls verbosity and returns run
#' times. `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` ->
#' detailed verbosity.
#'
#' @returns The object with the added kNN matrix based on BBKNN and the graph
#' based on the returned connectivities of the algorithm.
#'
#' @export
#'
#' @references Polański, et al., Bioinformatics, 2020
bbknn_gpu_sc <- S7::new_generic(
  name = "bbknn_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    batch_column,
    no_neighbours_to_keep = 5L,
    embd_to_use = "pca",
    no_embd_to_use = NULL,
    bbknn_params = params_sc_bbknn_gpu(),
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method bbknn_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(bbknn_gpu_sc, SingleCells) <- function(
  object,
  batch_column,
  no_neighbours_to_keep = 5L,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  bbknn_params = params_sc_bbknn_gpu(),
  seed = 42L,
  .verbose = TRUE
) {
  .bbknn_gpu(
    object = object,
    batch_column = batch_column,
    no_neighbours_to_keep = no_neighbours_to_keep,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    bbknn_params = bbknn_params,
    seed = seed,
    .verbose = .verbose
  )
}

## SingleCellsSubset -----------------------------------------------------------

#' @method bbknn_gpu_sc SingleCellsSubset
#'
#' @export
#'
#' @import bixverse
S7::method(bbknn_gpu_sc, SingleCellsSubset) <- function(
  object,
  batch_column,
  no_neighbours_to_keep = 5L,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  bbknn_params = params_sc_bbknn_gpu(),
  seed = 42L,
  .verbose = TRUE
) {
  .bbknn_gpu(
    object = object,
    batch_column = batch_column,
    no_neighbours_to_keep = no_neighbours_to_keep,
    embd_to_use = embd_to_use,
    no_embd_to_use = no_embd_to_use,
    bbknn_params = bbknn_params,
    seed = seed,
    .verbose = .verbose
  )
}

## implementation --------------------------------------------------------------

#' Shared implementation of the GPU BBKNN
#'
#' @description
#' Body behind both `bbknn_gpu_sc()` methods. Mirrors
#' [bixverse::bbknn_sc()] step for step, with [rs_bbknn_gpu()] in place of the
#' CPU search.
#'
#' @inheritParams bbknn_gpu_sc
#'
#' @returns The object with the kNN matrix and the connectivity graph set.
#'
#' @keywords internal
.bbknn_gpu <- function(
  object,
  batch_column,
  no_neighbours_to_keep,
  embd_to_use,
  no_embd_to_use,
  bbknn_params,
  seed,
  .verbose
) {
  # checks
  checkmate::assertTRUE(
    S7::S7_inherits(object, bixverse::SingleCells) ||
      S7::S7_inherits(object, bixverse::SingleCellsSubset)
  )
  checkmate::qassert(batch_column, "S1")
  checkmate::qassert(no_neighbours_to_keep, "I1[1,)")
  checkmate::assertChoice(embd_to_use, c("pca"))
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  assertScBbknnGpu(bbknn_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # presence probe, not a read: the kNN is about to be overwritten, so a stale
  # one here is not a problem worth signalling about
  if (bixverse:::.sc_has_artefact(object, "knn")) {
    warning("Prior kNN matrix found. Will be overwritten.")
  }

  # hard tier: the kNN built here feeds the graph and everything downstream
  assert_sc_state(object, artefacts = embd_to_use)

  embd <- switch(embd_to_use, pca = get_pca_factors(object))

  if (is.null(embd)) {
    warning(paste(
      "The desired embedding was not found. Please check the parameters.",
      "Returning NULL."
    ))
    return(NULL)
  }

  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take]
  }

  batch_index <- as.integer(factor(unlist(object[[batch_column]]))) - 1L

  # Rust errors on a single batch, the CPU method returns the object as is.
  # Catch it here so both paths behave the same.
  no_batches <- length(unique(batch_index))
  if (no_batches < 2L) {
    warning("The batch column only has one batch. Returning object as is.")
    return(object)
  }

  no_generated_neighbours <- no_batches *
    bbknn_params$neighbours_within_batch

  if (no_neighbours_to_keep > no_generated_neighbours) {
    warning(paste(
      "The number of desired neighbours cannot be generated with these BBKNN",
      "parameters (too few generated neighbours).",
      "Please adopt neighbours_within_batch accordingly.",
      "Returning all neighbours from BBKNN."
    ))
  }

  if (.verbose) {
    message("Running BBKNN algorithm on the GPU.")
  }

  bbknn_res <- rs_bbknn_gpu(
    embd = embd,
    batch_labels = batch_index,
    bbknn_params = bbknn_params,
    seed = seed,
    verbose = parse_verbosity(.verbose)
  )

  knn_data <- {
    no_k <- min(no_neighbours_to_keep, no_generated_neighbours)
    filtered <- rs_bbknn_filtering(
      indptr = bbknn_res$distances$indptr,
      indices = bbknn_res$distances$indices,
      data = bbknn_res$distances$data,
      no_neighbours_to_keep = no_k
    )
    list(
      indices = filtered$indices,
      dist = filtered$dist,
      dist_metric = bbknn_params[["ann_dist"]]
    )
  }

  storage.mode(knn_data$indices) <- "integer"

  used_cells <- get_cell_names(object, filtered = TRUE)
  sc_knn <- new_sc_knn(knn_data = knn_data, used_cells = used_cells)
  object <- set_knn(object, knn = sc_knn, from = embd_to_use)

  if (.verbose) {
    message(paste(
      "Generating graph based on BBKNN connectivities.",
      "Weights will be based on the connectivities",
      "and not shared nearest neighbour calculations."
    ))
  }

  sparse_mat <- Matrix::sparseMatrix(
    i = rep(
      seq_along(bbknn_res$connectivities$indptr[-1]),
      diff(bbknn_res$connectivities$indptr)
    ),
    j = bbknn_res$connectivities$indices + 1,
    x = bbknn_res$connectivities$data,
    dims = c(
      bbknn_res$connectivities$nrow,
      bbknn_res$connectivities$ncol
    ),
    index1 = TRUE
  )

  snn_graph <- igraph::graph_from_adjacency_matrix(
    sparse_mat,
    mode = "max",
    weighted = TRUE
  )

  set_snn_graph(object, snn_graph = snn_graph, from = "knn")
}
