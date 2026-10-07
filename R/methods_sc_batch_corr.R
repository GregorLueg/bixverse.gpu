# ------------------------------------------------------------------------------
# GPU-accelerated batch correction: Harmony v2, BBKNN and fastMNN.
# ------------------------------------------------------------------------------

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
    assert_gpu()

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
  assertScHarmonyV2GpuParams(harmony_params)
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

  batch_indices <- object[[batch_column]][[1]]
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
  assertScBbknnGpuParams(bbknn_params)
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

  batch_index <- as.integer(factor(object[[batch_column]][[1]])) - 1L

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

  # `"nndescent"` is the package-wide name, Rust only knows `"nndescent_gpu"`
  bbknn_params[["knn_method"]] <- .normalise_gpu_knn_method(
    bbknn_params[["knn_method"]]
  )

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

# ------------------------------------------------------------------------------
# GPU-accelerated fastMNN:
# - Wraps `rs_fast_mnn_gpu`.
# - Corrects the PCA already in the object. Unlike `bixverse::fast_mnn_sc()`
#   there is no PCA recomputation on batch-aware HVGs; run the PCA on those
#   first if that is what you want.
# - Writes back a `"mnn_gpu"` embedding, mirroring `"harmony_gpu"`.
# ------------------------------------------------------------------------------

# fast mnn (gpu) ---------------------------------------------------------------

#' Run fastMNN on the GPU
#'
#' @description GPU counterpart of [bixverse::fast_mnn_sc()], implementing the
#' fast mutual nearest neighbour correction from Haghverdi, et al. Batches are
#' merged one after the other: MNN pairs between the merged block and the next
#' batch give correction vectors, which are smoothed with a tricube kernel and
#' applied to every cell of that batch.
#'
#' @details Only the neighbour searches move to the device: both MNN
#' directions and the tricube search, three index builds per merge. Those
#' dominate the run time on the CPU path. Centring, MNN pairing and the
#' tricube correction are shared with the CPU implementation.
#'
#' The PCA stored in the object is corrected as is. If you want the PCA on
#' batch-aware HVGs, as [bixverse::fast_mnn_sc()] can recompute it, run
#' [bixverse::find_hvg_batch_aware_sc()] and the PCA first.
#'
#' Results match the CPU path with `knn_method = "exhaustive"` on both sides,
#' since both searches are exact. The approximate backends will not match.
#'
#' @param object `SingleCells` or `SingleCellsSubset` class from `bixverse`.
#' @param batch_column String. The column with the batch information in the
#' obs data of the class.
#' @param no_embd_to_use Optional integer. Number of PCs to use. If `NULL` all
#' will be used.
#' @param fastmnn_params List. Output of [params_sc_fastmnn_gpu()].
#' @param seed Integer. Random seed.
#' @param .verbose Boolean or integer. Controls verbosity and returns run
#' times. `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` ->
#' detailed verbosity.
#'
#' @returns The object with a `"mnn_gpu"` embedding added. If the batch column
#' only has one batch, the object is returned as is with a warning.
#'
#' @export
#'
#' @references Haghverdi, et al., Nat Biotechnol, 2018
fast_mnn_gpu_sc <- S7::new_generic(
  name = "fast_mnn_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    batch_column,
    no_embd_to_use = NULL,
    fastmnn_params = params_sc_fastmnn_gpu(),
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method fast_mnn_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(fast_mnn_gpu_sc, SingleCells) <- function(
  object,
  batch_column,
  no_embd_to_use = NULL,
  fastmnn_params = params_sc_fastmnn_gpu(),
  seed = 42L,
  .verbose = TRUE
) {
  .fast_mnn_gpu(
    object = object,
    batch_column = batch_column,
    no_embd_to_use = no_embd_to_use,
    fastmnn_params = fastmnn_params,
    seed = seed,
    .verbose = .verbose
  )
}

## SingleCellsSubset -----------------------------------------------------------

#' @method fast_mnn_gpu_sc SingleCellsSubset
#'
#' @export
#'
#' @import bixverse
S7::method(fast_mnn_gpu_sc, SingleCellsSubset) <- function(
  object,
  batch_column,
  no_embd_to_use = NULL,
  fastmnn_params = params_sc_fastmnn_gpu(),
  seed = 42L,
  .verbose = TRUE
) {
  .fast_mnn_gpu(
    object = object,
    batch_column = batch_column,
    no_embd_to_use = no_embd_to_use,
    fastmnn_params = fastmnn_params,
    seed = seed,
    .verbose = .verbose
  )
}

## implementation --------------------------------------------------------------

#' Shared implementation of the GPU fastMNN
#'
#' @description
#' Body behind both `fast_mnn_gpu_sc()` methods.
#'
#' @inheritParams fast_mnn_gpu_sc
#'
#' @returns The object with the `"mnn_gpu"` embedding set.
#'
#' @keywords internal
.fast_mnn_gpu <- function(
  object,
  batch_column,
  no_embd_to_use,
  fastmnn_params,
  seed,
  .verbose
) {
  # checks
  checkmate::assertTRUE(
    S7::S7_inherits(object, bixverse::SingleCells) ||
      S7::S7_inherits(object, bixverse::SingleCellsSubset)
  )
  checkmate::qassert(batch_column, "S1")
  checkmate::qassert(no_embd_to_use, c("I1[1,)", "0"))
  assertScFastmnnGpuParams(fastmnn_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # hard tier: the corrected embedding is written back onto the object
  assert_sc_state(object, artefacts = "pca")

  embd <- get_pca_factors(object)

  if (is.null(embd)) {
    warning("No PCA embeddings found in the object. Returning class as is.")
    return(object)
  }

  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take, drop = FALSE]
  }

  batch_index <- as.integer(factor(object[[batch_column]][[1]])) - 1L
  checkmate::assertTRUE(length(batch_index) == nrow(embd))

  # Rust errors on a single batch; returning the object matches bbknn_gpu_sc()
  if (length(unique(batch_index)) < 2L) {
    warning("The batch column only has one batch. Returning object as is.")
    return(object)
  }

  if (.verbose) {
    message("Running fastMNN on the GPU.")
  }

  # `"nndescent"` is the package-wide name, Rust only knows `"nndescent_gpu"`
  fastmnn_params[["knn_method"]] <- .normalise_gpu_knn_method(
    fastmnn_params[["knn_method"]]
  )

  mnn_embd <- rs_fast_mnn_gpu(
    embd = embd,
    batch_labels = batch_index,
    fastmnn_params = fastmnn_params,
    seed = seed,
    verbose = parse_verbosity(.verbose)
  )

  colnames(mnn_embd) <- sprintf("mnn_gpu_%s", seq_len(ncol(mnn_embd)))

  set_embedding(
    x = object,
    embd = mnn_embd,
    name = "mnn_gpu",
    from = "pca"
  )
}
