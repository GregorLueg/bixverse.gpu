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
  assertScFastmnnGpu(fastmnn_params)
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
