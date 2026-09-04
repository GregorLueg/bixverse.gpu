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
