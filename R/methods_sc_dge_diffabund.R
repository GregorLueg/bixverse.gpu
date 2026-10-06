# ------------------------------------------------------------------------------
# GPU-accelerated NEBULA:
# - Wraps `rs_nebula_sc_gpu`.
# - Mirrors `bixverse::nebula_sc()` and reuses its design, tested-coefficient
#   and result helpers, so the `ScNebula` result is the same class with the
#   same columns.
# - bixverse dispatches on its internal `ScOrScSubset` union. That union is not
#   exported, so the two exported classes get a method each and both delegate to
#   `.nebula_gpu()`.
# ------------------------------------------------------------------------------

# nebula (gpu) -----------------------------------------------------------------

#' Run NEBULA on single cells on the GPU
#'
#' @description
#' GPU counterpart of [bixverse::nebula_sc()]. Stage two of NEBULA, the
#' per-gene penalised fits, runs on the WGPU backend in `f32` and is finished on
#' the host in `f64`. Cell ordering, gene batching, the dispersion shrinkage and
#' the Wald test are the CPU code. Expect estimates close to the CPU ones, not
#' identical to them.
#'
#' REML is not implemented on the device, so [params_nebula_gpu()] does not
#' carry it. Everything else, the design handling, the result class and the
#' downstream code, is identical to the CPU version.
#'
#' @param object `SingleCells` or `SingleCellsSubset` class from `bixverse`.
#' @param subject_col String. The column in the obs table holding the subject
#' (donor) identifier. This is what the random effect is over.
#' @param design Formula. The experimental design, evaluated against the obs
#' table, e.g. `~ condition` or `~ condition + age`. Include the intercept.
#' @param coef Optional integer or character. Which coefficient of the design
#' the Wald test reports, as a 1-based column position or a column name.
#' Defaults to the last column.
#' @param contrast Optional numeric vector. One weight per design column.
#' Mutually exclusive with `coef`.
#' @param genes_to_use Optional character vector. The genes to fit. Defaults to
#' every gene in the object, which is usually too many.
#' @param offset Optional numeric vector. Strictly positive scaling factor per
#' cell, aligned to the cells that survive the design. Defaults to `NULL`,
#' which uses the library sizes.
#' @param nebula_params A list, see [params_nebula_gpu()]. The list has the
#' following parameters:
#' \itemize{
#'   \item nebula_method - String. One of `c("ln", "hl")`.
#'   \item min_sigma, max_sigma - Numeric. Bounds on the subject-level
#'   overdispersion.
#'   \item min_phi, max_phi - Numeric. Bounds on the cell-level overdispersion.
#'   \item cutoff_cell - Numeric. When to refit both overdispersions.
#'   \item kappa - Numeric. When to trust the stage-one subject overdispersion.
#'   \item cpc - Numeric. Minimum mean count per cell for a gene to be tested.
#'   \item mincp - Integer. Minimum number of cells expressing a gene.
#'   \item eps - Numeric. Optimiser stopping tolerance.
#'   \item gene_batch_size - Integer. Genes read and fitted per batch.
#'   \item shrink_dispersion - Boolean. Empirical Bayes shrinkage of the
#'   cell-level overdispersions.
#' }
#' @param .verbose Boolean or integer. Controls verbosity and returns run
#' times. `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` ->
#' detailed verbosity.
#'
#' @returns A `ScNebula` class, see `bixverse:::new_sc_nebula_res()`, with
#' \itemize{
#'   \item results - data.table. One row per gene that survived NEBULA's
#'   expression filter, with the Wald test and both overdispersions.
#'   \item coefficients - Numeric matrix of genes x coefficients.
#'   \item se - Numeric matrix of genes x coefficients.
#'   \item params - List. The parameters the run used.
#' }
#'
#' @export
#'
#' @references He, et al., Commun Biol, 2021
nebula_gpu_sc <- S7::new_generic(
  name = "nebula_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    subject_col,
    design,
    coef = NULL,
    contrast = NULL,
    genes_to_use = NULL,
    offset = NULL,
    nebula_params = params_nebula_gpu(),
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

S7::method(nebula_gpu_sc, SingleCells) <- function(
  object,
  subject_col,
  design,
  coef = NULL,
  contrast = NULL,
  genes_to_use = NULL,
  offset = NULL,
  nebula_params = params_nebula_gpu(),
  .verbose = TRUE
) {
  .nebula_gpu(
    object = object,
    subject_col = subject_col,
    design = design,
    coef = coef,
    contrast = contrast,
    genes_to_use = genes_to_use,
    offset = offset,
    nebula_params = nebula_params,
    .verbose = .verbose
  )
}

## SingleCellsSubset -----------------------------------------------------------

S7::method(nebula_gpu_sc, SingleCellsSubset) <- function(
  object,
  subject_col,
  design,
  coef = NULL,
  contrast = NULL,
  genes_to_use = NULL,
  offset = NULL,
  nebula_params = params_nebula_gpu(),
  .verbose = TRUE
) {
  .nebula_gpu(
    object = object,
    subject_col = subject_col,
    design = design,
    coef = coef,
    contrast = contrast,
    genes_to_use = genes_to_use,
    offset = offset,
    nebula_params = nebula_params,
    .verbose = .verbose
  )
}

## implementation --------------------------------------------------------------

#' GPU NEBULA implementation
#'
#' @description
#' Shared body of the [nebula_gpu_sc()] methods. Mirrors the CPU
#' `bixverse::nebula_sc()` method step for step and only swaps the Rust call.
#'
#' @inheritParams nebula_gpu_sc
#'
#' @returns A `ScNebula` class.
#'
#' @keywords internal
.nebula_gpu <- function(
  object,
  subject_col,
  design,
  coef,
  contrast,
  genes_to_use,
  offset,
  nebula_params,
  .verbose
) {
  # checks
  checkmate::assertTRUE(
    S7::S7_inherits(object, bixverse::SingleCells) ||
      S7::S7_inherits(object, bixverse::SingleCellsSubset)
  )
  checkmate::qassert(subject_col, "S1")
  checkmate::assertFormula(design)
  checkmate::qassert(genes_to_use, c("0", "S+"))
  checkmate::qassert(offset, c("0", "N+"))
  assertNebulaGpuParams(nebula_params)
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  obs <- bixverse::get_sc_obs(
    object,
    cols = unique(c("cell_idx", subject_col, all.vars(design))),
    filtered = TRUE
  )

  inputs <- bixverse:::.nebula_design(
    obs = obs,
    design = design,
    subject_col = subject_col
  )
  tested <- bixverse:::.resolve_tested_sc(
    design = inputs$design_mat,
    coef = coef,
    contrast = contrast
  )

  # obs cell_idx is 1-based, Rust wants 0-based global positions
  cells_to_keep <- as.integer(inputs$obs$cell_idx - 1L)

  if (!is.null(offset)) {
    checkmate::assertNumeric(
      offset,
      len = length(cells_to_keep),
      lower = .Machine$double.eps,
      any.missing = FALSE
    )
  }

  # Rust wants one subject label per cell in the store, not per selected cell.
  # The count comes off the store: a subset shares its parent's counts file and
  # keeps `cell_idx` in the parent's index space.
  n_cells_total <- bixverse::get_sc_rust_ptr(object)$get_shape()[1]
  subject_ids <- integer(n_cells_total)
  subject_ids[inputs$obs$cell_idx] <- as.integer(inputs$subject_fct) - 1L

  gene_indices <- if (is.null(genes_to_use)) {
    seq_along(bixverse::get_gene_names(object)) - 1L
  } else {
    bixverse::get_gene_indices(
      x = object,
      gene_ids = genes_to_use,
      rust_index = TRUE
    )
  }

  res <- rs_nebula_sc_gpu(
    f_path_genes = bixverse:::get_rust_count_gene_f_path(object),
    f_path_cells = bixverse:::get_rust_count_cell_f_path(object),
    cells_to_keep = cells_to_keep,
    gene_indices = as.integer(gene_indices),
    subject_ids = subject_ids,
    design = inputs$design_mat,
    offset = offset,
    nebula_params = c(nebula_params, tested),
    verbose = parse_verbosity(.verbose)
  )

  params <- list(
    subject_col = subject_col,
    design = deparse1(design),
    tested = if (is.null(tested$coef)) {
      "contrast"
    } else {
      colnames(inputs$design_mat)[tested$coef + 1L]
    },
    n_cells = length(cells_to_keep),
    n_subjects = nlevels(inputs$subject_fct),
    n_genes_requested = length(gene_indices),
    nebula_params = nebula_params
  )

  bixverse:::.nebula_res_to_class(
    res = res,
    gene_names = bixverse::get_gene_names(object),
    design_mat = inputs$design_mat,
    params = params
  )
}
