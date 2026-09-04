# ------------------------------------------------------------------------------
# GPU-accelerated SCENIC:
# - Wraps `rs_scenic_grn_gpu`, `rs_scenic_grn_streaming_gpu` and
#   `rs_mc_scenic_gpu`.
# - Reuses bixverse's CPU `scenic_gene_filter_sc` for target shortlisting.
# - GBM learners are rejected early because the Rust GPU driver does not
#   support them.
# ------------------------------------------------------------------------------

# scenic grn (gpu) -------------------------------------------------------------

#' GPU-accelerated SCENIC GRN inference
#'
#' @description
#' GPU counterpart of [bixverse::scenic_grn_sc()]. Runs multi-output tree
#' regression on the WGPU backend and returns a `ScenicGrn` object. Dispatches
#' on `SingleCells` (disk-backed .bin counts) or `MetaCells` (in-memory
#' sparse) from the `bixverse` package. GBM is not supported on GPU; use the
#' CPU version for `learner_type = "grnboost2"`.
#'
#' If `genes_to_take` is `NULL`, the CPU [bixverse::scenic_gene_filter_sc()] is
#' used to shortlist targets (cheap min-counts / min-cells scan).
#'
#' @param object `SingleCells` or `MetaCells` class from `bixverse`.
#' @param tf_ids Character vector. Gene identifiers of the transcription
#' factors to use as predictors.
#' @param scenic_params Named list. See [bixverse::params_scenic()].
#' @param wave_byte_budget Numeric. VRAM ceiling for per-wave histogram and
#' cumulative tensors (bytes). Default 4 GiB. Shrink on tight VRAM budgets,
#' raise on 16 GB+ adapters to let the scheduler pick a wider wave.
#' @param genes_to_take Optional character vector. Target genes to include.
#' If `NULL`, the CPU gene filter runs first.
#' @param cells_to_take Optional character vector. Cell names to include. If
#' `NULL`, all filtered cells are used.
#' @param streaming Optional boolean. Only used on `SingleCells`. If `TRUE`,
#' the streaming GPU driver is used (bounded host memory). If `NULL`, is
#' auto-picked from cell count via bixverse's internal `auto_streaming`.
#' Ignored for `MetaCells`.
#' @param random_seed Integer. For reproducibility.
#' @param .verbose Boolean or integer. Controls verbosity. `FALSE` -> quiet,
#' `TRUE` or `1L` -> normal, `2L` -> detailed.
#'
#' @returns A `ScenicGrn` object with the gene x TF importance matrix.
#'
#' @references Aibar et al., Nat Methods, 2017.
#'
#' @export
scenic_grn_sc_gpu <- S7::new_generic(
  name = "scenic_grn_sc_gpu",
  dispatch_args = "object",
  fun = function(
    object,
    tf_ids,
    scenic_params = bixverse::params_scenic(),
    wave_byte_budget = 4 * 1024^3,
    genes_to_take = NULL,
    cells_to_take = NULL,
    streaming = NULL,
    random_seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method scenic_grn_sc_gpu SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(scenic_grn_sc_gpu, SingleCells) <- function(
  object,
  tf_ids,
  scenic_params = bixverse::params_scenic(),
  wave_byte_budget = 4 * 1024^3,
  genes_to_take = NULL,
  cells_to_take = NULL,
  streaming = NULL,
  random_seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(tf_ids, "S+")
  bixverse:::assertScenicParams(scenic_params)
  checkmate::qassert(wave_byte_budget, "N1(0,)")
  checkmate::qassert(genes_to_take, c("S+", "0"))
  checkmate::qassert(cells_to_take, c("S+", "0"))
  checkmate::qassert(streaming, c("B1", "0"))
  checkmate::qassert(random_seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (scenic_params$learner_type == "grnboost2") {
    stop(
      "GRNBoost2 (gradient boosting) is not supported on GPU. ",
      "Use bixverse::scenic_grn_sc() for the CPU implementation."
    )
  }

  # resolve cells
  if (is.null(cells_to_take)) {
    cells_to_take <- get_cell_names(object, filtered = TRUE)
  }

  streaming <- bixverse:::auto_streaming(
    n_cells = length(cells_to_take),
    streaming = streaming,
    .verbose = .verbose
  )

  cell_indices <- get_cell_indices(
    object,
    cell_ids = cells_to_take,
    rust_index = TRUE
  )

  # resolve target genes
  if (is.null(genes_to_take)) {
    if (.verbose) {
      message("No target genes supplied, running CPU gene filter...")
    }
    genes_to_take <- bixverse::scenic_gene_filter_sc(
      object,
      scenic_params = scenic_params,
      cells_to_take = cells_to_take,
      .verbose = bixverse:::parse_verbosity(.verbose)
    )
  }

  gene_indices <- get_gene_indices(
    object,
    gene_ids = genes_to_take,
    rust_index = TRUE
  )

  # resolve TFs, drop missing
  all_gene_names <- get_gene_names(object)
  tf_found <- tf_ids[tf_ids %in% all_gene_names]
  n_dropped <- length(tf_ids) - length(tf_found)
  if (n_dropped > 0 && .verbose) {
    warning(sprintf(
      "%d TF identifier(s) not found in the object and dropped.",
      n_dropped
    ))
  }

  if (length(tf_found) == 0) {
    stop("No provided TF identifiers match genes in the object.")
  }

  tf_all_indices <- get_gene_indices(
    object,
    gene_ids = tf_found,
    rust_index = TRUE
  )

  # TFs must survive the target-gene filter
  tf_indices_red <- intersect(tf_all_indices, gene_indices)

  if (length(tf_indices_red) == 0) {
    stop(
      "No TFs remain after intersecting with target gene indices. ",
      "Consider relaxing min_counts / min_cells thresholds."
    )
  }

  if (.verbose) {
    message(sprintf(
      "SCENIC GPU: %d target genes, %d TFs, %d cells (streaming: %s)",
      length(gene_indices),
      length(tf_indices_red),
      length(cell_indices),
      streaming
    ))
  }

  scenic_fn <- if (streaming) {
    rs_scenic_grn_streaming_gpu
  } else {
    rs_scenic_grn_gpu
  }

  importance_matrix <- scenic_fn(
    f_path_genes = bixverse:::get_rust_count_gene_f_path(object),
    cell_indices = cell_indices,
    gene_indices = gene_indices,
    tf_indices = as.integer(tf_indices_red),
    scenic_params = scenic_params,
    wave_byte_budget = wave_byte_budget,
    seed = random_seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  tf_names <- get_gene_names_from_idx(
    object,
    gene_idx = as.integer(tf_indices_red),
    rust_based = TRUE
  )
  rownames(importance_matrix) <- genes_to_take
  colnames(importance_matrix) <- tf_names

  new_scenic_grn(
    importance_matrix = importance_matrix,
    gene_ids = genes_to_take,
    tf_ids = tf_names,
    params = scenic_params
  )
}

## MetaCells ------------------------------------------------------------------

#' @method scenic_grn_sc_gpu MetaCells
#'
#' @import bixverse
#'
#' @export
S7::method(scenic_grn_sc_gpu, MetaCells) <- function(
  object,
  tf_ids,
  scenic_params = bixverse::params_scenic(),
  wave_byte_budget = 4 * 1024^3,
  genes_to_take = NULL,
  cells_to_take = NULL,
  streaming = NULL,
  random_seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, MetaCells))
  checkmate::qassert(tf_ids, "S+")
  bixverse:::assertScenicParams(scenic_params)
  checkmate::qassert(wave_byte_budget, "N1(0,)")
  checkmate::qassert(genes_to_take, c("S+", "0"))
  checkmate::qassert(cells_to_take, c("S+", "0"))
  checkmate::qassert(streaming, c("B1", "0"))
  checkmate::qassert(random_seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (scenic_params$learner_type == "grnboost2") {
    stop(
      "GRNBoost2 (gradient boosting) is not supported on GPU. ",
      "Use bixverse::scenic_grn_sc() for the CPU implementation."
    )
  }

  # streaming is a no-op for the in-memory MetaCells path
  if (isTRUE(streaming) && .verbose) {
    message("`streaming` is ignored for MetaCells (in-memory driver).")
  }

  # resolve cells
  cell_indices <- if (is.null(cells_to_take)) {
    NULL
  } else {
    get_cell_indices(object, cell_ids = cells_to_take, rust_index = FALSE)
  }

  # lower default leaf size for meta cells, same rationale as CPU version
  if (scenic_params$min_samples_leaf >= 20) {
    if (.verbose) {
      message(paste(
        "The mean leafs per sample is set quite high for meta cells.",
        "Reducing to 10L."
      ))
    }
    scenic_params$min_samples_leaf <- 10L
  }

  # resolve target genes
  if (is.null(genes_to_take)) {
    if (.verbose) {
      message("No target genes supplied, running CPU gene filter...")
    }
    genes_to_take <- bixverse::scenic_gene_filter_sc(
      object,
      scenic_params = scenic_params,
      cells_to_take = cells_to_take,
      .verbose = .verbose
    )
  }

  gene_idx <- get_gene_indices(
    object,
    gene_ids = genes_to_take,
    rust_index = FALSE
  )

  # resolve TFs
  all_gene_names <- S7::prop(object, "var_table")$gene_id
  tf_found <- tf_ids[tf_ids %in% all_gene_names]
  n_dropped <- length(tf_ids) - length(tf_found)
  if (n_dropped > 0 && .verbose) {
    warning(sprintf(
      "%d TF identifier(s) not found in the object and dropped.",
      n_dropped
    ))
  }

  if (length(tf_found) == 0) {
    stop("No provided TF identifiers match genes in the object.")
  }

  tf_in_targets <- intersect(tf_found, genes_to_take)

  if (length(tf_in_targets) == 0) {
    stop(
      "No TFs remain after intersecting with target gene indices. ",
      "Consider relaxing min_counts / min_cells thresholds."
    )
  }

  sparse_data <- mc_counts_to_list(
    object,
    cell_indices = cell_indices,
    gene_indices = gene_idx,
    assay = "raw"
  )

  # 0-indexed positions within the filtered matrix
  tf_indices_rust <- match(tf_in_targets, genes_to_take) - 1L

  if (.verbose) {
    n_cells <- if (is.null(cell_indices)) {
      S7::prop(object, "dims")[1]
    } else {
      length(cell_indices)
    }
    message(sprintf(
      "SCENIC GPU: %d target genes, %d TFs, %d cells",
      length(genes_to_take),
      length(tf_indices_rust),
      n_cells
    ))
  }

  importance_matrix <- rs_mc_scenic_gpu(
    sparse_data = sparse_data,
    tf_indices = as.integer(tf_indices_rust),
    scenic_params = scenic_params,
    wave_byte_budget = wave_byte_budget,
    seed = random_seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  rownames(importance_matrix) <- genes_to_take
  colnames(importance_matrix) <- tf_in_targets

  new_scenic_grn(
    importance_matrix = importance_matrix,
    gene_ids = genes_to_take,
    tf_ids = tf_in_targets,
    params = scenic_params
  )
}
