# ------------------------------------------------------------------------------
# GPU-accelerated NMF:
# - Wraps `rs_nmf_*_sc_gpu` and `rs_nmf_*_mc_gpu`.
# - Reuses bixverse's `params_nmf_hals()` / `params_nmf_consensus()`, its
#   selection resolvers and its result constructors, so the objects that come
#   back are the same classes the CPU functions return.
# - Only the HALS solver runs on the device. The consensus clustering, the
#   density filter and the NNDSVD initialisation stay on the CPU.
# ------------------------------------------------------------------------------

# nmf (gpu) --------------------------------------------------------------------

## rank cap --------------------------------------------------------------------

#' Maximum rank the GPU HALS kernels support
#'
#' @description
#' The sweep kernels tier their workgroup width by rank and stop at 128, where
#' the width is already down to one SIMD group on Apple Silicon.
#'
#' @keywords internal
NMF_GPU_MAX_RANK <- 128L

#' Check the rank against the GPU solver cap
#'
#' @description
#' Above [NMF_GPU_MAX_RANK] the Rust side refuses the solve, so catch it here
#' with something that says what to do instead.
#'
#' @param k Integer vector. The rank(s) to check.
#'
#' @returns `NULL`, invisibly. Called for the error.
#'
#' @keywords internal
.assert_nmf_gpu_rank <- function(k) {
  checkmate::assertIntegerish(k, lower = 1L, min.len = 1L, any.missing = FALSE)

  if (max(k) > NMF_GPU_MAX_RANK) {
    stop(sprintf(
      paste(
        "The GPU NMF solver caps the rank at %d, but %d was requested.",
        "Lower k, or use the CPU version in bixverse (nmf_sc(),",
        "consensus_nmf_sc(), nmf_k_sweep_sc()), which has no such cap."
      ),
      NMF_GPU_MAX_RANK,
      max(k)
    ))
  }

  invisible(NULL)
}

## single run ------------------------------------------------------------------

#' Run single-run NMF on the GPU over single cell or meta cell data
#'
#' @description
#' GPU counterpart of [bixverse::nmf_sc()]. Runs one HALS NMF on a chosen subset
#' of cells and genes. The counts are uploaded once and the whole HALS loop runs
#' on the WGPU backend; the NNDSVD initialisation stays on the CPU.
#'
#' For `SingleCells` the counts are streamed from the Rust binary files, for
#' `MetaCells` the in-memory sparse counts are used. Params, result class and
#' downstream code are identical to the CPU version.
#'
#' A single run is here for parity rather than speed. The GPU pays off when the
#' same matrix serves many solves, so reach for [consensus_nmf_gpu_sc()] or
#' [nmf_k_sweep_gpu_sc()] if you want the speed-up.
#'
#' @param object `SingleCells` or `MetaCells` class from `bixverse`.
#' @param k Integer. Number of latent factors to return. At most 128, see
#' [NMF_GPU_MAX_RANK].
#' @param cell_ids Optional character. Cell ids (or meta cell ids) to restrict
#' the NMF to. If `NULL`, uses [bixverse::get_cells_to_keep()] for `SingleCells`
#' and all meta cells for `MetaCells`.
#' @param gene_ids Optional character. Gene ids to restrict the NMF to. If
#' `NULL`, uses [bixverse::get_hvg()] on the object.
#' @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
#' @param use_second_layer Boolean. If `TRUE`, runs NMF on the normalised
#' counts (recommended); if `FALSE`, on the raw counts.
#' @param nmf_hals_params List, see [bixverse::params_nmf_hals()].
#' @param seed Integer. Random seed for initialisation.
#' @param .verbose Boolean or integer. Controls verbosity. `FALSE` -> quiet,
#' `TRUE` or `1L` -> normal verbosity, `2L` -> detailed verbosity.
#'
#' @returns An `NmfResult` object, the same class [bixverse::nmf_sc()] returns.
#'
#' @export
nmf_gpu_sc <- S7::new_generic(
  name = "nmf_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    k,
    cell_ids = NULL,
    gene_ids = NULL,
    preprocessing = "none",
    use_second_layer = TRUE,
    nmf_hals_params = bixverse::params_nmf_hals(),
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

#' @method nmf_gpu_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(nmf_gpu_sc, SingleCells) <- function(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(k, "I1[1,)")
  .assert_nmf_gpu_rank(k)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_sc_nmf_selection(object, cell_ids, gene_ids)

  nmf_res <- rs_nmf_single_sc_gpu(
    f_path_gene = bixverse:::get_rust_count_gene_f_path(object),
    gene_indices = sel$gene_indices,
    cell_indices = sel$cell_indices,
    k = k,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  params <- c(
    nmf_hals_params,
    list(
      k = k,
      preprocessing = preprocessing,
      use_second_layer = use_second_layer,
      seed = seed
    )
  )

  bixverse::new_nmf_result(
    nmf_res = nmf_res,
    gene_ids = sel$gene_ids,
    cell_ids = sel$cell_ids,
    cell_indices = sel$cell_indices,
    source_class = "SingleCells",
    params = params
  )
}

#' @method nmf_gpu_sc MetaCells
#'
#' @import bixverse
#'
#' @export
S7::method(nmf_gpu_sc, MetaCells) <- function(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, MetaCells))
  checkmate::qassert(k, "I1[1,)")
  .assert_nmf_gpu_rank(k)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_mc_nmf_selection(object, cell_ids, gene_ids)

  count_list <- .mc_nmf_counts(object, sel, use_second_layer)

  nmf_res <- rs_nmf_single_mc_gpu(
    sparse_data = count_list,
    k = k,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  params <- c(
    nmf_hals_params,
    list(
      k = k,
      preprocessing = preprocessing,
      use_second_layer = use_second_layer,
      seed = seed
    )
  )

  bixverse::new_nmf_result(
    nmf_res = nmf_res,
    gene_ids = sel$gene_ids,
    cell_ids = sel$cell_ids,
    cell_indices = sel$cell_indices_rust,
    source_class = "MetaCells",
    params = params
  )
}

## stabilised ------------------------------------------------------------------

#' Run stabilised (multi-run) NMF on the GPU over single cell or meta cell data
#'
#' @description
#' GPU counterpart of [bixverse::stabilised_nmf_sc()]. Runs `n_runs` HALS NMF
#' with random initialisations seeded by `seed + i`. The `nmf_init` field in
#' `nmf_hals_params` is ignored; random init is always used.
#'
#' The counts upload once and serve every restart, but the restarts themselves
#' run one after the other on the single device, where the CPU version spreads
#' them across cores. On a small matrix the CPU can still win.
#'
#' @inheritParams nmf_gpu_sc
#' @param k Integer. Number of latent factors per run. At most 128, see
#' [NMF_GPU_MAX_RANK].
#' @param n_runs Integer. Number of random restarts.
#'
#' @returns A `StabilisedNmfResult` object, the same class
#' [bixverse::stabilised_nmf_sc()] returns.
#'
#' @export
stabilised_nmf_gpu_sc <- S7::new_generic(
  name = "stabilised_nmf_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    k,
    cell_ids = NULL,
    gene_ids = NULL,
    preprocessing = "none",
    use_second_layer = TRUE,
    nmf_hals_params = bixverse::params_nmf_hals(),
    n_runs = 30L,
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

#' @method stabilised_nmf_gpu_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(stabilised_nmf_gpu_sc, SingleCells) <- function(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(k, "I1[1,)")
  .assert_nmf_gpu_rank(k)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  checkmate::qassert(n_runs, "I1[1,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_sc_nmf_selection(object, cell_ids, gene_ids)

  nmf_res <- rs_nmf_multi_sc_gpu(
    f_path_gene = bixverse:::get_rust_count_gene_f_path(object),
    gene_indices = sel$gene_indices,
    cell_indices = sel$cell_indices,
    k = k,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    n_runs = n_runs,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  params <- c(
    nmf_hals_params,
    list(
      k = k,
      preprocessing = preprocessing,
      use_second_layer = use_second_layer,
      n_runs = n_runs,
      seed = seed
    )
  )

  bixverse::new_stabilised_nmf_result(
    nmf_res = nmf_res,
    gene_ids = sel$gene_ids,
    cell_ids = sel$cell_ids,
    cell_indices = sel$cell_indices,
    source_class = "SingleCells",
    params = params
  )
}

#' @method stabilised_nmf_gpu_sc MetaCells
#'
#' @import bixverse
#'
#' @export
S7::method(stabilised_nmf_gpu_sc, MetaCells) <- function(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, MetaCells))
  checkmate::qassert(k, "I1[1,)")
  .assert_nmf_gpu_rank(k)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  checkmate::qassert(n_runs, "I1[1,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_mc_nmf_selection(object, cell_ids, gene_ids)

  count_list <- .mc_nmf_counts(object, sel, use_second_layer)

  nmf_res <- rs_nmf_multi_mc_gpu(
    sparse_data = count_list,
    k = k,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    n_runs = n_runs,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  params <- c(
    nmf_hals_params,
    list(
      k = k,
      preprocessing = preprocessing,
      use_second_layer = use_second_layer,
      n_runs = n_runs,
      seed = seed
    )
  )

  bixverse::new_stabilised_nmf_result(
    nmf_res = nmf_res,
    gene_ids = sel$gene_ids,
    cell_ids = sel$cell_ids,
    cell_indices = sel$cell_indices_rust,
    source_class = "MetaCells",
    params = params
  )
}

## consensus -------------------------------------------------------------------

#' Run consensus NMF on the GPU over single cell or meta cell data
#'
#' @description
#' GPU counterpart of [bixverse::consensus_nmf_sc()]. Runs `n_runs` HALS
#' restarts on the device, pools their components, drops unstable ones by local
#' density, k-means clusters the survivors and refits the partner factor against
#' the per-cluster median.
#'
#' Prefer this over [stabilised_nmf_gpu_sc()], which picks the lowest-loss
#' restart. Use [nmf_k_sweep_gpu_sc()] first if you do not already know `k`.
#'
#' @details
#' Only the restarts move to the GPU. The pooling, the density filter, the
#' k-means and the silhouette all run on the CPU, shared with the bixverse
#' implementation, so the speed-up tracks how much of the run the solves own.
#'
#' The restart factors are dense and all held at once, so budget for `n_runs`
#' times `k` times the cell count on top of the counts themselves.
#'
#' If the density filter leaves fewer than `k` components, or a cluster comes
#' out empty, the run errors rather than returning a partial answer. Raise
#' `density_threshold` (2 switches the filter off) or increase `n_runs`.
#'
#' @inheritParams nmf_gpu_sc
#' @param k Integer. Number of latent factors. At least 2, at most 128, see
#' [NMF_GPU_MAX_RANK].
#' @param nmf_hals_params List, see [bixverse::params_nmf_hals()]. The
#' `nmf_init` field is ignored, restarts always use random initialisation.
#' @param nmf_consensus_params List, see [bixverse::params_nmf_consensus()].
#' @param n_runs Integer. Number of restarts. At least 2.
#'
#' @returns A `ConsensusNmfResult` object, the same class
#' [bixverse::consensus_nmf_sc()] returns.
#'
#' @references Kotliar et al., eLife, 2019
#'
#' @export
consensus_nmf_gpu_sc <- S7::new_generic(
  name = "consensus_nmf_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    k,
    cell_ids = NULL,
    gene_ids = NULL,
    preprocessing = "none",
    use_second_layer = TRUE,
    nmf_hals_params = bixverse::params_nmf_hals(),
    nmf_consensus_params = bixverse::params_nmf_consensus(),
    n_runs = 30L,
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

#' @method consensus_nmf_gpu_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(consensus_nmf_gpu_sc, SingleCells) <- function(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  nmf_consensus_params = bixverse::params_nmf_consensus(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(k, "I1[2,)")
  .assert_nmf_gpu_rank(k)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  bixverse:::assertNmfConsensus(nmf_consensus_params)
  checkmate::qassert(n_runs, "I1[2,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_sc_nmf_selection(object, cell_ids, gene_ids)

  bixverse:::.warn_consensus_target_w(
    nmf_consensus_params,
    n_samples = length(sel$cell_indices),
    k = k,
    n_runs = n_runs
  )

  nmf_res <- bixverse:::.run_consensus_nmf(
    .rs_call = rs_nmf_consensus_sc_gpu,
    nmf_consensus_params = nmf_consensus_params,
    seed = seed,
    f_path_gene = bixverse:::get_rust_count_gene_f_path(object),
    gene_indices = sel$gene_indices,
    cell_indices = sel$cell_indices,
    k = k,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    n_runs = n_runs,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  params <- c(
    nmf_hals_params,
    list(
      k = k,
      preprocessing = preprocessing,
      use_second_layer = use_second_layer,
      nmf_consensus_params = nmf_consensus_params,
      n_runs = n_runs,
      seed = seed
    )
  )

  bixverse::new_consensus_nmf_result(
    nmf_res = nmf_res,
    gene_ids = sel$gene_ids,
    cell_ids = sel$cell_ids,
    cell_indices = sel$cell_indices,
    source_class = "SingleCells",
    params = params
  )
}

#' @method consensus_nmf_gpu_sc MetaCells
#'
#' @import bixverse
#'
#' @export
S7::method(consensus_nmf_gpu_sc, MetaCells) <- function(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  nmf_consensus_params = bixverse::params_nmf_consensus(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, MetaCells))
  checkmate::qassert(k, "I1[2,)")
  .assert_nmf_gpu_rank(k)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  bixverse:::assertNmfConsensus(nmf_consensus_params)
  checkmate::qassert(n_runs, "I1[2,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_mc_nmf_selection(object, cell_ids, gene_ids)

  bixverse:::.warn_consensus_target_w(
    nmf_consensus_params,
    n_samples = length(sel$cell_indices_1b),
    k = k,
    n_runs = n_runs
  )

  count_list <- .mc_nmf_counts(object, sel, use_second_layer)

  nmf_res <- bixverse:::.run_consensus_nmf(
    .rs_call = rs_nmf_consensus_mc_gpu,
    nmf_consensus_params = nmf_consensus_params,
    seed = seed,
    sparse_data = count_list,
    k = k,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    n_runs = n_runs,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  params <- c(
    nmf_hals_params,
    list(
      k = k,
      preprocessing = preprocessing,
      use_second_layer = use_second_layer,
      nmf_consensus_params = nmf_consensus_params,
      n_runs = n_runs,
      seed = seed
    )
  )

  bixverse::new_consensus_nmf_result(
    nmf_res = nmf_res,
    gene_ids = sel$gene_ids,
    cell_ids = sel$cell_ids,
    cell_indices = sel$cell_indices_rust,
    source_class = "MetaCells",
    params = params
  )
}

## k sweep ---------------------------------------------------------------------

#' Sweep k for consensus NMF on the GPU over single cell or meta cell data
#'
#' @description
#' GPU counterpart of [bixverse::nmf_k_sweep_sc()]. Runs the consensus step
#' across a range of ranks and reports stability against reconstruction error,
#' keeping no factors. Pick the last `k` before stability falls away while the
#' error curve is still coming down, then fit there with
#' [consensus_nmf_gpu_sc()].
#'
#' @details
#' This is the shape the GPU path is really for. The counts upload once and
#' serve all `length(k_range) * n_runs` solves, where the CPU pays full memory
#' traffic over the matrix for every one of them. The scratch is sized once at
#' the largest rank in `k_range`.
#'
#' It is a diagnostic, so it leaves the object alone and hands the result back
#' directly. `plot()` on it gives you the two curves.
#'
#' @inheritParams consensus_nmf_gpu_sc
#' @param k_range Integer vector. The ranks to evaluate. Every entry at least 2
#' and at most 128, see [NMF_GPU_MAX_RANK].
#'
#' @returns An `NmfKSweepResult`, which is a data.table with one row per `k`.
#'
#' @references Kotliar et al., eLife, 2019
#'
#' @export
nmf_k_sweep_gpu_sc <- S7::new_generic(
  name = "nmf_k_sweep_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    k_range,
    cell_ids = NULL,
    gene_ids = NULL,
    preprocessing = "none",
    use_second_layer = TRUE,
    nmf_hals_params = bixverse::params_nmf_hals(),
    nmf_consensus_params = bixverse::params_nmf_consensus(),
    n_runs = 30L,
    seed = 42L,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

#' @method nmf_k_sweep_gpu_sc SingleCells
#'
#' @import bixverse
#'
#' @export
S7::method(nmf_k_sweep_gpu_sc, SingleCells) <- function(
  object,
  k_range,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  nmf_consensus_params = bixverse::params_nmf_consensus(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  k_range <- bixverse:::.assert_nmf_k_range(k_range)
  .assert_nmf_gpu_rank(k_range)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  bixverse:::assertNmfConsensus(nmf_consensus_params)
  checkmate::qassert(n_runs, "I1[2,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_sc_nmf_selection(object, cell_ids, gene_ids)

  bixverse:::.warn_consensus_target_w(
    nmf_consensus_params,
    n_samples = length(sel$cell_indices),
    k = max(k_range),
    n_runs = n_runs
  )

  sweep_res <- rs_nmf_k_sweep_sc_gpu(
    f_path_gene = bixverse:::get_rust_count_gene_f_path(object),
    gene_indices = sel$gene_indices,
    cell_indices = sel$cell_indices,
    k_range = k_range,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    nmf_consensus_params = bixverse:::.inject_consensus_seed(
      nmf_consensus_params,
      seed
    ),
    n_runs = n_runs,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  bixverse::new_nmf_k_sweep_result(
    sweep_res = sweep_res,
    source_class = "SingleCells",
    params = c(
      nmf_hals_params,
      list(
        k_range = k_range,
        preprocessing = preprocessing,
        use_second_layer = use_second_layer,
        nmf_consensus_params = nmf_consensus_params,
        n_runs = n_runs,
        seed = seed
      )
    )
  )
}

#' @method nmf_k_sweep_gpu_sc MetaCells
#'
#' @import bixverse
#'
#' @export
S7::method(nmf_k_sweep_gpu_sc, MetaCells) <- function(
  object,
  k_range,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  nmf_consensus_params = bixverse::params_nmf_consensus(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, MetaCells))
  k_range <- bixverse:::.assert_nmf_k_range(k_range)
  .assert_nmf_gpu_rank(k_range)
  checkmate::qassert(cell_ids, c("0", "S+"))
  checkmate::qassert(gene_ids, c("0", "S+"))
  checkmate::assertChoice(preprocessing, c("none", "sd", "sqrt_sd"))
  checkmate::qassert(use_second_layer, "B1")
  bixverse:::assertNmfHals(nmf_hals_params)
  bixverse:::assertNmfConsensus(nmf_consensus_params)
  checkmate::qassert(n_runs, "I1[2,)")
  checkmate::qassert(seed, "I1")
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  # function body
  sel <- bixverse:::.resolve_mc_nmf_selection(object, cell_ids, gene_ids)

  bixverse:::.warn_consensus_target_w(
    nmf_consensus_params,
    n_samples = length(sel$cell_indices_1b),
    k = max(k_range),
    n_runs = n_runs
  )

  count_list <- .mc_nmf_counts(object, sel, use_second_layer)

  sweep_res <- rs_nmf_k_sweep_mc_gpu(
    sparse_data = count_list,
    k_range = k_range,
    preprocessing = preprocessing,
    use_second_layer = use_second_layer,
    nmf_hals_params = nmf_hals_params,
    nmf_consensus_params = bixverse:::.inject_consensus_seed(
      nmf_consensus_params,
      seed
    ),
    n_runs = n_runs,
    seed = seed,
    verbose = bixverse:::parse_verbosity(.verbose)
  )

  bixverse::new_nmf_k_sweep_result(
    sweep_res = sweep_res,
    source_class = "MetaCells",
    params = c(
      nmf_hals_params,
      list(
        k_range = k_range,
        preprocessing = preprocessing,
        use_second_layer = use_second_layer,
        nmf_consensus_params = nmf_consensus_params,
        n_runs = n_runs,
        seed = seed
      )
    )
  )
}

## helpers ---------------------------------------------------------------------

#' Pull the meta cell counts for an NMF run
#'
#' @description
#' All four `MetaCells` methods slice the counts the same way, so the assay
#' choice lives here rather than four times over.
#'
#' @param object `MetaCells` class from `bixverse`.
#' @param sel List. Output of `bixverse:::.resolve_mc_nmf_selection()`.
#' @param use_second_layer Boolean. If `TRUE`, takes the normalised counts.
#'
#' @returns A named list with `data`, `indptr`, `indices`, `cs_type`, `nrow`
#' and `ncol`, ready for the Rust bindings.
#'
#' @keywords internal
.mc_nmf_counts <- function(object, sel, use_second_layer) {
  checkmate::assertList(sel)
  checkmate::qassert(use_second_layer, "B1")

  bixverse::mc_counts_to_list(
    object = object,
    cell_indices = sel$cell_indices_1b,
    gene_indices = sel$gene_indices_1b,
    assay = if (use_second_layer) "norm" else "raw"
  )
}
