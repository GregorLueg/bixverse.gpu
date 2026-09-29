# ------------------------------------------------------------------------------
# GPU-accelerated Bonsai:
# - Wraps `rs_sc_bonsai_gpu`, which runs Sanity on the device and the tree
#   search on the CPU.
# - Everything around the Rust call (candidate genes, cells, the timings) is
#   bixverse's own `.bonsai_sc_run()`, so the result is the same `BonsaiTree`
#   S3 object as `bixverse::bonsai_sc()` and its print, plot and relayout
#   methods work unchanged.
# - Only `SingleCells` gets a method, as for `bixverse::bonsai_sc()`.
# ------------------------------------------------------------------------------

# bonsai (gpu) -----------------------------------------------------------------

#' Build a Bonsai tree over the cells, Sanity on the GPU
#'
#' @description GPU counterpart of [bixverse::bonsai_sc()]. Sanity, which turns
#' the raw counts into posterior log fold changes with error bars and is most of
#' the runtime on the CPU, runs on the WGPU backend. Genes are streamed through
#' it in chunks and only the ones passing Bonsai's signal-to-noise filter are
#' kept. The tree search and the layout run on the CPU, as they do in
#' `bixverse`.
#'
#' The device computes in `f32`, so the posteriors match the CPU run to within
#' the resolution of Sanity's variance grid rather than bit for bit, and a gene
#' sitting right at the signal-to-noise threshold can end up on the other side.
#'
#' @param object `SingleCells` class.
#' @param hvg Optional integer. Restrict the candidate genes to these. Please
#' provide 1-indexed genes here! If `NULL`, every gene in the object is a
#' candidate.
#' @param bonsai_params List. See [bixverse::params_sc_bonsai()].
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @returns A `BonsaiTree` S3 object, see [bixverse::bonsai_sc()].
#'
#' @export
#'
#' @references de Groot, et al., Nat. Biotechnol., 2026; Breda, et al., Nat.
#' Biotechnol., 2021.
bonsai_gpu_sc <- S7::new_generic(
  name = "bonsai_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    hvg = NULL,
    bonsai_params = bixverse::params_sc_bonsai(),
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method bonsai_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(bonsai_gpu_sc, SingleCells) <- function(
  object,
  hvg = NULL,
  bonsai_params = bixverse::params_sc_bonsai(),
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, bixverse::SingleCells))

  bixverse:::.bonsai_sc_run(
    object = object,
    hvg = hvg,
    bonsai_params = bonsai_params,
    runner = rs_sc_bonsai_gpu,
    .verbose = .verbose
  )
}
