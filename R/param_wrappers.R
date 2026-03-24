# param wrappers ---------------------------------------------------------------

## cagra -----------------------------------------------------------------------

#' Default parameters for CAGRA-style kNN search
#'
#' @param k_query Integer. Number of neighbours to identify.
#' @param ann_dist Character. Distance metric to use. One of `"euclidean"` or
#' `"cosine"`.
#' @param k Optional integer. Final node degree of the CAGRA navigational
#' graph. If `NULL`, defaults to `30` on the Rust side.
#' @param k_build Optional integer. Number of k-neighbours during the
#' NNDescent build phase before CAGRA pruning. If `NULL`, defaults to `2 * k`
#' on the Rust side.
#' @param refine_sweeps Integer. Number of refinement sweeps during graph
#' generation.
#' @param max_iters Optional integer. Maximum iterations for the NNDescent
#' rounds. If `NULL`, determined automatically.
#' @param n_trees Optional integer. Number of trees to use in the initial
#' GPU-accelerated forest. If `NULL`, determined automatically.
#' @param delta Numeric. Termination criterion for the NNDescent iterations.
#' @param rho Optional numeric. Sampling rate during NNDescent iterations.
#' If `NULL`, determined automatically.
#' @param beam_width Optional integer. Beam width during querying. If `NULL`,
#' determined automatically.
#' @param max_beam_iters Optional integer. Maximum beam iterations. If `NULL`,
#' determined automatically.
#' @param n_entry_points Optional integer. Number of entry points into the
#' graph. If `NULL`, determined automatically.
#'
#' @return A list with the parameters.
#'
#' @export
params_sc_cagra <- function(
  k_query = 15L,
  ann_dist = "cosine",
  k = NULL,
  k_build = NULL,
  refine_sweeps = 2L,
  max_iters = NULL,
  n_trees = NULL,
  delta = 0.001,
  rho = NULL,
  beam_width = NULL,
  max_beam_iters = NULL,
  n_entry_points = NULL
) {
  # checks
  checkmate::qassert(k_query, "I1[1,)")
  checkmate::qassert(ann_dist, "S1")
  checkmate::assert_choice(ann_dist, c("euclidean", "cosine"))
  checkmate::qassert(k, c("I1[1,)", "0"))
  checkmate::qassert(k_build, c("I1[1,)", "0"))
  checkmate::qassert(refine_sweeps, "I1[0,)")
  checkmate::qassert(max_iters, c("I1[1,)", "0"))
  checkmate::qassert(n_trees, c("I1[1,)", "0"))
  checkmate::qassert(delta, "N1(0,)")
  checkmate::qassert(rho, c("N1(0,1]", "0"))
  checkmate::qassert(beam_width, c("I1[1,)", "0"))
  checkmate::qassert(max_beam_iters, c("I1[1,)", "0"))
  checkmate::qassert(n_entry_points, c("I1[1,)", "0"))
  # return
  list(
    k_query = k_query,
    ann_dist = ann_dist,
    k = k,
    k_build = k_build,
    refine_sweeps = refine_sweeps,
    max_iters = max_iters,
    n_trees = n_trees,
    delta = delta,
    rho = rho,
    beam_width = beam_width,
    max_beam_iters = max_beam_iters,
    n_entry_points = n_entry_points
  )
}

## ivf -------------------------------------------------------------------------

#' Default parameters for IVF-GPU kNN search
#'
#' @param k Integer. Number of neighbours to identify.
#' @param ann_dist Character. Distance metric to use. One of `"euclidean"` or
#' `"cosine"`.
#' @param nlist Optional integer. Number of clusters to partition the index
#' into. If `NULL`, defaults to √n.
#' @param nprobe Optional integer. Number of clusters to probe at query time.
#' If `NULL`, defaults to √nlist.
#' @param nquery Optional integer. Number of query vectors processed per GPU
#' batch. If `NULL`, defaults to 100,000.
#' @param max_iters Optional integer. Maximum k-means iterations during index
#' build. If `NULL`, defaults to 30.
#' @param seed Integer. Seed for k-means initialisation.
#'
#' @return A list with the parameters.
#'
#' @export
params_sc_ivf <- function(
  k = 15L,
  ann_dist = "cosine",
  nlist = NULL,
  nprobe = NULL,
  nquery = NULL,
  max_iters = NULL,
  seed = 42L
) {
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(ann_dist, "S1")
  checkmate::assert_choice(ann_dist, c("euclidean", "cosine"))
  checkmate::qassert(nlist, c("I1[1,)", "0"))
  checkmate::qassert(nprobe, c("I1[1,)", "0"))
  checkmate::qassert(nquery, c("I1[1,)", "0"))
  checkmate::qassert(max_iters, c("I1[1,)", "0"))
  checkmate::qassert(seed, "I1[0,)")

  list(
    k = k,
    ann_dist = ann_dist,
    nlist = nlist,
    nprobe = nprobe,
    nquery = nquery,
    max_iters = max_iters,
    seed = seed
  )
}
