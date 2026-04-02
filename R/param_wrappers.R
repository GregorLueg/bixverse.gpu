# param wrappers ---------------------------------------------------------------

## knn -------------------------------------------------------------------------

### cagra ----------------------------------------------------------------------

#' Default parameters for CAGRA-style kNN search
#'
#' @param k_query Integer. Number of neighbours to identify.
#' @param ann_dist Character. Distance metric to use. One of `"euclidean"` or
#' `"cosine"`.
#' @param k Optional integer. Final node degree of the CAGRA navigational
#' graph. If `NULL`, defaults to `30` on the Rust side.
#' @param k_build Optional integer. Number of k-neighbours during the
#' NNDescent build phase before CAGRA pruning. If `NULL`, defaults to `1.5 * k`
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
  refine_sweeps = 0L,
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

### ivf ------------------------------------------------------------------------

#' Default parameters for IVF-GPU kNN search
#'
#' @param k Integer. Number of neighbours to identify.
#' @param ann_dist Character. Distance metric to use. One of `"euclidean"` or
#' `"cosine"`.
#' @param nlist Optional integer. Number of clusters to partition the index
#' into. If `NULL`, defaults to `sqrt(n)`.
#' @param nprobe Optional integer. Number of clusters to probe at query time.
#' If `NULL`, defaults to `sqrt(nlist)`.
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

## parametric umap -------------------------------------------------------------

#' Wrapper function to generate parametric UMAP parameters
#'
#' @param local_connectivity Numeric. Number of nearest neighbours assumed to
#' be at distance zero. Defaults to `1.0`.
#' @param bandwidth Numeric. Convergence tolerance for smooth kNN distance
#' binary search. Defaults to `1e-5`.
#' @param mix_weight Numeric. Balance between fuzzy union and directed graph
#' during symmetrisation. Defaults to `1.0`.
#' @param hidden_layers Integer vector. Hidden layer sizes for the MLP encoder.
#' Defaults to `c(128L, 64L, 32L)`.
#' @param lr Numeric. Learning rate for the neural network optimiser. Defaults
#' to `0.001`.
#' @param corr_weight Numeric. Coefficient for the negative Pearson correlation
#' loss that encourages similar distances in embedding and original space.
#' Defaults to `0.0`.
#' @param n_epochs Integer. Number of training epochs. Defaults to `500L`.
#' @param batch_size Integer. Training batch size. Defaults to `256L`.
#' @param neg_sample_rate Integer. Number of negative samples per positive
#' edge. Defaults to `5L`.
#'
#' @returns A list with the parametric UMAP parameters.
#'
#' @export
params_parametric_umap <- function(
  local_connectivity = 1.0,
  bandwidth = 1e-5,
  mix_weight = 1.0,
  hidden_layers = c(128L, 64L, 32L),
  lr = 0.001,
  corr_weight = 0.0,
  n_epochs = 500L,
  batch_size = 256L,
  neg_sample_rate = 5L
) {
  # checks
  checkmate::qassert(local_connectivity, "N1")
  checkmate::qassert(bandwidth, "N1")
  checkmate::qassert(mix_weight, "N1")
  checkmate::assert_integerish(
    hidden_layers,
    lower = 1L,
    min.len = 1L,
    any.missing = FALSE
  )
  checkmate::qassert(lr, "N1(0,)")
  checkmate::qassert(corr_weight, "N1")
  checkmate::qassert(n_epochs, "I1[1,)")
  checkmate::qassert(batch_size, "I1[1,)")
  checkmate::qassert(neg_sample_rate, "I1[1,)")

  list(
    local_connectivity = local_connectivity,
    bandwidth = bandwidth,
    mix_weight = mix_weight,
    hidden_layers = as.integer(hidden_layers),
    lr = lr,
    corr_weight = corr_weight,
    n_epochs = as.integer(n_epochs),
    batch_size = as.integer(batch_size),
    neg_sample_rate = as.integer(neg_sample_rate)
  )
}
