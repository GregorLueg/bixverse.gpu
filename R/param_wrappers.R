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
  ann_dist = "euclidean",
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
  ann_dist = "euclidean",
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

## gpu-accelerated k means -----------------------------------------------------

#' Default parameters for GPU k-means
#'
#' @param k_means_iter Integer. Number of k-means iterations.
#' @param k_means_init Optional character. Initialisation method. One of
#' `"random"`, `"parallel"`, or `"plusplus"`. If `NULL`, determined on the
#' Rust side.
#' @param metric String. One of `c("euclidean", "cosine")`.
#' @param fixed Logical. Whether cluster centres are fixed after initialisation.
#' @param quantise Logical. Whether to quantise data to f16 before clustering.
#'
#' @return A list with the parameters.
#'
#' @export
params_kmeans_gpu <- function(
  k_means_iter = 50L,
  k_means_init = NULL,
  metric = c("euclidean", "cosine"),
  fixed = TRUE,
  quantise = FALSE
) {
  metric <- match.arg(metric)

  checkmate::qassert(k_means_iter, "I1[1,)")
  if (!is.null(k_means_init)) {
    checkmate::qassert(k_means_init, "S1")
    checkmate::assert_choice(k_means_init, c("random", "parallel", "plusplus"))
  }
  checkmate::qassert(fixed, "B1")
  checkmate::qassert(quantise, "B1")
  checkmate::assertChoice(metric, c("euclidean", "cosine"))

  list(
    k_means_iter = k_means_iter,
    k_means_init = k_means_init,
    metric = metric,
    fixed = fixed,
    quantise = quantise
  )
}

## umap gpu --------------------------------------------------------------------

### nearest neighbours ---------------------------------------------------------

#' Wrapper function to generate GPU nearest neighbour parameters
#'
#' @param dist_metric Character. The distance metric to use. Defaults to
#' `"euclidean"`.
#' @param n_list Optional integer. Number of clusters to use for IVF. If `NULL`,
#' will default to `sqrt(n)`.
#' @param n_probes Optional integer. Number of clusters to probe for IVF. If
#' `NULL`, will default to `sqrt(n_list)`.
#' @param k Optional integer. Number of nearest neighbours to search for.
#' @param k_build Optional integer. Number of nearest neighbours to use
#' during graph build.
#' @param n_tree Optional integer. Number of trees for graph build.
#' @param delta Float. Precision parameter for NN descent. Defaults to `0.001`.
#' @param rho Optional float. Sample rate parameter for NN descent.
#' @param beam_width Optional integer. Beam width for beam search.
#' @param max_beam_iters Optional integer. Maximum number of beam search
#' iterations.
#' @param n_entry_points Optional integer. Number of entry points for beam
#' search.
#'
#' @returns A list with the GPU nearest neighbour parameters.
#'
#' @export
#'
#'
params_nn_gpu <- function(
  dist_metric = c("euclidean", "cosine"),
  n_list = NULL,
  n_probes = NULL,
  k = NULL,
  k_build = NULL,
  n_tree = NULL,
  delta = 0.001,
  rho = NULL,
  beam_width = NULL,
  max_beam_iters = NULL,
  n_entry_points = NULL
) {
  dist_metric <- match.arg(dist_metric)

  # checks
  checkmate::assertChoice(dist_metric, c("euclidean", "cosine"))
  checkmate::qassert(n_list, c("I1", "0"))
  checkmate::qassert(n_probes, c("I1", "0"))
  checkmate::qassert(k, c("I1", "0"))
  checkmate::qassert(k_build, c("I1", "0"))
  checkmate::qassert(n_tree, c("I1", "0"))
  checkmate::qassert(delta, "N1")
  checkmate::qassert(rho, c("N1", "0"))
  checkmate::qassert(beam_width, c("I1", "0"))
  checkmate::qassert(max_beam_iters, c("I1", "0"))
  checkmate::qassert(n_entry_points, c("I1", "0"))

  # results
  list(
    dist_metric = dist_metric,
    n_list = n_list,
    n_probes = n_probes,
    k = k,
    k_build = k_build,
    n_tree = n_tree,
    delta = delta,
    rho = rho,
    beam_width = beam_width,
    max_beam_iters = max_beam_iters,
    n_entry_points = n_entry_points
  )
}

### umap specific --------------------------------------------------------------

#' Wrapper function to generate UMAP parameters (GPU version)
#'
#' @param local_connectivity Numeric. Number of nearest neighbours assumed to
#' be at distance zero. Defaults to `1.0`.
#' @param bandwidth Numeric. Convergence tolerance for smooth kNN distance
#' binary search. Defaults to `1e-5`.
#' @param mix_weight Numeric. Balance between fuzzy union and directed graph
#' during symmetrisation. Defaults to `1.0`.
#' @param lr Numeric. Learning rate. Defaults to `1.0`.
#' @param n_epochs Integer or `NULL`. Number of optimisation epochs. Defaults
#' to `NULL`, resolved downstream based on data size.
#' @param neg_sample_rate Integer. Number of negative samples per positive
#' sample. Defaults to `5L`.
#' @param gamma Numeric. Repulsion strength. Defaults to `1.0`.
#' @param optimiser Character. One of `"adam_gpu"`, `"sgd"`, `"adam"`, or
#' `"adam_parallel"`. Defaults to `"adam_gpu"`.
#' @param init Character. Embedding initialisation method. One of `"spectral"`,
#' `"pca"`, or `"random"`. Defaults to `"spectral"`.
#' @param randomised Logical. Use randomised SVD for PCA initialisation.
#' Defaults to `FALSE`.
#'
#' @returns A list with the UMAP parameters.
#'
#' @export
params_umap_gpu <- function(
  local_connectivity = 1.0,
  bandwidth = 1e-5,
  mix_weight = 1.0,
  lr = 1.0,
  n_epochs = NULL,
  neg_sample_rate = 5L,
  gamma = 1.0,
  optimiser = c("adam_gpu", "adam_parallel", "sgd", "adam"),
  init = c("spectral", "pca", "random"),
  randomised = FALSE
) {
  optimiser <- match.arg(optimiser)
  init <- match.arg(init)

  checkmate::qassert(local_connectivity, "N1")
  checkmate::qassert(bandwidth, "N1")
  checkmate::qassert(mix_weight, "N1")
  checkmate::qassert(lr, "N1")
  checkmate::assert(
    checkmate::checkNull(n_epochs),
    checkmate::checkInt(n_epochs, lower = 1L)
  )
  checkmate::qassert(neg_sample_rate, "I1")
  checkmate::qassert(gamma, "N1")
  checkmate::assertChoice(
    optimiser,
    c("adam_gpu", "sgd", "adam", "adam_parallel")
  )
  checkmate::assertChoice(init, c("spectral", "pca", "random"))
  checkmate::qassert(randomised, "B1")

  list(
    local_connectivity = local_connectivity,
    bandwidth = bandwidth,
    mix_weight = mix_weight,
    lr = lr,
    n_epochs = n_epochs,
    neg_sample_rate = neg_sample_rate,
    gamma = gamma,
    optimiser = optimiser,
    init = init,
    randomised = randomised
  )
}

## single cells ----------------------------------------------------------------

### harmony v2 GPU -------------------------------------------------------------

#' Default parameters for Harmony v2 GPU batch correction
#'
#' @param k Optional integer. Number of clusters for k-means clustering. If
#' not provided, it will be automatically determined as
#' `min(round(N / 30), 100)`.
#' @param sigma Numeric vector. Per-cluster diversity weights. Either a single
#' value (broadcast to all clusters) or a vector of length k.
#' @param theta Numeric vector. Per-variable diversity penalty. Must be a single
#' value; only one batch covariate is supported on the GPU path.
#' @param lambda Numeric vector. Ridge regression penalty for the linear model.
#' Typically a single value. Ignored when `use_dynamic_lambda = TRUE`.
#' @param max_iter_kmeans Integer. Maximum number of k-means Jacobi sweeps per
#' Harmony round.
#' @param max_iter_harmony Integer. Maximum number of Harmony outer iterations.
#' @param epsilon_kmeans Numeric. Convergence threshold for k-means clustering.
#' @param epsilon_harmony Numeric. Convergence threshold for Harmony.
#' @param window_size Integer. Number of previous iterations to consider when
#' checking convergence.
#' @param alpha Numeric. Scaling factor for dynamic lambda estimation. Must be
#' in (0, 1). Only relevant when `use_dynamic_lambda = TRUE`.
#' @param tau Numeric. Scaling factor for theta based on batch size. A value of
#' 0 disables batch-size scaling of theta.
#' @param batch_proportion_cutoff Numeric. Cutoff for pruning batches with small
#' proportions during ridge regression.
#' @param use_dynamic_lambda Boolean. If `TRUE`, lambda is estimated dynamically
#' per cluster instead of using the fixed `lambda` value.
#' @param csr_cube_count Integer. Number of parallel thread groups used when
#' building the level-CSR index on the GPU. Adjust for your hardware if needed.
#' @param k_means_iter Integer. Maximum number of k-means iterations for the
#' initial centroid computation.
#' @param k_means_init Optional string. Initialisation strategy for k-means.
#' @param fixed Boolean. If `TRUE`, centroids are fixed after initialisation.
#' @param quantise Boolean. If `TRUE`, quantises intermediate values to f16
#' during k-means.
#'
#' @return A list with the parameters.
#'
#' @export
params_sc_harmony_v2_gpu <- function(
  k = NULL,
  sigma = 0.1,
  theta = 2.0,
  lambda = 1.0,
  max_iter_kmeans = 4L,
  max_iter_harmony = 10L,
  epsilon_kmeans = 1e-3,
  epsilon_harmony = 1e-2,
  window_size = 3L,
  alpha = 0.2,
  tau = 0.0,
  batch_proportion_cutoff = 1e-5,
  use_dynamic_lambda = FALSE,
  csr_cube_count = 256L,
  k_means_iter = 30L,
  k_means_init = NULL,
  fixed = FALSE,
  quantise = FALSE
) {
  checkmate::qassert(k, c("I1[1,)", "0"))
  checkmate::qassert(sigma, "N+[0,)")
  checkmate::qassert(theta, "N+[0,)")
  checkmate::qassert(lambda, "N+[0,)")
  checkmate::qassert(max_iter_kmeans, "I1[1,)")
  checkmate::qassert(max_iter_harmony, "I1[1,)")
  checkmate::qassert(epsilon_kmeans, "N1(0,)")
  checkmate::qassert(epsilon_harmony, "N1(0,)")
  checkmate::qassert(window_size, "I1[1,)")
  checkmate::qassert(alpha, "N1(0,1)")
  checkmate::qassert(tau, "N1[0,)")
  checkmate::qassert(batch_proportion_cutoff, "N1(0,)")
  checkmate::qassert(use_dynamic_lambda, "B1")
  checkmate::qassert(csr_cube_count, "I1[1,)")
  checkmate::qassert(k_means_iter, "I1[1,)")
  if (!is.null(k_means_init)) {
    checkmate::qassert(k_means_init, "S1")
  }
  checkmate::qassert(fixed, "B1")
  checkmate::qassert(quantise, "B1")

  list(
    k = k,
    sigma = sigma,
    theta = theta,
    lambda = lambda,
    max_iter_kmeans = max_iter_kmeans,
    max_iter_harmony = max_iter_harmony,
    epsilon_kmeans = epsilon_kmeans,
    epsilon_harmony = epsilon_harmony,
    window_size = window_size,
    alpha = alpha,
    tau = tau,
    batch_proportion_cutoff = batch_proportion_cutoff,
    use_dynamic_lambda = use_dynamic_lambda,
    csr_cube_count = csr_cube_count,
    k_means_iter = k_means_iter,
    k_means_init = k_means_init,
    fixed = fixed,
    quantise = quantise
  )
}
