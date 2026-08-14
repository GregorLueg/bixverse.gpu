# param wrappers ---------------------------------------------------------------

## knn -------------------------------------------------------------------------

### cagra ----------------------------------------------------------------------

#' Default parameters for CAGRA-style kNN search
#'
#' @param k Integer. Number of neighbours to identify.
#' @param ann_dist Character. Distance metric to use. One of `"euclidean"` or
#' `"cosine"`.
#' @param node_degree_final Optional integer. Final node degree of the CAGRA
#' navigational graph. If `NULL`, defaults to `30` on the Rust side.
#' @param k_build Optional integer. Number of k-neighbours during the
#' NNDescent build phase before CAGRA pruning. If `NULL`, defaults to
#' `1.5 * node_degree_final` on the Rust side.
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
  k = 15L,
  ann_dist = "euclidean",
  node_degree_final = NULL,
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
  checkmate::qassert(k, "I1[1,)")
  checkmate::qassert(ann_dist, "S1")
  checkmate::assert_choice(ann_dist, c("euclidean", "cosine"))
  checkmate::qassert(node_degree_final, c("I1[1,)", "0"))
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
    k = k,
    ann_dist = ann_dist,
    node_degree_final = node_degree_final,
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
#' @param fixed Logical. Shall the algorithm be run for a fixed number of
#' iterations, without checking for convergence.
#' @param quantise Logical. Whether to quantise data to `fp16` before
#' clustering. This can improve performance in circumstances where it is
#' memory bound.
#'
#' @return A list with the parameters.
#'
#' @export
params_kmeans_gpu <- function(
  k_means_iter = 50L,
  k_means_init = NULL,
  metric = c("euclidean", "cosine"),
  fixed = FALSE,
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
#' @param n_list Optional integer. IVF GPU: Number of clusters to use. If
#' `NULL`, will default to `sqrt(n)`.
#' @param n_probes Optional integer. IVF GPU: Number of clusters to probe. If
#' `NULL`, will default to `sqrt(n_list)`.
#' @param node_degree_final Optional integer. Final node degree of the CAGRA
#' navigational graph. If `NULL`, defaults to `30` on the Rust side.
#' @param k_build Optional integer. Number of k-neighbours during the
#' NNDescent build phase before CAGRA pruning. If `NULL`, defaults to
#' `1.5 * node_degree_final` on the Rust side. (Cannot be smaller than
#' `node_degree_final`)
#' @param n_tree Optional integer. CAGRA GPU: Number of trees for graph build.
#' Automatically if `NULL`.
#' @param delta Float. CAGRA GPU: Early termination parameter for NN descent.
#' Defaults to `0.001`.
#' @param refine_sweeps Integer. Number of refinement sweeps during graph
#' generation.
#' @param rho Optional float. CAGRA GPU: Sample rate parameter for NN descent.
#' Defaults to `0.5` if not provided.
#' @param beam_width Optional integer. CAGRA GPU: Beam width for beam search. If
#' not provided will be set to `max(c(k, node_degree_final, 16L)) * 2`.
#' @param max_beam_iters Optional integer. CAGRA GPU: Maximum number of beam
#' search iterations. If not provided, defaults to `3 * beam_width`.
#' @param n_entry_points Optional integer. CAGRA GPU: Number of entry points for
#' beam search. If not provided, defaults to `8L`.
#'
#' @returns A list with the GPU nearest neighbour parameters.
#'
#' @export
params_nn_gpu <- function(
  dist_metric = c("euclidean", "cosine"),
  n_list = NULL,
  n_probes = NULL,
  node_degree_final = NULL,
  k_build = NULL,
  n_tree = NULL,
  refine_sweeps = 0L,
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
  checkmate::qassert(node_degree_final, c("I1", "0"))
  checkmate::qassert(k_build, c("I1", "0"))
  checkmate::qassert(n_tree, c("I1", "0"))
  checkmate::qassert(refine_sweeps, "I1[0,)")
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
    node_degree_final = node_degree_final,
    k_build = k_build,
    n_tree = n_tree,
    refine_sweeps = refine_sweeps,
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

## tsne gpu -------------------------------------------------------------------

### tsne specific -------------------------------------------------------------

#' Wrapper function to generate t-SNE parameters (GPU version)
#'
#' @param lr Optional numeric. Learning rate. If `NULL` (the default), the Rust
#' backend sets it to `max((n_samples / 12), 200)`, following the N-dependent
#' heuristic of Belkina et al. (2019).
#' @param n_epochs Integer. Number of optimisation epochs. Defaults to `1000L`.
#' @param early_exag_iter Integer. Number of early exaggeration iterations.
#' Defaults to `250L`.
#' @param early_exag_factor Numeric. Early exaggeration factor. Defaults to
#' `12.0`.
#' @param late_exag_factor Optional numeric. If you wish to also use late
#' exaggerations. Can be useful on large data sets (set it to `2.0` to `4.0`).
#' @param theta Numeric. Barnes-Hut approximation angle. Lower values increase
#' accuracy at the cost of speed. Defaults to `0.5`.
#' @param n_interp_points Integer. Number of interpolation points per grid cell
#' for FFT acceleration. Defaults to `3L`.
#' @param init Character. Embedding initialisation method. One of `"pca"`,
#' `"spectral"`, or `"random"`. Defaults to `"pca"`.
#' @param randomised Logical. Use randomised SVD for PCA initialisation.
#' Defaults to `TRUE`.
#'
#' @returns A list with the t-SNE parameters.
#'
#' @references Belkina, et al., Nat. Commun., 2019
#'
#' @export
params_tsne_gpu <- function(
  lr = NULL,
  n_epochs = 1000L,
  early_exag_iter = 250L,
  early_exag_factor = 12.0,
  late_exag_factor = NULL,
  theta = 0.5,
  n_interp_points = 3L,
  init = c("pca", "spectral", "random"),
  randomised = TRUE
) {
  init <- match.arg(init)

  checkmate::qassert(lr, c("N1", "0"))
  checkmate::qassert(n_epochs, "I1[1,)")
  checkmate::qassert(early_exag_iter, "I1[1,)")
  checkmate::qassert(early_exag_factor, "N1")
  checkmate::qassert(late_exag_factor, c("N1", "0"))
  checkmate::qassert(theta, "N1[0,1]")
  checkmate::qassert(n_interp_points, "I1[1,)")
  checkmate::assertChoice(init, c("pca", "spectral", "random"))
  checkmate::qassert(randomised, "B1")

  list(
    lr = lr,
    n_epochs = n_epochs,
    early_exag_iter = early_exag_iter,
    early_exag_factor = early_exag_factor,
    late_exag_factor = late_exag_factor,
    theta = theta,
    n_interp_points = n_interp_points,
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

### fast clustering GPU --------------------------------------------------------

#' Default parameters for GPU fast Louvain clustering
#'
#' @description
#' GPU counterpart to [bixverse::params_sc_fast_cluster()]. The mini-batch
#' k-means knobs are gone (the GPU k-means is full-batch Lloyd's) and the
#' k-means block comes from the GPU parameters instead. Two knobs the CPU
#' wrapper never exposed, `same_weight` and `multi_level_louvain`, are
#' available here.
#'
#' The k-means distance is taken from `knn$ann_dist`, so the coarsening and the
#' centroid graph agree on the geometry. There is no separate `metric`
#' argument, and `"manhattan"` is not supported by the GPU k-means.
#'
#' @param k_means_iter Integer. Maximum number of k-means iterations.
#' @param k_means_init Optional character. Initialisation method. One of
#' `"random"`, `"parallel"` or `"plusplus"`. If `NULL`, picked on the Rust side
#' based on the number of centroids.
#' @param fixed Boolean. Shall k-means run for a fixed number of iterations,
#' without checking for convergence.
#' @param quantise Boolean. Shall the data buffer be held at fp16 on the GPU.
#' Halves the buffer and helps when the assignment kernels are memory bound.
#' @param same_weight Boolean. If `TRUE`, all kNN edges get weight `1.0`.
#' Otherwise edges with a reverse counterpart are double counted.
#' @param full_snn Boolean. Shall the full shared nearest neighbour graph be
#' generated, including edges between centroids that are not neighbours.
#' @param pruning Optional numeric. Weights below this threshold are set to 0
#' when generating the sNN graph. If `NULL`, defaults to `1 / ceiling(k * 0.8)`.
#' @param snn_similarity String. One of `c("jaccard", "rank")`. Jaccard
#' computes the Jaccard index between neighbour sets; rank weights edges by the
#' best combined rank of a shared neighbour. Both are normalised to `[0, 1]`.
#' @param louvain_iters Integer. Number of Louvain iterations.
#' @param multi_level_louvain Boolean. Shall multi-level Louvain be applied.
#' @param knn List. Optional overrides for the kNN parameters applied to the
#' centroids. See [bixverse::params_knn_defaults()] for the available
#' parameters. Defaults to `k = 5L`.
#'
#' @returns A named list with the GPU fast clustering parameters.
#'
#' @export
params_sc_fast_cluster_gpu <- function(
  # kmeans
  k_means_iter = 50L,
  k_means_init = NULL,
  fixed = TRUE,
  quantise = FALSE,
  # knn
  same_weight = FALSE,
  # snn
  full_snn = FALSE,
  pruning = NULL,
  snn_similarity = c("jaccard", "rank"),
  # louvain
  louvain_iters = 10L,
  multi_level_louvain = TRUE,
  # knn params
  knn = list(k = 5L)
) {
  snn_similarity <- match.arg(snn_similarity)

  # checks
  checkmate::qassert(k_means_iter, "I1[1,)")
  if (!is.null(k_means_init)) {
    checkmate::qassert(k_means_init, "S1")
    checkmate::assertChoice(k_means_init, c("random", "parallel", "plusplus"))
  }
  checkmate::qassert(fixed, "B1")
  checkmate::qassert(quantise, "B1")
  checkmate::qassert(same_weight, "B1")
  checkmate::qassert(full_snn, "B1")
  checkmate::qassert(pruning, c("N1[0,1]", "0"))
  checkmate::qassert(louvain_iters, "I1[1,)")
  checkmate::qassert(multi_level_louvain, "B1")
  checkmate::assertChoice(snn_similarity, c("jaccard", "rank"))
  checkmate::assertList(knn)

  knn_params <- utils::modifyList(
    bixverse::params_knn_defaults(),
    knn,
    keep.null = TRUE
  )

  c(
    list(
      k_means_iter = k_means_iter,
      k_means_init = k_means_init,
      fixed = fixed,
      quantise = quantise,
      same_weight = same_weight,
      full_snn = full_snn,
      pruning = pruning,
      snn_similarity = snn_similarity,
      louvain_iters = louvain_iters,
      multi_level_louvain = multi_level_louvain
    ),
    knn_params
  )
}

### scrublet GPU ---------------------------------------------------------------

#' Default parameters for the GPU nearest neighbour backends
#'
#' @description GPU sibling of [bixverse::params_knn_defaults()]. The GPU
#' indices take a much smaller knob set: there is no Annoy, no NN-descent and
#' no HNSW on the device, so only the exhaustive and IVF parameters survive.
#'
#' @returns A named list with the following parameters:
#' \itemize{
#'   \item k - Number of neighbours. `0L` hands the choice to Rust, which uses
#'   `sqrt(n_cells) * 0.5` and then adjusts for the simulated doublets.
#'   \item knn_method - One of `"exhaustive"` or `"ivf"`.
#'   \item ann_dist - One of `"euclidean"` or `"cosine"`. Manhattan is not
#'   supported by the GPU kernels.
#'   \item n_list - IVF only. Number of clusters. `NULL` gives `sqrt(n)`.
#'   \item n_probe - IVF only. Clusters to probe. `NULL` gives `sqrt(n_list)`.
#' }
#'
#' @export
params_knn_gpu_defaults <- function() {
  list(
    k = 15L,
    knn_method = "exhaustive",
    ann_dist = "euclidean",
    n_list = NULL,
    n_probe = NULL
  )
}

#' Wrapper function for GPU Scrublet doublet detection parameters
#'
#' @description GPU counterpart to [bixverse::params_scrublet()]. Two
#' differences from the CPU list. The `pca` sub-list is gone: the GPU SVD is
#' always randomised, so `random_svd` and `sparse` have nothing to switch and
#' `no_pcs` is a plain argument. And the kNN block is backend dependent, see
#' `knn_backend`.
#'
#' @param sim_doublet_ratio Numeric. Number of doublets to simulate relative to
#' the number of observed cells. Defaults to `1.5`.
#' @param expected_doublet_rate Numeric in `[0, 1]`. Expected doublet rate,
#' typically 0.05-0.10 depending on cell loading. Defaults to `0.1`.
#' @param stdev_doublet_rate Numeric in `[0, 1]`. Uncertainty in the expected
#' doublet rate. Defaults to `0.02`.
#' @param n_bins_histogram Integer. Histogram bins for the Otsu threshold
#' search. Defaults to `100L`.
#' @param manual_threshold Optional numeric. Fixed doublet score threshold. If
#' `NULL`, Otsu's method picks it.
#' @param no_pcs Integer. Number of principal components. Defaults to `30L`.
#' @param normalisation List. Optional overrides. See
#' [bixverse::params_norm_doublets_defaults()] for the available parameters:
#' `log_transform`, `mean_center`, `normalise_variance`, `target_size`.
#' @param hvg List. Optional overrides. See [bixverse::params_hvg_defaults()]
#' for the available parameters: `min_gene_var_pctl`, `hvg_method`,
#' `loess_span`, `clip_max`, `n_bins`, `binning_strategy`.
#' @param knn_backend String. One of `"gpu"` or `"cpu"`. Picks which nearest
#' neighbour index runs over the combined observed-plus-simulated embedding,
#' and with it which keys `knn` accepts. `"gpu"` is the fast default; `"cpu"`
#' buys the exact CPU indices at the cost of a host round trip on a matrix
#' that is `(1 + sim_doublet_ratio) * n_cells` rows tall.
#' @param knn List. Optional overrides for the kNN block. Validated against
#' [params_knn_gpu_defaults()] when `knn_backend = "gpu"` and against
#' [bixverse::params_knn_defaults()] when `knn_backend = "cpu"`. Unknown keys
#' are an error, not a silent pass-through. Defaults to `list(k = 0L)`, which
#' asks Rust to pick `k`.
#'
#' @returns A flat named list with all GPU Scrublet parameters.
#'
#' @export
#'
#' @references Wolock, et al., Cell Syst, 2020
params_scrublet_gpu <- function(
  sim_doublet_ratio = 1.5,
  expected_doublet_rate = 0.1,
  stdev_doublet_rate = 0.02,
  n_bins_histogram = 100L,
  manual_threshold = NULL,
  no_pcs = 30L,
  normalisation = list(),
  hvg = list(),
  knn_backend = c("gpu", "cpu"),
  knn = list(k = 0L)
) {
  knn_backend <- match.arg(knn_backend)

  # checks
  checkmate::qassert(sim_doublet_ratio, "N1(0,)")
  checkmate::qassert(expected_doublet_rate, "N1[0,1]")
  checkmate::qassert(stdev_doublet_rate, "N1[0,1]")
  checkmate::qassert(n_bins_histogram, "I1[10,)")
  checkmate::qassert(manual_threshold, c("N1[0,)", "0"))
  checkmate::qassert(no_pcs, "I1[1,)")
  checkmate::assertChoice(knn_backend, c("gpu", "cpu"))
  checkmate::assertList(normalisation)
  checkmate::assertList(hvg)
  checkmate::assertList(knn)

  knn_defaults <- if (knn_backend == "gpu") {
    params_knn_gpu_defaults()
  } else {
    bixverse::params_knn_defaults()
  }

  # the two backends share five key names, so a CPU-only knob silently doing
  # nothing on the GPU arm is the easy mistake here. Catch it where the user
  # types rather than three layers down in Rust.
  unknown_knn <- setdiff(names(knn), names(knn_defaults))
  if (length(unknown_knn) > 0L) {
    stop(sprintf(
      "Unknown kNN parameter(s) for backend '%s': %s. Allowed: %s.",
      knn_backend,
      paste(unknown_knn, collapse = ", "),
      paste(names(knn_defaults), collapse = ", ")
    ))
  }

  params <- list(
    knn_backend = knn_backend,
    normalisation = utils::modifyList(
      bixverse::params_norm_doublets_defaults(),
      normalisation,
      keep.null = TRUE
    ),
    hvg = utils::modifyList(
      bixverse::params_hvg_defaults(),
      hvg,
      keep.null = TRUE
    ),
    no_pcs = no_pcs,
    sim_doublet_ratio = sim_doublet_ratio,
    expected_doublet_rate = expected_doublet_rate,
    stdev_doublet_rate = stdev_doublet_rate,
    n_bins_hist = n_bins_histogram,
    manual_threshold = manual_threshold,
    knn = utils::modifyList(knn_defaults, knn, keep.null = TRUE)
  )

  purrr::list_flatten(params, name_spec = "{inner}")
}
