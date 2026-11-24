spec_parametric_umap <- param_spec(
  name = "parametric_umap",
  title = "Wrapper function to generate parametric UMAP parameters",
  checker = "ParametricUmap",
  label = "parametric UMAP params",
  hint = paste(
    "local_connectivity/bandwidth/mix_weight/corr_weight must be numeric,",
    "lr must be a positive numeric, hidden_layers must be a positive",
    "integer vector, and n_epochs/batch_size/neg_sample_rate must be",
    "positive integers."
  ),
  extra_ctor = quote(hidden_layers <- as.integer(hidden_layers)),
  fields = list(
    local_connectivity = p_dbl(
      1,
      doc = "Number of nearest neighbours assumed to be at distance zero."
    ),
    bandwidth = p_dbl(
      1e-05,
      doc = "Convergence tolerance for smooth kNN distance binary search."
    ),
    mix_weight = p_dbl(
      1,
      doc = paste(
        "Balance between fuzzy union and directed graph during",
        "symmetrisation."
      )
    ),
    hidden_layers = p_int(
      c(128L, 64L, 32L),
      range = "[1,)",
      len = "+",
      integerish = TRUE,
      doc = "Hidden layer sizes for the MLP encoder."
    ),
    lr = p_dbl(
      0.001,
      range = "(0,)",
      doc = "Learning rate for the neural network optimiser."
    ),
    corr_weight = p_dbl(
      0,
      doc = paste(
        "Coefficient for the negative Pearson correlation loss that",
        "encourages similar distances in embedding and original",
        "space."
      )
    ),
    n_epochs = p_int(500L, range = "[1,)", doc = "Number of training epochs."),
    batch_size = p_int(256L, range = "[1,)", doc = "Training batch size."),
    neg_sample_rate = p_int(
      5L,
      range = "[1,)",
      doc = "Number of negative samples per positive edge."
    )
  )
)

spec_kmeans_gpu <- param_spec(
  name = "kmeans_gpu",
  title = "Default parameters for GPU k-means",
  checker = "KMeansGpu",
  label = "GPU k-means params",
  extra_ctor = quote(
    if (!is.null(k_means_init)) {
      checkmate::assertChoice(k_means_init, c("random", "parallel", "plusplus"))
    }
  ),
  extra_check = quote(
    if (
      !is.null(x[["k_means_init"]]) &&
        !checkmate::testChoice(
          x[["k_means_init"]],
          c("random", "parallel", "plusplus")
        )
    ) {
      return(paste(
        "Element `k_means_init` must be NULL or one of",
        "'random', 'parallel', or 'plusplus'."
      ))
    }
  ),
  fields = list(
    k_means_algorithm = p_choice(
      "emeans",
      c("emeans", "lloyd"),
      doc = paste(
        "The k-means algorithm. `\"emeans\"` (Bessa et al., 2026) runs",
        "mini-batch EMA updates and ignores `k_means_iter`, `k_means_init`",
        "and `fixed`. `\"lloyd\"` is Lloyd's algorithm."
      )
    ),
    e_means_epochs = p_dbl(
      NULL,
      range = "(0,)",
      null_ok = TRUE,
      doc = paste(
        "e-means only. Passes over the data. `NULL` takes the Rust default",
        "of 5."
      )
    ),
    k_means_iter = p_int(
      50L,
      range = "[1,)",
      doc = "Number of k-means iterations. Lloyd's only."
    ),
    k_means_init = p_chr(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "Initialisation method. One of `\"random\"`, `\"parallel\"`,",
        "or `\"plusplus\"`. If `NULL`, determined on the Rust side."
      )
    ),
    metric = p_choice(
      "euclidean",
      c("euclidean", "cosine"),
      doc = "The distance metric."
    ),
    fixed = p_lgl(
      FALSE,
      doc = paste(
        "Shall the algorithm be run for a fixed number of iterations,",
        "without checking for convergence."
      )
    ),
    quantise = p_lgl(
      FALSE,
      doc = paste(
        "Whether to quantise data to `fp16` before clustering. This",
        "can improve performance in circumstances where it is memory",
        "bound."
      )
    )
  )
)

spec_nn_gpu <- param_spec(
  name = "nn_gpu",
  title = paste(
    "Wrapper function to generate GPU nearest neighbour",
    "parameters"
  ),
  checker = "NnGpu",
  label = "GPU nearest neighbour params",
  fields = list(
    dist_metric = p_choice(
      "euclidean",
      c("euclidean", "cosine"),
      doc = "The distance metric to use."
    ),
    n_list = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "IVF GPU: Number of clusters to use. If `NULL`, will default",
        "to `sqrt(n)`."
      )
    ),
    n_probes = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "IVF GPU: Number of clusters to probe. If `NULL`, will",
        "default to `sqrt(n_list)`."
      )
    ),
    node_degree_final = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Final node degree of the navigational graph. If",
        "`NULL`, defaults to `30` on the Rust side. Ignored by UMAP, tSNE",
        "and PHATE."
      )
    ),
    k_build = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NNDescent and CAGRA GPU: Working degree of the descent. If",
        "`NULL`, defaults to `1.5 * k` on the Rust side. For CAGRA it",
        "cannot be smaller than `node_degree_final`."
      )
    ),
    n_tree = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NNDescent and CAGRA GPU: Number of trees seeding the descent.",
        "Automatically if `NULL`."
      )
    ),
    delta = p_dbl(
      0.001,
      doc = paste(
        "NNDescent and CAGRA GPU: Early termination parameter for the",
        "descent."
      )
    ),
    rho = p_dbl(
      NULL,
      null_ok = TRUE,
      doc = "NNDescent and CAGRA GPU: Sample rate for the descent."
    ),
    beam_width = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Beam width for beam search. If not provided will",
        "be set to `max(c(k, node_degree_final, 16L)) * 2`. Ignored by",
        "UMAP, tSNE and PHATE."
      )
    ),
    max_beam_iters = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Maximum number of beam search iterations. If not",
        "provided, defaults to `3 * beam_width`. Ignored by UMAP, tSNE",
        "and PHATE."
      )
    ),
    n_entry_points = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Number of entry points for beam search. If not",
        "provided, defaults to `8L`. Ignored by UMAP, tSNE and PHATE."
      )
    )
  )
)

spec_umap_gpu <- param_spec(
  name = "umap_gpu",
  title = "Wrapper function to generate UMAP parameters (GPU version)",
  checker = "UmapGpu",
  label = "GPU UMAP params",
  fields = list(
    local_connectivity = p_dbl(
      1,
      doc = "Number of nearest neighbours assumed to be at distance zero."
    ),
    bandwidth = p_dbl(
      1e-05,
      doc = "Convergence tolerance for smooth kNN distance binary search."
    ),
    mix_weight = p_dbl(
      1,
      doc = paste(
        "Balance between fuzzy union and directed graph during",
        "symmetrisation."
      )
    ),
    lr = p_dbl(1, doc = "Learning rate."),
    n_epochs = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = "Number of optimisation epochs."
    ),
    neg_sample_rate = p_int(
      5L,
      doc = "Number of negative samples per positive sample."
    ),
    gamma = p_dbl(1, doc = "Repulsion strength."),
    optimiser = p_choice(
      "adam_gpu",
      c(
        "adam_gpu",
        "adam_v06_gpu",
        "adam_parallel",
        "adam_v06",
        "sgd",
        "sgd_parallel",
        "adam"
      ),
      doc = paste(
        "The optimiser for the embedding. `\"sgd_parallel\"` is lock-free",
        "(Hogwild!) and not reproducible run to run with more than one thread.",
        "`\"adam_v06_gpu\"` (GPU) and `\"adam_v06\"` (CPU) are the Adam of",
        "umap-learn 0.6, meant to pair with `init = \"recursive\"`."
      )
    ),
    init = p_choice(
      "spectral",
      c("spectral", "pca", "random", "recursive"),
      doc = paste(
        "Embedding initialisation method. `\"recursive\"` is the recursive",
        "label propagation initialisation of umap-learn 0.6."
      )
    ),
    randomised = p_lgl(
      FALSE,
      doc = "Use randomised SVD for PCA initialisation."
    )
  )
)

spec_tsne_gpu <- param_spec(
  name = "tsne_gpu",
  title = "Wrapper function to generate t-SNE parameters (GPU version)",
  references = "Belkina, et al., Nat. Commun., 2019",
  checker = "TsneGpu",
  label = "GPU t-SNE params",
  fields = list(
    lr = p_dbl(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "Learning rate. If `NULL` (the default), the Rust backend",
        "sets it to `max((n_samples / 12), 200)`, following the",
        "N-dependent heuristic of Belkina et al. (2019)."
      )
    ),
    n_epochs = p_int(
      1000L,
      range = "[1,)",
      doc = "Number of optimisation epochs."
    ),
    early_exag_iter = p_int(
      250L,
      range = "[1,)",
      doc = "Number of early exaggeration iterations."
    ),
    early_exag_factor = p_dbl(12, doc = "Early exaggeration factor."),
    late_exag_factor = p_dbl(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "If you wish to also use late exaggerations. Can be useful on",
        "large data sets (set it to `2.0` to `4.0`)."
      )
    ),
    theta = p_dbl(
      0.5,
      range = "[0,1]",
      doc = paste(
        "Barnes-Hut approximation angle. Lower values increase",
        "accuracy at the cost of speed."
      )
    ),
    n_interp_points = p_int(
      3L,
      range = "[1,)",
      doc = paste(
        "Number of interpolation points per grid cell for FFT",
        "acceleration (`\"fft\"`, `\"fft_3k\"` and `\"fft_3k_gpu\"`)."
      )
    ),
    max_depth = p_int(
      7L,
      range = "[1,)",
      doc = paste(
        "Maximum Barnes-Hut tree depth for the quick-and-dirty optimiser",
        "(`\"bh_qd\"`). Smaller is faster and coarser; qdtsne recommends `7` to",
        "`10`. No effect on the other optimisers."
      )
    ),
    init = p_choice(
      "pca",
      c("pca", "spectral", "random"),
      doc = "Embedding initialisation method."
    ),
    randomised = p_lgl(TRUE, doc = "Use randomised SVD for PCA initialisation.")
  )
)

spec_phate_gpu <- param_spec(
  name = "phate_gpu",
  title = "Wrapper function to generate PHATE parameters (GPU version)",
  references = "Moon, et al., Nat. Biotechnol., 2019",
  checker = "PhateGpu",
  label = "GPU PHATE params",
  fields = list(
    decay = p_dbl(
      40,
      range = "(0,)",
      null_ok = TRUE,
      doc = paste(
        "Alpha decay exponent of the kernel. Higher values give sharper",
        "transitions. `NULL` gives a binary connectivity kernel."
      )
    ),
    bandwidth_scale = p_dbl(
      1,
      range = "(0,)",
      doc = "Multiplicative factor on the kernel bandwidth."
    ),
    thresh = p_dbl(
      1e-4,
      range = "[0,)",
      doc = paste(
        "Affinities below this are set to zero. Neighbours past `k` join the",
        "kernel while their affinity clears it."
      )
    ),
    graph_symmetry = p_choice(
      "additive",
      c("additive", "multiplicative", "mnn", "none"),
      doc = "How to symmetrise the affinity graph."
    ),
    n_landmarks = p_int(
      2048L,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "Number of landmarks for the compressed diffusion operator. `NULL`,",
        "or a value of at least the number of samples, runs the full N x N",
        "operator, which gets painful fast."
      )
    ),
    landmark_method = p_choice(
      "spectral",
      c("spectral", "random", "density"),
      doc = paste(
        "How to select the landmarks. `\"spectral\"` runs its randomised SVD",
        "and k-means on the GPU (or on MLX with `backend = \"mlx\"`)."
      )
    ),
    n_svd = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "Singular vectors for spectral landmark selection. If `NULL`,",
        "defaults to `100` on the Rust side."
      )
    ),
    t_max = p_int(
      100L,
      range = "[1,)",
      doc = paste(
        "Largest diffusion time the Von Neumann entropy knee search",
        "considers."
      )
    ),
    t_custom = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = "Fixed diffusion time. If `NULL`, picks the knee automatically."
    ),
    gamma = p_dbl(
      1,
      range = "[-1,1]",
      doc = "Informational distance constant. `1` is the log potential."
    ),
    mds_method = p_choice(
      "sgd_dense",
      c("sgd_dense", "classic"),
      doc = "MDS algorithm for the final embedding."
    ),
    mds_iter = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "Number of SGD-MDS iterations. If `NULL`, the Rust side runs 300",
        "below 1000 points, 500 below 5000 and 800 otherwise (points being",
        "the landmarks when landmarks are used)."
      )
    ),
    randomised = p_lgl(
      TRUE,
      doc = paste(
        "Use randomised SVD for classic MDS (`mds_method = \"classic\"`).",
        "SGD-MDS initialises from a randomised classic MDS either way."
      )
    )
  )
)

spec_sc_harmony_v2_gpu <- param_spec(
  name = "sc_harmony_v2_gpu",
  title = "Default parameters for Harmony v2 GPU batch correction",
  checker = "ScHarmonyV2Gpu",
  label = "Harmony v2 GPU params",
  fields = list(
    k = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "Number of clusters for k-means clustering. If not provided,",
        "it will be automatically determined as `min(round(N / 30),",
        "100)`."
      )
    ),
    sigma = p_dbl(
      0.1,
      range = "[0,)",
      len = "+",
      doc = paste(
        "Per-cluster diversity weights. Either a single value",
        "(broadcast to all clusters) or a vector of length k."
      )
    ),
    theta = p_dbl(
      2,
      range = "[0,)",
      len = "+",
      doc = paste(
        "Per-variable diversity penalty. Must be a single value; only",
        "one batch covariate is supported on the GPU path."
      )
    ),
    lambda = p_dbl(
      1,
      range = "[0,)",
      len = "+",
      doc = paste(
        "Ridge regression penalty for the linear model. Typically a",
        "single value. Ignored when `use_dynamic_lambda = TRUE`."
      )
    ),
    max_iter_kmeans = p_int(
      4L,
      range = "[1,)",
      doc = "Maximum number of k-means Jacobi sweeps per Harmony round."
    ),
    max_iter_harmony = p_int(
      10L,
      range = "[1,)",
      doc = "Maximum number of Harmony outer iterations."
    ),
    epsilon_kmeans = p_dbl(
      0.001,
      range = "(0,)",
      doc = "Convergence threshold for k-means clustering."
    ),
    epsilon_harmony = p_dbl(
      0.01,
      range = "(0,)",
      doc = "Convergence threshold for Harmony."
    ),
    window_size = p_int(
      3L,
      range = "[1,)",
      doc = paste(
        "Number of previous iterations to consider when checking",
        "convergence."
      )
    ),
    alpha = p_dbl(
      0.2,
      range = "(0,1)",
      doc = paste(
        "Scaling factor for dynamic lambda estimation. Must be in (0,",
        "1). Only relevant when `use_dynamic_lambda = TRUE`."
      )
    ),
    tau = p_dbl(
      0,
      range = "[0,)",
      doc = paste(
        "Scaling factor for theta based on batch size. A value of 0",
        "disables batch-size scaling of theta."
      )
    ),
    batch_proportion_cutoff = p_dbl(
      1e-05,
      range = "(0,)",
      doc = paste(
        "Cutoff for pruning batches with small proportions during",
        "ridge regression."
      )
    ),
    use_dynamic_lambda = p_lgl(
      TRUE,
      doc = paste(
        "If `TRUE`, lambda is estimated dynamically per cluster",
        "instead of using the fixed `lambda` value."
      )
    ),
    csr_cube_count = p_int(
      256L,
      range = "[1,)",
      doc = paste(
        "Number of parallel thread groups used when building the",
        "level-CSR index on the GPU. Adjust for your hardware if",
        "needed."
      )
    ),
    k_means_algorithm = p_choice(
      "emeans",
      c("emeans", "lloyd"),
      doc = paste(
        "The k-means algorithm. `\"emeans\"` (Bessa et al., 2026) runs",
        "mini-batch EMA updates and ignores `k_means_iter`, `k_means_init`",
        "and `fixed`. `\"lloyd\"` is Lloyd's algorithm."
      )
    ),
    e_means_epochs = p_dbl(
      NULL,
      range = "(0,)",
      null_ok = TRUE,
      doc = paste(
        "e-means only. Passes over the data. `NULL` takes the Rust default",
        "of 5."
      )
    ),
    k_means_iter = p_int(
      10L,
      range = "[1,)",
      doc = paste(
        "Maximum number of k-means iterations for the initial",
        "centroid computation. Lloyd's only."
      )
    ),
    k_means_init = p_chr(
      NULL,
      null_ok = TRUE,
      doc = "Initialisation strategy for k-means."
    ),
    fixed = p_lgl(
      FALSE,
      doc = "If `TRUE`, centroids are fixed after initialisation."
    ),
    quantise = p_lgl(
      FALSE,
      doc = paste(
        "If `TRUE`, quantises intermediate values to f16 during",
        "k-means."
      )
    )
  )
)

spec_sc_fast_cluster_gpu <- param_spec(
  name = "sc_fast_cluster_gpu",
  title = "Default parameters for GPU fast clustering",
  description = paste(
    "GPU counterpart to [bixverse::params_sc_fast_cluster()]. Fast",
    "clustering coarsens the cells with k-means, builds a graph over the",
    "centroids and runs Louvain or Leiden on it. The community of each",
    "centroid is handed back to its cells. Only the k-means (and the",
    "cell-mediated top-m search) run on the GPU. The mini-batch k-means",
    "knobs are gone: the GPU k-means is e-means or full-batch Lloyd's."
  ),
  references = "Wei, et al., PLoS Comput Biol, 2022.",
  checker = "ScFastClusterGpu",
  label = "GPU fast clustering params",
  hint = paste(
    "See ?params_sc_fast_cluster_gpu and",
    "?bixverse::params_knn_graph_defaults."
  ),
  extra_ctor = quote({
    if (!is.null(k_means_init)) {
      checkmate::assertChoice(k_means_init, c("random", "parallel", "plusplus"))
    }
    if (!is.null(local_moving)) {
      checkmate::assertChoice(local_moving, c("queue", "coloured"))
    }
    if (!is.null(node_order)) {
      checkmate::assertChoice(node_order, c("bfs", "shuffle"))
    }
    bixverse:::assertKnnGraphParams(centroid_graph)
  }),
  # The graph block comes from bixverse, so its rules are checked there.
  extra_check = quote({
    res <- bixverse:::checkKnnGraphParams(x)
    if (!isTRUE(res)) {
      return(res)
    }
    if (
      !is.null(x[["k_means_init"]]) &&
        !checkmate::testChoice(
          x[["k_means_init"]],
          c("random", "parallel", "plusplus")
        )
    ) {
      return(paste(
        "Element `k_means_init` must be NULL or one of 'random',",
        "'parallel' or 'plusplus'."
      ))
    }
    if (
      !is.null(x$local_moving) && !x$local_moving %in% c("queue", "coloured")
    ) {
      return("The element `local_moving` must be one of `queue` or `coloured`.")
    }
    if (!is.null(x$node_order) && !x$node_order %in% c("bfs", "shuffle")) {
      return("The element `node_order` must be one of `bfs` or `shuffle`.")
    }
  }),
  fields = list(
    n_centroids = p_int(
      NULL,
      range = "[2,)",
      null_ok = TRUE,
      doc = paste(
        "Number of k-means centroids. If `NULL`, defaults to",
        "`floor(4 * sqrt(n_cells))`."
      )
    ),
    k_means_algorithm = p_choice(
      "emeans",
      c("emeans", "lloyd"),
      doc = paste(
        "The k-means algorithm. `\"emeans\"` (Bessa et al., 2026) runs",
        "mini-batch EMA updates and ignores `k_means_iter`, `k_means_init`",
        "and `fixed`. `\"lloyd\"` is Lloyd's algorithm."
      )
    ),
    e_means_epochs = p_dbl(
      NULL,
      range = "(0,)",
      null_ok = TRUE,
      doc = paste(
        "e-means only. Passes over the data. `NULL` takes the Rust default",
        "of 5."
      )
    ),
    k_means_iter = p_int(
      50L,
      range = "[1,)",
      doc = "Maximum number of k-means iterations. Lloyd's only."
    ),
    k_means_init = p_chr(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "Initialisation method. One of `\"random\"`, `\"parallel\"`",
        "or `\"plusplus\"`. If `NULL`, picked on the Rust side based",
        "on the number of centroids."
      )
    ),
    fixed = p_lgl(
      TRUE,
      doc = paste(
        "Shall k-means run for a fixed number of iterations, without",
        "checking for convergence. wgpu only; MLX always runs every",
        "iteration."
      )
    ),
    quantise = p_lgl(
      FALSE,
      doc = paste(
        "Shall the data buffer be held at fp16 on the GPU. Halves the",
        "buffer and helps when the assignment kernels are memory",
        "bound. wgpu only."
      )
    ),
    dist = p_choice(
      "euclidean",
      c("euclidean", "cosine"),
      doc = "Distance for the k-means and the centroid graph."
    ),
    cell_mediated = p_lgl(
      FALSE,
      doc = paste(
        "Shall the cell-mediated centroid graph from Secuer be built",
        "instead of the kNN graph on the centroids. Centroids are then",
        "linked by the cells that sit near both, so two centroids that are",
        "close in space but have no cells between them stay apart. If",
        "`TRUE`, `centroid_k` and `centroid_graph` are ignored."
      )
    ),
    n_anchors = p_int(
      5L,
      range = "[1,)",
      doc = paste(
        "Number of nearest centroids each cell links to in the",
        "cell-mediated graph. Only used if `cell_mediated = TRUE`."
      )
    ),
    centroid_k = p_int(
      15L,
      range = "[1,)",
      doc = paste(
        "Neighbours per centroid in the exact centroid kNN. Only used if",
        "`cell_mediated = FALSE`."
      )
    ),
    centroid_graph = p_merge(
      quote(bixverse::params_knn_graph_defaults()),
      doc = paste(
        "Optional overrides for how the centroid kNN is turned into a",
        "graph. See [bixverse::params_knn_graph_defaults()]. Only used if",
        "`cell_mediated = FALSE`."
      )
    ),
    algorithm = p_choice(
      "louvain",
      c("louvain", "leiden"),
      doc = "Community detection on the centroid graph."
    ),
    n_iterations = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "Number of iterations of the community detection. A negative value",
        "iterates until stable. If `NULL`, the bixverse default for the",
        "chosen `algorithm`."
      )
    ),
    beta = p_dbl(
      0.01,
      range = "(0,)",
      doc = "Leiden only. Randomness of the refinement step."
    ),
    min_gain = p_dbl(
      0,
      range = "[0,)",
      doc = paste(
        "With a negative `n_iterations`, also stop once an iteration gains",
        "no more than this in modularity."
      )
    ),
    local_moving = p_chr(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "`\"queue\"` is Traag's sequential queue. `\"coloured\"` moves the",
        "nodes of one colour class in parallel, which only kicks in on",
        "larger levels of the graph. `NULL` is `\"queue\"` for Leiden and",
        "`\"coloured\"` for Louvain."
      )
    ),
    node_order = p_chr(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "Visit order of the queue, `\"bfs\"` (sweep in BFS order from a",
        "random start) or `\"shuffle\"` (as igraph). `NULL` is `\"bfs\"`",
        "for Leiden and `\"shuffle\"` for Louvain."
      )
    ),
    multi_level = p_lgl(
      TRUE,
      doc = "Shall the communities be aggregated and refined in levels."
    ),
    max_evals = p_int(
      10L,
      range = "[1,)",
      doc = paste(
        "Cap on the node evaluations per local moving phase, in units of",
        "the number of nodes."
      )
    )
  )
)

spec_knn_gpu_defaults <- param_defaults(
  name = "knn_gpu_defaults",
  title = "Default parameters for the GPU nearest neighbour backends",
  description = paste(
    "GPU sibling of [bixverse::params_knn_defaults()]. The GPU indices take",
    "a different knob set: there is no Annoy and no HNSW on the device, so",
    "what survives is exhaustive, IVF and NN-descent."
  ),
  details = paste(
    "`\"nndescent\"` hands back the kNN graph the descent built, with no",
    "index and no search; rows the descent left short are topped up with an",
    "exact search of just those rows. Where the query is a different set of",
    "cells (BBKNN, fastMNN) it builds and beam searches a CAGRA graph",
    "instead. `\"cagra\"` always does the latter, self-queries included.",
    "The descent itself dominates either way, and its build degree tracks",
    "`k`, so both are low-`k` tools on the GPU. Above `k` of roughly 30",
    "exhaustive and IVF beat them."
  ),
  checker = NULL,
  fields = list(
    k = p_int(
      15L,
      range = "[0,)",
      doc = paste(
        "Number of neighbours. `0L` hands the choice to Rust, which uses",
        "`sqrt(n_cells) * 0.5` and then adjusts for the simulated doublets."
      )
    ),
    knn_method = p_choice(
      "exhaustive",
      c("exhaustive", "ivf", "nndescent", "cagra"),
      doc = "The GPU index to use."
    ),
    ann_dist = p_choice(
      "euclidean",
      c("euclidean", "cosine"),
      doc = "Manhattan is not supported by the GPU kernels."
    ),
    n_list = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = "IVF only. Number of clusters. `NULL` gives `sqrt(n)`."
    ),
    n_probe = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = "IVF only. Clusters to probe. `NULL` gives `sqrt(n_list)`."
    ),
    graph_k = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "CAGRA only. Node degree of the graph after pruning. `NULL` gives",
        "30."
      )
    ),
    k_build = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "NN-descent and CAGRA. Working degree of the descent. `NULL` gives",
        "`max(k, floor(1.5 * k))`."
      )
    ),
    n_tree = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = "NN-descent and CAGRA. Trees seeding the descent."
    ),
    delta = p_dbl(
      0.001,
      doc = "NN-descent and CAGRA. Termination criterium for the descent."
    ),
    rho = p_dbl(
      NULL,
      null_ok = TRUE,
      doc = "NN-descent and CAGRA. Sampling rate for the descent."
    ),
    refine_knn = p_int(
      NULL,
      range = "[0,)",
      null_ok = TRUE,
      doc = paste(
        "NN-descent and CAGRA. 2-hop refinement sweeps after the descent. Buys",
        "graph quality at a linear cost. `NULL` gives 0."
      )
    ),
    beam_width = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "CAGRA only, which includes NN-descent on a cross-query. Beam",
        "width when querying."
      )
    ),
    max_beam_iters = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "CAGRA only, which includes NN-descent on a cross-query. Beam",
        "search iterations."
      )
    ),
    n_entry_points = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "CAGRA only, which includes NN-descent on a cross-query. Entry",
        "points when querying."
      )
    )
  )
)

spec_nebula_gpu <- param_spec(
  name = "nebula_gpu",
  title = "Wrapper function for parameters for GPU NEBULA",
  description = paste(
    "GPU counterpart to [bixverse::params_nebula()]. Same knobs and",
    "defaults, minus `reml`: the device fit does not implement it.",
    "Stage two of NEBULA, the per-gene penalised fits, runs in `f32`",
    "on the device and is finished on the host in `f64`, so the",
    "estimates sit close to the CPU ones rather than on them."
  ),
  references = "He, et al., Commun Biol, 2021",
  checker = "NebulaGpu",
  label = "GPU NEBULA params",
  hint = paste(
    "The overdispersion bounds and `eps` must be strictly",
    "positive; `gene_batch_size` must be at least 1."
  ),
  extra_ctor = quote({
    if (min_sigma >= max_sigma) {
      stop("`min_sigma` needs to be below `max_sigma`.")
    }
    if (min_phi >= max_phi) {
      stop("`min_phi` needs to be below `max_phi`.")
    }
  }),
  extra_check = quote({
    if (x[["min_sigma"]] >= x[["max_sigma"]]) {
      return("`min_sigma` in GPU NEBULA params is not below `max_sigma`.")
    }
    if (x[["min_phi"]] >= x[["max_phi"]]) {
      return("`min_phi` in GPU NEBULA params is not below `max_phi`.")
    }
  }),
  fields = list(
    nebula_method = p_choice(
      "ln",
      c("ln", "hl"),
      doc = paste(
        "Which variant to run. NEBULA downgrades `\"ln\"` to `\"hl\"`",
        "below 30 cells per subject, as the R package does."
      )
    ),
    min_sigma = p_dbl(
      1e-04,
      range = "(0,)",
      doc = "Lower bound on the subject-level overdispersion."
    ),
    min_phi = p_dbl(
      1e-04,
      range = "(0,)",
      doc = "Lower bound on the cell-level overdispersion."
    ),
    max_sigma = p_dbl(
      10,
      range = "(0,)",
      doc = "Upper bound on the subject-level overdispersion."
    ),
    max_phi = p_dbl(
      1000,
      range = "(0,)",
      doc = "Upper bound on the cell-level overdispersion."
    ),
    cutoff_cell = p_dbl(
      20,
      range = "[0,)",
      doc = paste(
        "Refit both overdispersions when the product of the cells per",
        "subject and the estimated `phi` falls below this."
      )
    ),
    kappa = p_dbl(
      800,
      range = "[0,)",
      doc = paste(
        "Threshold on NEBULA's `kappa_obs` above which the",
        "subject-level overdispersion from stage one is trusted as",
        "is."
      )
    ),
    cpc = p_dbl(
      0.005,
      range = "[0,)",
      doc = "Drop a gene whose mean count per cell is at most this."
    ),
    mincp = p_int(
      5L,
      range = "[0,)",
      doc = "Drop a gene expressed in fewer than this many cells."
    ),
    min_subjects = p_int(
      0L,
      range = "[0,)",
      doc = paste(
        "Drop a gene that fewer than this many subjects express, a subject",
        "expressing it when its own mean count per cell is above `cpc`.",
        "`cpc` and `mincp` pool every cell, so one subject can carry a gene",
        "through on its own. `0` switches the check off, as in the `nebula`",
        "package."
      )
    ),
    eps = p_dbl(
      1e-06,
      range = "(0,)",
      doc = "Absolute stopping tolerance for the optimiser."
    ),
    gene_batch_size = p_int(
      1000L,
      range = "[1,)",
      doc = paste(
        "Genes read and fitted per batch. Bounds how much of the",
        "store is resident at once and changes nothing about the",
        "answer, since NEBULA is gene-independent."
      )
    ),
    shrink_dispersion = p_lgl(
      TRUE,
      doc = paste(
        "Shrink the cell-level overdispersions towards an empirical",
        "Bayes prior once the sweep is done."
      )
    )
  )
)

spec_sc_bbknn_gpu <- param_spec(
  name = "sc_bbknn_gpu",
  title = "Wrapper function for the GPU BBKNN parameters",
  description = paste(
    "GPU counterpart to [bixverse::params_sc_bbknn()]. Same BBKNN knobs, but",
    "the kNN block is the GPU one, see [params_knn_gpu_defaults()]."
  ),
  details = paste(
    "`k` in the GPU kNN block is rejected rather than silently ignored: it is",
    "set by `neighbours_within_batch`. BBKNN builds one index per batch and",
    "queries each with every cell, so `knn_method = \"nndescent\"` builds and",
    "beam searches a CAGRA graph here."
  ),
  references = "Polański, et al., Bioinformatics, 2020",
  checker = "ScBbknnGpu",
  label = "GPU BBKNN params",
  hint = "See ?params_sc_bbknn_gpu.",
  fields = list(
    neighbours_within_batch = p_int(
      3L,
      range = "[1,)",
      doc = "Number of neighbours to consider per batch."
    ),
    set_op_mix_ratio = p_dbl(
      1,
      range = "[0,1]",
      doc = "Mixing ratio between union (1.0) and intersection (0.0)."
    ),
    local_connectivity = p_dbl(
      1,
      doc = paste(
        "UMAP connectivity computation parameter, how many nearest",
        "neighbours of each cell are assumed to be fully connected."
      )
    ),
    trim = p_int(
      NULL,
      range = "[1,)",
      null_ok = TRUE,
      doc = paste(
        "Trim the neighbours of each cell to these many top connectivities.",
        "May help with population independence and improve the tidiness of",
        "clustering. If `NULL`, it defaults to",
        "`10 * neighbours_within_batch * n_batches`, as in the reference",
        "implementation."
      )
    ),
    knn = p_merge(
      "knn_gpu_defaults",
      drop = "k",
      strict = TRUE,
      doc = "Optional overrides for the kNN block."
    )
  )
)

spec_sc_fastmnn_gpu <- param_spec(
  name = "sc_fastmnn_gpu",
  title = "Wrapper function for the GPU fastMNN parameters",
  description = paste(
    "GPU counterpart to [bixverse::params_sc_fastmnn()]. Same fastMNN knobs",
    "minus the PCA ones, since the GPU path corrects the PCA already stored",
    "in the object. The kNN block is the GPU one, see",
    "[params_knn_gpu_defaults()], with `k` and `ann_dist` defaulting to the",
    "CPU values."
  ),
  details = paste(
    "Every search in fastMNN is a cross-query between two sets of cells, so",
    "`knn_method = \"nndescent\"` builds and beam searches a CAGRA graph",
    "here. `k = 0L` is rejected, since fastMNN has no data-driven fallback",
    "for it."
  ),
  references = "Haghverdi, et al., Nat Biotechnol, 2018",
  checker = "ScFastmnnGpu",
  label = "GPU fastMNN params",
  hint = "See ?params_sc_fastmnn_gpu.",
  extra_check = quote(
    if (x[["k"]] < 1L) {
      return("`k` in GPU fastMNN params must be at least 1.")
    }
  ),
  fields = list(
    ndist = p_dbl(
      3,
      range = "(0,)",
      doc = "Number of median distances for the tricube kernel bandwidth."
    ),
    cos_norm = p_lgl(
      TRUE,
      doc = "Apply cosine normalisation before computing distances."
    ),
    knn = p_merge(
      "knn_gpu_defaults",
      overrides = list(k = 20L, ann_dist = "cosine"),
      strict = TRUE,
      doc = "Optional overrides for the kNN block."
    )
  )
)
