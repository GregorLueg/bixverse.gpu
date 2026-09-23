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
    k_means_iter = p_int(
      50L,
      range = "[1,)",
      doc = "Number of k-means iterations."
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
        "Final node degree of the CAGRA navigational graph. If",
        "`NULL`, defaults to `30` on the Rust side."
      )
    ),
    k_build = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "Number of k-neighbours during the NNDescent build phase",
        "before CAGRA pruning. If `NULL`, defaults to `1.5 *",
        "node_degree_final` on the Rust side. (Cannot be smaller than",
        "`node_degree_final`)"
      )
    ),
    n_tree = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Number of trees for graph build. Automatically if",
        "`NULL`."
      )
    ),
    delta = p_dbl(
      0.001,
      doc = "CAGRA GPU: Early termination parameter for NN descent."
    ),
    rho = p_dbl(
      NULL,
      null_ok = TRUE,
      doc = "CAGRA GPU: Sample rate parameter for NN descent."
    ),
    beam_width = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Beam width for beam search. If not provided will",
        "be set to `max(c(k, node_degree_final, 16L)) * 2`."
      )
    ),
    max_beam_iters = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Maximum number of beam search iterations. If not",
        "provided, defaults to `3 * beam_width`."
      )
    ),
    n_entry_points = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "CAGRA GPU: Number of entry points for beam search. If not",
        "provided, defaults to `8L`."
      )
    ),
    extract_knn = p_lgl(
      FALSE,
      doc = paste(
        "CAGRA GPU: Skip the beam search and take the graph the",
        "NNDescent left it. Faster, slightly lower recall. Ignored by",
        "the other two searches."
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
      c("adam_gpu", "adam_parallel", "sgd", "adam"),
      doc = "The optimiser for the embedding."
    ),
    init = p_choice(
      "spectral",
      c("spectral", "pca", "random"),
      doc = "Embedding initialisation method."
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
        "acceleration."
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
      FALSE,
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
    k_means_iter = p_int(
      30L,
      range = "[1,)",
      doc = paste(
        "Maximum number of k-means iterations for the initial",
        "centroid computation."
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
  title = "Default parameters for GPU fast Louvain clustering",
  description = paste(
    "GPU counterpart to [bixverse::params_sc_fast_cluster()]. The",
    "mini-batch k-means knobs are gone (the GPU k-means is",
    "full-batch Lloyd's) and the k-means block comes from the GPU",
    "parameters instead. Two knobs the CPU wrapper never exposed,",
    "`same_weight` and `multi_level_louvain`, are available here.",
    "The k-means distance is taken from `knn$ann_dist`, so the",
    "coarsening and the centroid graph agree on the geometry.",
    "There is no separate `metric` argument, and `\"manhattan\"`",
    "is not supported by the GPU k-means."
  ),
  checker = "ScFastClusterGpu",
  label = "GPU fast clustering params",
  hint = "See ?params_sc_fast_cluster_gpu and ?bixverse::params_knn_defaults.",
  extra_ctor = quote(
    if (!is.null(k_means_init)) {
      checkmate::assertChoice(k_means_init, c("random", "parallel", "plusplus"))
    }
  ),
  # The kNN block comes from bixverse, so its rules are spelled out here.
  extra_check = quote({
    res <- check_list_shape(x, c("k", "knn_method", "ann_dist"))
    if (!isTRUE(res)) {
      return(res)
    }
    res <- apply_qtest_rules(
      x,
      list(
        k = "I1[1,)",
        n_trees = "I1[1,)",
        m = "I1[1,)",
        ef_construction = "I1[1,)",
        ef_search = "I1[1,)",
        search_budget = c("I1[1,)", "0"),
        ef_budget = c("I1[1,)", "0"),
        n_list = c("I1[1,)", "0"),
        n_probe = c("I1[1,)", "0"),
        delta = "N1[0,1]",
        diversify_prob = "N1[0,1]"
      ),
      label = "GPU fast clustering params"
    )
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
    # the GPU k-means takes its metric from here, and rejects "manhattan"
    res <- checkmate::checkChoice(x[["ann_dist"]], c("euclidean", "cosine"))
    if (!isTRUE(res)) {
      return(res)
    }
  }),
  fields = list(
    k_means_iter = p_int(
      50L,
      range = "[1,)",
      doc = "Maximum number of k-means iterations."
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
        "checking for convergence."
      )
    ),
    quantise = p_lgl(
      FALSE,
      doc = paste(
        "Shall the data buffer be held at fp16 on the GPU. Halves the",
        "buffer and helps when the assignment kernels are memory",
        "bound."
      )
    ),
    same_weight = p_lgl(
      FALSE,
      doc = paste(
        "If `TRUE`, all kNN edges get weight `1.0`. Otherwise edges",
        "with a reverse counterpart are double counted."
      )
    ),
    full_snn = p_lgl(
      FALSE,
      doc = paste(
        "Shall the full shared nearest neighbour graph be generated,",
        "including edges between centroids that are not neighbours."
      )
    ),
    pruning = p_dbl(
      NULL,
      range = "[0,1]",
      null_ok = TRUE,
      doc = paste(
        "Weights below this threshold are set to 0 when generating",
        "the sNN graph. If `NULL`, defaults to `1 / ceiling(k *",
        "0.8)`."
      )
    ),
    snn_similarity = p_choice(
      "jaccard",
      c("jaccard", "rank"),
      doc = paste(
        "Jaccard computes the Jaccard index between neighbour sets;",
        "rank weights edges by the best combined rank of a shared",
        "neighbour. Both are normalised to `[0, 1]`."
      )
    ),
    louvain_iters = p_int(
      10L,
      range = "[1,)",
      doc = "Number of Louvain iterations."
    ),
    multi_level_louvain = p_lgl(
      TRUE,
      doc = "Shall multi-level Louvain be applied."
    ),
    knn = p_merge(
      quote(bixverse::params_knn_defaults()),
      default = list(k = 5L),
      doc = paste(
        "Optional overrides for the kNN parameters applied to the",
        "centroids. See [bixverse::params_knn_defaults()] for the",
        "available parameters."
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
    "NN-descent builds a CAGRA graph and, with `extract_knn = TRUE`,",
    "hands that graph back rather than beam searching over it. Note that",
    "this saves the query, not the build: the descent itself dominates, and",
    "its build degree tracks `k`. NN-descent is therefore a low-`k` tool on",
    "the GPU. Above `k` of roughly 30 both exhaustive and IVF beat it, and",
    "by `k = 200` they beat it by more than an order of magnitude."
  ),
  checker = NULL,
  fields = list(
    k = p_int(
      15L,
      doc = paste(
        "Number of neighbours. `0L` hands the choice to Rust, which uses",
        "`sqrt(n_cells) * 0.5` and then adjusts for the simulated doublets."
      )
    ),
    knn_method = p_choice(
      "exhaustive",
      c("exhaustive", "ivf", "nndescent"),
      doc = "The GPU index to use."
    ),
    ann_dist = p_choice(
      "euclidean",
      c("euclidean", "cosine"),
      doc = "Manhattan is not supported by the GPU kernels."
    ),
    n_list = p_int(
      NULL,
      null_ok = TRUE,
      doc = "IVF only. Number of clusters. `NULL` gives `sqrt(n)`."
    ),
    n_probe = p_int(
      NULL,
      null_ok = TRUE,
      doc = "IVF only. Clusters to probe. `NULL` gives `sqrt(n_list)`."
    ),
    graph_k = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NN-descent only. Node degree of the graph after pruning. `NULL`",
        "gives 30, widened to cover `k` when `extract_knn` is set."
      )
    ),
    k_build = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NN-descent only. Build degree before pruning. `NULL` gives",
        "`max(k, floor(1.5 * k))`."
      )
    ),
    n_tree = p_int(
      NULL,
      null_ok = TRUE,
      doc = "NN-descent only. Trees seeding the descent."
    ),
    delta = p_dbl(
      0.001,
      doc = "NN-descent only. Termination criterium for the descent."
    ),
    rho = p_dbl(
      NULL,
      null_ok = TRUE,
      doc = "NN-descent only. Sampling rate for the descent."
    ),
    refine_knn = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NN-descent only. 2-hop refinement sweeps after the descent. Buys",
        "graph quality at a linear cost. `NULL` gives 0."
      )
    ),
    beam_width = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NN-descent only. Beam width when querying. Ignored when",
        "`extract_knn` is set."
      )
    ),
    max_beam_iters = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NN-descent only. Beam search iterations. Ignored when",
        "`extract_knn` is set."
      )
    ),
    n_entry_points = p_int(
      NULL,
      null_ok = TRUE,
      doc = paste(
        "NN-descent only. Entry points when querying. Ignored when",
        "`extract_knn` is set."
      )
    ),
    extract_knn = p_lgl(
      FALSE,
      doc = paste(
        "NN-descent only. Hand back the graph the descent built instead of",
        "beam searching over it."
      )
    )
  )
)
