# checkmate extensions ---------------------------------------------------------

## parametric umap -------------------------------------------------------------

#' Check parametric UMAP parameters
#'
#' @description Checkmate extension for checking the parametric UMAP
#' parameters.
#'
#' @param x The list to check.
#'
#' @return `TRUE` if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkParametricUmapParams <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "local_connectivity",
      "bandwidth",
      "mix_weight",
      "hidden_layers",
      "lr",
      "corr_weight",
      "n_epochs",
      "batch_size",
      "neg_sample_rate"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  checks <- list(
    "local_connectivity" = checkmate::qtest(x$local_connectivity, "N1"),
    "bandwidth" = checkmate::qtest(x$bandwidth, "N1"),
    "mix_weight" = checkmate::qtest(x$mix_weight, "N1"),
    "hidden_layers" = checkmate::testIntegerish(
      x$hidden_layers,
      lower = 1L,
      min.len = 1L,
      any.missing = FALSE
    ),
    "lr" = checkmate::qtest(x$lr, "N1(0,)"),
    "corr_weight" = checkmate::qtest(x$corr_weight, "N1"),
    "n_epochs" = checkmate::qtest(x$n_epochs, "I1[1,)"),
    "batch_size" = checkmate::qtest(x$batch_size, "I1[1,)"),
    "neg_sample_rate" = checkmate::qtest(x$neg_sample_rate, "I1[1,)")
  )

  failed <- names(checks)[!unlist(checks)]
  if (length(failed) > 0L) {
    return(sprintf(
      paste(
        "Element `%s` in parametric UMAP params does not conform.",
        "local_connectivity/bandwidth/mix_weight/corr_weight must be numeric,",
        "lr must be a positive numeric,",
        "hidden_layers must be a positive integer vector,",
        "and n_epochs/batch_size/neg_sample_rate must be positive integers."
      ),
      failed[1]
    ))
  }

  return(TRUE)
}

#' Assert parametric UMAP parameters
#'
#' @description Checkmate extension for asserting the parametric UMAP
#' parameters.
#'
#' @inheritParams checkParametricUmapParams
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertParametricUmapParams <- checkmate::makeAssertionFunction(
  checkParametricUmapParams
)

## umap gpu --------------------------------------------------------------------

### nearest neighbours ---------------------------------------------------------

#' Check GPU nearest neighbour parameters
#'
#' @description Checkmate extension for checking the GPU nearest neighbour
#' parameters.
#'
#' @param x The list to check.
#'
#' @return `TRUE` if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkNnParamsGpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }
  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "dist_metric",
      "n_list",
      "n_probes",
      "node_degree_final",
      "k_build",
      "n_tree",
      "delta",
      "rho",
      "beam_width",
      "max_beam_iters",
      "n_entry_points",
      "extract_knn"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }
  rules <- list(
    "dist_metric" = list(type = "choice", choices = c("cosine", "euclidean")),
    "n_list" = list(type = "nullable_int"),
    "n_probes" = list(type = "nullable_int"),
    "node_degree_final" = list(type = "nullable_int"),
    "k_build" = list(type = "nullable_int"),
    "n_tree" = list(type = "nullable_int"),
    "delta" = list(type = "fixed", rule = "N1"),
    "rho" = list(type = "nullable_numeric"),
    "beam_width" = list(type = "nullable_int"),
    "max_beam_iters" = list(type = "nullable_int"),
    "n_entry_points" = list(type = "nullable_int"),
    "extract_knn" = list(type = "fixed", rule = "B1")
  )
  res <- purrr::imap_lgl(x, \(val, name) {
    spec <- rules[[name]]
    # unknown keys pass through; the deprecated wrappers carry a `k` the GPU
    # params struct has no field for
    if (is.null(spec)) {
      return(TRUE)
    }
    if (spec$type == "choice") {
      checkmate::testChoice(val, spec$choices)
    } else if (spec$type == "nullable_int") {
      is.null(val) || checkmate::qtest(val, "I1")
    } else if (spec$type == "nullable_numeric") {
      is.null(val) || checkmate::qtest(val, "N1")
    } else {
      checkmate::qtest(val, spec$rule)
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(
      sprintf(
        paste(
          "The following element `%s` in GPU nearest neighbour params does",
          "not conform to the expected format.",
          "dist_metric must be one of 'cosine' or 'euclidean',",
          "n_list/n_probes/k/k_build/n_tree/beam_width/max_beam_iters/",
          "n_entry_points must be integers or NULL,",
          "extract_knn must be a boolean,",
          "delta must be a numeric,",
          "and rho must be a numeric or NULL."
        ),
        broken_elem
      )
    )
  }
  return(TRUE)
}

#' Assert GPU nearest neighbour parameters
#'
#' @description Checkmate extension for asserting the GPU nearest neighbour
#' parameters.
#'
#' @inheritParams checkNnParamsGpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertNnParamsGpu <- checkmate::makeAssertionFunction(checkNnParamsGpu)

### umap -----------------------------------------------------------------------

#' Check UMAP parameters (GPU version)
#'
#' @description Checkmate extension for checking the UMAP parameters (GPU
#' version).
#'
#' @param x The list to check.
#'
#' @return `TRUE` if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkUmapParamsGpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "local_connectivity",
      "bandwidth",
      "mix_weight",
      "lr",
      "n_epochs",
      "neg_sample_rate",
      "gamma",
      "optimiser",
      "init",
      "randomised"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  rules <- list(
    "local_connectivity" = list(type = "fixed", rule = "N1"),
    "bandwidth" = list(type = "fixed", rule = "N1"),
    "mix_weight" = list(type = "fixed", rule = "N1"),
    "lr" = list(type = "fixed", rule = "N1"),
    "n_epochs" = list(type = "nullable_int"),
    "neg_sample_rate" = list(type = "fixed", rule = "I1"),
    "gamma" = list(type = "fixed", rule = "N1"),
    "optimiser" = list(
      type = "choice",
      choices = c("sgd", "adam", "adam_parallel", "adam_gpu")
    ),
    "init" = list(type = "choice", choices = c("spectral", "pca", "random")),
    "randomised" = list(type = "fixed", rule = "B1")
  )

  res <- purrr::imap_lgl(x, \(val, name) {
    spec <- rules[[name]]
    if (spec$type == "choice") {
      checkmate::testChoice(val, spec$choices)
    } else if (spec$type == "nullable_int") {
      is.null(val) || checkmate::qtest(val, "I1")
    } else {
      checkmate::qtest(val, spec$rule)
    }
  })

  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "Element `%s` in UMAP params does not conform.",
        "local_connectivity/bandwidth/mix_weight/lr/gamma must be numeric,",
        "neg_sample_rate must be an integer,",
        "n_epochs must be a positive integer or NULL,",
        "optimiser must be one of 'sgd'/'adam'/'adam_parallel'/'adam_gpu',",
        "init must be one of 'spectral'/'pca'/'random',",
        "and randomised must be logical."
      ),
      broken_elem
    ))
  }

  return(TRUE)
}

#' Assert UMAP parameters
#'
#' @description Checkmate extension for asserting the UMAP parameters.
#'
#' @inheritParams checkUmapParamsGpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertUmapParamsGpu <- checkmate::makeAssertionFunction(checkUmapParamsGpu)

### tsne ----------------------------------------------------------------------

#' Check t-SNE parameters (GPU version)
#'
#' @description Checkmate extension for checking the t-SNE parameters (GPU
#' version).
#'
#' @param x The list to check.
#'
#' @return `TRUE` if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkTsneParamsGpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "lr",
      "n_epochs",
      "early_exag_iter",
      "early_exag_factor",
      "late_exag_factor",
      "theta",
      "n_interp_points",
      "init",
      "randomised"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  rules <- list(
    "lr" = list(type = "nullable_numeric"),
    "n_epochs" = list(type = "fixed", rule = "I1[1,)"),
    "early_exag_iter" = list(type = "fixed", rule = "I1[1,)"),
    "early_exag_factor" = list(type = "fixed", rule = "N1"),
    "late_exag_factor" = list(type = "nullable_numeric"),
    "theta" = list(type = "fixed", rule = "N1[0,1]"),
    "n_interp_points" = list(type = "fixed", rule = "I1[1,)"),
    "init" = list(type = "choice", choices = c("pca", "spectral", "random")),
    "randomised" = list(type = "fixed", rule = "B1")
  )

  res <- purrr::imap_lgl(x, \(val, name) {
    spec <- rules[[name]]
    if (spec$type == "choice") {
      checkmate::testChoice(val, spec$choices)
    } else if (spec$type == "nullable_numeric") {
      is.null(val) || checkmate::qtest(val, "N1")
    } else {
      checkmate::qtest(val, spec$rule)
    }
  })

  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "Element `%s` in t-SNE params does not conform.",
        "lr and late_exag_factor must be numeric or NULL,",
        "n_epochs/early_exag_iter/n_interp_points must be integers >= 1,",
        "early_exag_factor must be numeric, theta must be numeric in [0,1],",
        "init must be one of 'pca'/'spectral'/'random',",
        "and randomised must be logical."
      ),
      broken_elem
    ))
  }

  return(TRUE)
}

#' Assert t-SNE parameters (GPU version)
#'
#' @description Checkmate extension for asserting the t-SNE parameters (GPU
#' version).
#'
#' @inheritParams checkTsneParamsGpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertTsneParamsGpu <- checkmate::makeAssertionFunction(checkTsneParamsGpu)

## gpu-accelerated k means -----------------------------------------------------

#' Check KMeansGpu parameters
#'
#' @description Checkmate extension for checking KMeansGpu parameters.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkKMeansGpuParams <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }
  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "k_means_iter",
      "k_means_init",
      "metric",
      "fixed",
      "quantise"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }
  if (!checkmate::qtest(x[["k_means_iter"]], "I1[1,)")) {
    return("Element `k_means_iter` must be an integer >= 1.")
  }
  if (!is.null(x[["k_means_init"]])) {
    if (
      !checkmate::test_choice(
        x[["k_means_init"]],
        c("random", "parallel", "plusplus")
      )
    ) {
      return(
        paste(
          "Element `k_means_init` must be NULL or one of",
          "'random', 'parallel', or 'plusplus'."
        )
      )
    }
  }
  if (!checkmate::qtest(x[["fixed"]], "B1")) {
    return("Element `fixed` must be a logical scalar.")
  }
  if (!checkmate::qtest(x[["quantise"]], "B1")) {
    return("Element `quantise` must be a logical scalar.")
  }

  res <- checkmate::checkChoice(x[["metric"]], c("euclidean", "cosine"))

  if (!isTRUE(res)) {
    return(res)
  }

  return(TRUE)
}

#' Assert KMeansGpu parameters
#'
#' @description Checkmate extension for asserting the KMeansGpu parameters.
#'
#' @inheritParams checkKMeansGpuParams
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertKMeansGpuParams <- checkmate::makeAssertionFunction(
  checkKMeansGpuParams
)

## single cells ----------------------------------------------------------------

### harmony gpu ----------------------------------------------------------------

#' Check Harmony v2 GPU parameters
#'
#' @description Checkmate extension for checking Harmony v2 GPU parameters.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkScHarmonyParamsV2Gpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "k",
      "sigma",
      "theta",
      "lambda",
      "max_iter_kmeans",
      "max_iter_harmony",
      "epsilon_kmeans",
      "epsilon_harmony",
      "window_size",
      "alpha",
      "tau",
      "batch_proportion_cutoff",
      "use_dynamic_lambda",
      "csr_cube_count",
      "k_means_iter",
      "k_means_init",
      "fixed",
      "quantise"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  integer_rules <- list(
    "k" = c("I1[1,)", "0"),
    "max_iter_kmeans" = "I1[1,)",
    "max_iter_harmony" = "I1[1,)",
    "window_size" = "I1[1,)",
    "csr_cube_count" = "I1[1,)",
    "k_means_iter" = "I1[1,)"
  )

  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(integer_rules)) {
      checkmate::qtest(x, integer_rules[[name]])
    } else {
      TRUE
    }
  })

  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The following element `%s` in Harmony v2 GPU parameters is incorrect:",
        "max_iter_kmeans, max_iter_harmony, window_size, csr_cube_count,",
        "and k_means_iter must be integers >= 1. k must be NULL or an integer."
      ),
      broken_elem
    ))
  }

  vector_rules <- list(
    "sigma" = "N+[0,)",
    "theta" = "N+[0,)",
    "lambda" = "N+[0,)"
  )

  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(vector_rules)) {
      checkmate::qtest(x, vector_rules[[name]])
    } else {
      TRUE
    }
  })

  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The following element `%s` in Harmony v2 GPU parameters is incorrect:",
        "sigma, theta, and lambda must be numeric vectors",
        "with non-negative values."
      ),
      broken_elem
    ))
  }

  scalar_rules <- list(
    "epsilon_kmeans" = "N1(0,)",
    "epsilon_harmony" = "N1(0,)",
    "alpha" = "N1(0,1)",
    "tau" = "N1[0,)",
    "batch_proportion_cutoff" = "N1(0,)"
  )

  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(scalar_rules)) {
      checkmate::qtest(x, scalar_rules[[name]])
    } else {
      TRUE
    }
  })

  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The following element `%s` in Harmony v2 GPU parameters is incorrect:",
        "epsilon_kmeans, epsilon_harmony, and batch_proportion_cutoff must be > 0;",
        "alpha must be in (0,1); tau must be >= 0."
      ),
      broken_elem
    ))
  }

  bool_rules <- list(
    "use_dynamic_lambda" = "B1",
    "fixed" = "B1",
    "quantise" = "B1"
  )

  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(bool_rules)) {
      checkmate::qtest(x, bool_rules[[name]])
    } else {
      TRUE
    }
  })

  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The following element `%s` in Harmony v2 GPU parameters is incorrect:",
        "use_dynamic_lambda, fixed, and quantise must be single logicals."
      ),
      broken_elem
    ))
  }

  k_means_init <- x[["k_means_init"]]
  if (!is.null(k_means_init) && !checkmate::qtest(k_means_init, "S1")) {
    return("k_means_init must be NULL or a single string.")
  }

  return(TRUE)
}

#' Assert Harmony v2 GPU parameters
#'
#' @description Checkmate extension for asserting the Harmony v2 GPU parameters.
#'
#' @inheritParams checkScHarmonyParamsV2Gpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertScHarmonyParamsV2Gpu <- checkmate::makeAssertionFunction(
  checkScHarmonyParamsV2Gpu
)

### fast clustering gpu --------------------------------------------------------

#' Check GPU fast clustering parameters
#'
#' @description Checkmate extension for checking the GPU fast Louvain
#' clustering parameters.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkScFastClusterGpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "k_means_iter",
      "k_means_init",
      "fixed",
      "quantise",
      "same_weight",
      "full_snn",
      "pruning",
      "snn_similarity",
      "louvain_iters",
      "multi_level_louvain",
      "k",
      "knn_method",
      "ann_dist"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  int_rules <- list(
    "k_means_iter" = "I1[1,)",
    "louvain_iters" = "I1[1,)",
    "k" = "I1[1,)",
    "n_trees" = "I1[1,)",
    "m" = "I1[1,)",
    "ef_construction" = "I1[1,)",
    "ef_search" = "I1[1,)"
  )
  bool_rules <- list(
    "fixed" = "B1",
    "quantise" = "B1",
    "same_weight" = "B1",
    "full_snn" = "B1",
    "multi_level_louvain" = "B1"
  )
  optional_rules <- list(
    "pruning" = c("N1[0,1]", "0"),
    "search_budget" = c("I1[1,)", "0"),
    "ef_budget" = c("I1[1,)", "0"),
    "n_list" = c("I1[1,)", "0"),
    "n_probe" = c("I1[1,)", "0")
  )
  numeric_rules <- list(
    "delta" = "N1[0,1]",
    "diversify_prob" = "N1[0,1]"
  )

  rules <- c(int_rules, bool_rules, optional_rules, numeric_rules)

  res <- purrr::imap_lgl(x, \(elem, name) {
    if (name %in% names(rules)) checkmate::qtest(elem, rules[[name]]) else TRUE
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The element `%s` in the GPU fast clustering parameters is incorrect.",
        "See ?params_sc_fast_cluster_gpu and ?bixverse::params_knn_defaults."
      ),
      broken_elem
    ))
  }

  k_means_init <- x[["k_means_init"]]
  if (
    !is.null(k_means_init) &&
      !checkmate::test_choice(
        k_means_init,
        c("random", "parallel", "plusplus")
      )
  ) {
    return(paste(
      "Element `k_means_init` must be NULL or one of 'random',",
      "'parallel' or 'plusplus'."
    ))
  }

  res <- checkmate::checkChoice(x[["snn_similarity"]], c("jaccard", "rank"))
  if (!isTRUE(res)) {
    return(res)
  }

  # the GPU k-means takes its metric from here, and rejects "manhattan"
  checkmate::checkChoice(x[["ann_dist"]], c("euclidean", "cosine"))
}

#' Assert GPU fast clustering parameters
#'
#' @description Checkmate extension for asserting the GPU fast Louvain
#' clustering parameters.
#'
#' @inheritParams checkScFastClusterGpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertScFastClusterGpu <- checkmate::makeAssertionFunction(
  checkScFastClusterGpu
)

## scrublet gpu ----------------------------------------------------------------

# Keys the kNN block may carry, per backend. The GPU indices take a strict
# subset of the CPU names, which is why the flat list needs `knn_backend` to
# disambiguate: `knn_method = "exhaustive"` is legal on both sides and means a
# different code path each time.
.SCRUBLET_GPU_KNN_KEYS <- list(
  gpu = c("k", "knn_method", "ann_dist", "n_list", "n_probe"),
  cpu = c(
    "k",
    "knn_method",
    "ann_dist",
    "n_trees",
    "search_budget",
    "delta",
    "diversify_prob",
    "ef_budget",
    "m",
    "ef_construction",
    "ef_search",
    "n_list",
    "n_probe"
  )
)

#' Check GPU Scrublet parameters
#'
#' @description Checkmate extension for checking the GPU Scrublet parameters.
#' The kNN block is validated against whichever backend `knn_backend` names.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkScrubletGpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "knn_backend",
      "log_transform",
      "mean_center",
      "normalise_variance",
      "target_size",
      "min_gene_var_pctl",
      "hvg_method",
      "loess_span",
      "clip_max",
      "n_bins",
      "binning_strategy",
      "no_pcs",
      "sim_doublet_ratio",
      "expected_doublet_rate",
      "stdev_doublet_rate",
      "n_bins_hist",
      "manual_threshold",
      "k",
      "knn_method",
      "ann_dist"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  backend <- x[["knn_backend"]]
  res <- checkmate::checkChoice(backend, c("gpu", "cpu"))
  if (!isTRUE(res)) {
    return(res)
  }

  int_rules <- list(
    "no_pcs" = "I1[1,)",
    "n_bins" = "I1[1,)",
    "n_bins_hist" = "I1[10,)"
  )
  bool_rules <- list(
    "log_transform" = "B1",
    "mean_center" = "B1",
    "normalise_variance" = "B1"
  )
  numeric_rules <- list(
    "min_gene_var_pctl" = "N1[0,1]",
    "loess_span" = "N1(0,)",
    "sim_doublet_ratio" = "N1(0,)",
    "expected_doublet_rate" = "N1[0,1]",
    "stdev_doublet_rate" = "N1[0,1]"
  )
  optional_rules <- list(
    "target_size" = c("N1(0,)", "0"),
    "clip_max" = c("N1(0,)", "0"),
    "manual_threshold" = c("N1[0,)", "0")
  )

  rules <- c(int_rules, bool_rules, numeric_rules, optional_rules)

  res <- purrr::imap_lgl(x, \(elem, name) {
    if (name %in% names(rules)) checkmate::qtest(elem, rules[[name]]) else TRUE
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The element `%s` in the GPU Scrublet parameters is incorrect.",
        "See ?params_scrublet_gpu."
      ),
      broken_elem
    ))
  }

  res <- checkmate::checkChoice(
    x[["hvg_method"]],
    c("vst", "mvb", "dispersion")
  )
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkChoice(
    x[["binning_strategy"]],
    c("equal_width", "equal_frequency")
  )
  if (!isTRUE(res)) {
    return(res)
  }

  # backend-dependent kNN block
  knn_block <- x[names(x) %in% .SCRUBLET_GPU_KNN_KEYS[[backend]]]

  if (backend == "cpu") {
    return(bixverse:::checkKnnParams(
      knn_block,
      required_params = .SCRUBLET_GPU_KNN_KEYS[["cpu"]]
    ))
  }

  res <- checkmate::checkNames(
    names(knn_block),
    must.include = .SCRUBLET_GPU_KNN_KEYS[["gpu"]]
  )
  if (!isTRUE(res)) {
    return(res)
  }

  # k = 0L is legal and asks Rust for sqrt(n_obs) * 0.5
  gpu_knn_rules <- list(
    "k" = "I1[0,)",
    "n_list" = c("I1[1,)", "0"),
    "n_probe" = c("I1[1,)", "0")
  )

  res <- purrr::imap_lgl(knn_block, \(elem, name) {
    if (name %in% names(gpu_knn_rules)) {
      checkmate::qtest(elem, gpu_knn_rules[[name]])
    } else {
      TRUE
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The element `%s` in the GPU kNN block is incorrect.",
        "k must be an integer >= 0 (0 = automatic);",
        "n_list and n_probe must be NULL or integers >= 1."
      ),
      broken_elem
    ))
  }

  res <- checkmate::checkChoice(
    knn_block[["knn_method"]],
    c("exhaustive", "ivf")
  )
  if (!isTRUE(res)) {
    return(res)
  }

  checkmate::checkChoice(knn_block[["ann_dist"]], c("euclidean", "cosine"))
}

#' Assert GPU Scrublet parameters
#'
#' @description Checkmate extension for asserting the GPU Scrublet parameters.
#'
#' @inheritParams checkScrubletGpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertScrubletGpu <- checkmate::makeAssertionFunction(checkScrubletGpu)
