# checkmate extensions ---------------------------------------------------------

## scrublet gpu ----------------------------------------------------------------

# Keys the kNN block may carry, per backend. The GPU indices take a strict
# subset of the CPU names, which is why the flat list needs `knn_backend` to
# disambiguate: `knn_method = "exhaustive"` is legal on both sides and means a
# different code path each time.
.SCRUBLET_GPU_KNN_KEYS <- list(
  gpu = c(
    "k",
    "knn_method",
    "ann_dist",
    "n_list",
    "n_probe",
    "graph_k",
    "k_build",
    "n_tree",
    "delta",
    "rho",
    "refine_knn",
    "beam_width",
    "max_beam_iters",
    "n_entry_points",
    "extract_knn"
  ),
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
    "n_probe",
    "extract_knn"
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
    return(bixverse:::checkKnnParams(knn_block))
  }

  .check_gpu_knn_block(knn_block, .SCRUBLET_GPU_KNN_KEYS[["gpu"]])
}

#' Check a GPU kNN parameter block
#'
#' @description Shared validation for the flat GPU kNN block, used by both the
#' GPU Scrublet and the GPU BBKNN parameters. They take the same keys bar `k`
#' and `extract_knn`, which BBKNN ignores, hence `required` rather than a
#' fixed name set.
#'
#' @param x The kNN block to check.
#' @param required Character vector of key names that must be present.
#'
#' @return `TRUE` if the check was successful, otherwise an error message.
#'
#' @keywords internal
.check_gpu_knn_block <- function(x, required) {
  res <- checkmate::checkNames(names(x), must.include = required)
  if (!isTRUE(res)) {
    return(res)
  }

  # k = 0L is legal and asks Rust for sqrt(n_obs) * 0.5
  rules <- list(
    "k" = "I1[0,)",
    "n_list" = c("I1[1,)", "0"),
    "n_probe" = c("I1[1,)", "0"),
    "graph_k" = c("I1[1,)", "0"),
    "k_build" = c("I1[1,)", "0"),
    "n_tree" = c("I1[1,)", "0"),
    "delta" = "N1",
    "rho" = c("N1", "0"),
    "refine_knn" = c("I1[0,)", "0"),
    "beam_width" = c("I1[1,)", "0"),
    "max_beam_iters" = c("I1[1,)", "0"),
    "n_entry_points" = c("I1[1,)", "0"),
    "extract_knn" = "B1"
  )

  res <- purrr::imap_lgl(x, \(elem, name) {
    if (name %in% names(rules)) checkmate::qtest(elem, rules[[name]]) else TRUE
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The element `%s` in the GPU kNN block is incorrect.",
        "k must be an integer >= 0 (0 = automatic);",
        "delta must be numeric, rho numeric or NULL,",
        "extract_knn a boolean,",
        "and the remaining knobs NULL or integers >= 1."
      ),
      broken_elem
    ))
  }

  # `nndescent` is what the user types, `nndescent_gpu` what the parameter
  # wrappers store. Both are legal here, hand-built lists included.
  res <- checkmate::checkChoice(
    x[["knn_method"]],
    c("exhaustive", "ivf", "nndescent", "nndescent_gpu", "cagra")
  )
  if (!isTRUE(res)) {
    return(res)
  }

  checkmate::checkChoice(x[["ann_dist"]], c("euclidean", "cosine"))
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

## bbknn gpu -------------------------------------------------------------------

# `k` and `extract_knn` are deliberately absent: BBKNN takes its neighbour
# count from `neighbours_within_batch`, and the per-batch searches are
# cross-queries, so there is no index graph to hand back.
.BBKNN_GPU_KNN_KEYS <- c(
  "knn_method",
  "ann_dist",
  "n_list",
  "n_probe",
  "graph_k",
  "k_build",
  "n_tree",
  "delta",
  "rho",
  "refine_knn",
  "beam_width",
  "max_beam_iters",
  "n_entry_points"
)

#' Check GPU BBKNN parameters
#'
#' @description Checkmate extension for checking the GPU BBKNN parameters.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkScBbknnGpu <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "neighbours_within_batch",
      "set_op_mix_ratio",
      "local_connectivity",
      "trim",
      "knn_method",
      "ann_dist"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  rules <- list(
    "neighbours_within_batch" = "I1[1,)",
    "trim" = c("0", "I1[1,)"),
    "set_op_mix_ratio" = "N1[0,1]",
    "local_connectivity" = "N1"
  )

  res <- purrr::imap_lgl(x, \(elem, name) {
    if (name %in% names(rules)) checkmate::qtest(elem, rules[[name]]) else TRUE
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(sprintf(
      paste(
        "The element `%s` in the GPU BBKNN parameters is incorrect.",
        "neighbours_within_batch must be an integer >= 1;",
        "trim must be NULL or an integer >= 1;",
        "set_op_mix_ratio must be a numeric in [0, 1];",
        "local_connectivity must be numeric.",
        "See ?params_sc_bbknn_gpu."
      ),
      broken_elem
    ))
  }

  .check_gpu_knn_block(
    x[names(x) %in% .BBKNN_GPU_KNN_KEYS],
    .BBKNN_GPU_KNN_KEYS
  )
}

#' Assert GPU BBKNN parameters
#'
#' @description Checkmate extension for asserting the GPU BBKNN parameters.
#'
#' @inheritParams checkScBbknnGpu
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertionFunction()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertScBbknnGpu <- checkmate::makeAssertionFunction(checkScBbknnGpu)
