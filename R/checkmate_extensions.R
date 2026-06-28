# checkmate extensions ---------------------------------------------------------

## cagra -----------------------------------------------------------------------

#' Check CAGRA parameters
#'
#' @description Checkmate extension for checking CAGRA parameters.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkScCagraParams <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }
  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "k_query",
      "ann_dist",
      "k",
      "k_build",
      "refine_sweeps",
      "max_iters",
      "n_trees",
      "delta",
      "rho",
      "beam_width",
      "max_beam_iters",
      "n_entry_points"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }
  # ann_dist must be one of the allowed values
  if (!checkmate::test_choice(x[["ann_dist"]], c("euclidean", "cosine"))) {
    return("Element `ann_dist` must be one of 'euclidean' or 'cosine'.")
  }
  # Required integer fields (must not be NULL)
  required_int_rules <- list(
    "k_query" = "I1[1,)",
    "refine_sweeps" = "I1[1,)"
  )
  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(required_int_rules)) {
      checkmate::qtest(x, required_int_rules[[name]])
    } else {
      TRUE
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(
      sprintf(
        paste(
          "The following element `%s` in CAGRA parameters is incorrect:",
          "k_query and refine_sweeps must be integers >= 1."
        ),
        broken_elem
      )
    )
  }
  # Optional integer fields (NULL or integer >= 1)
  optional_int_rules <- list(
    "k" = c("I1[1,)", "0"),
    "k_build" = c("I1[1,)", "0"),
    "max_iters" = c("I1[1,)", "0"),
    "n_trees" = c("I1[1,)", "0"),
    "beam_width" = c("I1[1,)", "0"),
    "max_beam_iters" = c("I1[1,)", "0"),
    "n_entry_points" = c("I1[1,)", "0")
  )
  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(optional_int_rules)) {
      checkmate::qtest(x, optional_int_rules[[name]])
    } else {
      TRUE
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(
      sprintf(
        paste(
          "The following element `%s` in CAGRA parameters is incorrect:",
          "k, k_build, max_iters, n_trees, beam_width, max_beam_iters,",
          "and n_entry_points must be NULL or integers >= 1."
        ),
        broken_elem
      )
    )
  }
  # Required scalar numeric
  scalar_rules <- list(
    "delta" = "N1(0,)"
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
    return(
      sprintf(
        paste(
          "The following element `%s` in CAGRA parameters is incorrect:",
          "delta must be a numeric value > 0."
        ),
        broken_elem
      )
    )
  }
  # Optional scalar numeric (NULL or in (0, 1])
  optional_numeric_rules <- list(
    "rho" = c("N1(0,1]", "0")
  )
  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(optional_numeric_rules)) {
      checkmate::qtest(x, optional_numeric_rules[[name]])
    } else {
      TRUE
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(
      sprintf(
        paste(
          "The following element `%s` in CAGRA parameters is incorrect:",
          "rho must be NULL or a numeric value in (0, 1]."
        ),
        broken_elem
      )
    )
  }
  return(TRUE)
}

#' Assert CAGRA parameters
#'
#' @description Checkmate extension for asserting the CAGRA parameters.
#'
#' @inheritParams checkScCagraParams
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertScCagraParams <- checkmate::makeAssertionFunction(checkScCagraParams)

## ivf -------------------------------------------------------------------------

#' Check IVF parameters
#'
#' @description Checkmate extension for checking IVF parameters.
#'
#' @param x The list to check/assert.
#'
#' @return \code{TRUE} if the check was successful, otherwise an error message.
#'
#' @keywords internal
checkScIvfParams <- function(x) {
  res <- checkmate::checkList(x)
  if (!isTRUE(res)) {
    return(res)
  }

  res <- checkmate::checkNames(
    names(x),
    must.include = c(
      "k",
      "ann_dist",
      "nlist",
      "nprobe",
      "nquery",
      "max_iters",
      "seed"
    )
  )
  if (!isTRUE(res)) {
    return(res)
  }

  if (!checkmate::test_choice(x[["ann_dist"]], c("euclidean", "cosine"))) {
    return("Element `ann_dist` must be one of 'euclidean' or 'cosine'.")
  }

  required_int_rules <- list(
    "k" = "I1[1,)",
    "seed" = "I1[0,)"
  )
  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(required_int_rules)) {
      checkmate::qtest(x, required_int_rules[[name]])
    } else {
      TRUE
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(
      sprintf(
        paste(
          "The following element `%s` in IVF parameters is incorrect:",
          "k must be an integer >= 1 and seed must be an integer >= 0."
        ),
        broken_elem
      )
    )
  }

  optional_int_rules <- list(
    "nlist" = c("I1[1,)", "0"),
    "nprobe" = c("I1[1,)", "0"),
    "nquery" = c("I1[1,)", "0"),
    "max_iters" = c("I1[1,)", "0")
  )
  res <- purrr::imap_lgl(x, \(x, name) {
    if (name %in% names(optional_int_rules)) {
      checkmate::qtest(x, optional_int_rules[[name]])
    } else {
      TRUE
    }
  })
  if (!isTRUE(all(res))) {
    broken_elem <- names(res)[which(!res)][1]
    return(
      sprintf(
        paste(
          "The following element `%s` in IVF parameters is incorrect:",
          "nlist, nprobe, nquery, and max_iters must be NULL or integers >= 1."
        ),
        broken_elem
      )
    )
  }

  return(TRUE)
}

#' Assert IVF parameters
#'
#' @description Checkmate extension for asserting the IVF parameters.
#'
#' @inheritParams checkScIvfParams
#'
#' @param .var.name Name of the checked object to print in assertions. Defaults
#' to the heuristic implemented in checkmate.
#' @param add Collection to store assertion messages. See
#' [checkmate::makeAssertCollection()].
#'
#' @return Invisibly returns the checked object if the assertion is successful.
#'
#' @keywords internal
assertScIvfParams <- checkmate::makeAssertionFunction(checkScIvfParams)

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
