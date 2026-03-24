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
