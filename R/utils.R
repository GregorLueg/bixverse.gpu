# helpers ----------------------------------------------------------------------

## verbosity -------------------------------------------------------------------

#' Helper to parse the verbosity
#'
#' @param input Boolean or integer to parse
#'
#' @returns The integer controlling the verbosity
#'
#' @keywords internal
parse_verbosity <- function(input) {
  # checks
  checkmate::qassert(input, c("B1", "I1[0, 2]"))

  as.integer(sum(input))
}
