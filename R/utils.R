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

## cache provenance ------------------------------------------------------------

#' Parents of a GPU manifold embedding (UMAP, t-SNE)
#'
#' @description
#' Builds the `from` argument for `bixverse::set_embedding()` so the resulting
#' embedding joins the provenance chain rather than being recorded as a root.
#' The source embedding is read from `cache_modality` while the result is
#' written under `modality`, and the two differ for `"wnn"`, so both parents
#' are spelled out modality qualified.
#'
#' Kept local rather than reaching for the equivalent `bixverse` internal: four
#' lines are not worth coupling to an unexported function.
#'
#' @param embd_to_use String. Name of the source embedding.
#' @param cache_modality String. Modality the source embedding was read from.
#' @param modality String. Modality the result is written to.
#' @param has_knn Boolean. Whether a cached kNN fed the manifold.
#'
#' @returns Character vector of parent artefact names.
#'
#' @keywords internal
.manifold_from_gpu <- function(
  embd_to_use,
  cache_modality,
  modality,
  has_knn
) {
  # checks
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(cache_modality, "S1")
  checkmate::qassert(modality, "S1")
  checkmate::qassert(has_knn, "B1")

  c(
    sprintf("%s:%s", cache_modality, embd_to_use),
    if (has_knn) sprintf("%s:knn", modality) else NULL
  )
}
