# helpers ----------------------------------------------------------------------

## device ----------------------------------------------------------------------

#' Is a GPU available
#'
#' @description
#' Reports whether a usable WGPU adapter is present. Every function in this
#' package needs one, and without it the Rust side aborts rather than returning
#' an error, so this is the way to branch before calling into it.
#'
#' The probe initialises the same client the GPU functions use, so a `TRUE`
#' means they will genuinely run rather than that a device merely exists. It is
#' cached for the session, and the client is left warm, so the first real call
#' afterwards skips the setup cost.
#'
#' @returns Boolean. `TRUE` when a WGPU adapter could be initialised.
#'
#' @examples
#' if (gpu_available()) {
#'   # ... GPU path
#' }
#'
#' @export
gpu_available <- function() {
  rs_gpu_available()
}

#' Assert that a GPU is available
#'
#' @description
#' Hard errors when no WGPU adapter can be initialised. Sits at the top of the
#' user-facing functions so the failure names the function the user called
#' rather than the `rs_` wrapper underneath. The Rust side carries the same
#' guard for direct `rs_` calls.
#'
#' @returns Invisibly `TRUE`. Called for the error.
#'
#' @keywords internal
assert_gpu <- function() {
  if (!gpu_available()) {
    stop(
      paste(
        "No usable GPU adapter found. bixverse.gpu needs a working WGPU",
        "adapter; check your GPU drivers and",
        "https://burn.dev/books/cubecl/getting-started/installation.html.",
        "Probe with `gpu_available()`."
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

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
