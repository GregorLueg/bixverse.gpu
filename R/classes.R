# ParametricUmapModel ----------------------------------------------------------

## primitives ------------------------------------------------------------------

### print ----------------------------------------------------------------------

#' Print a parametric UMAP model
#'
#' @param x A `ParametricUmapModel` object.
#' @param ... Ignored.
#'
#' @return Invisibly returns `x`.
#'
#' @export
print.ParametricUmapModel <- function(x, ...) {
  p <- x$params
  cat("Parametric UMAP Model\n")
  cat(sprintf(
    "  Samples: %d | Features: %d | Embedding dims: %d\n",
    p$n_samples,
    p$n_features,
    p$n_dim
  ))
  cat(sprintf(
    "  k: %d | min_dist: %.3f | spread: %.1f | knn: %s\n",
    p$k,
    p$min_dist,
    p$spread,
    p$knn_method
  ))
  invisible(x)
}

### predict --------------------------------------------------------------------

#' Predict embeddings for new data using a trained parametric UMAP model
#'
#' @param object A `ParametricUmapModel` object.
#' @param newdata Numerical matrix or data frame. New data of shape
#' samples x features. Must have the same number of features as the
#' training data.
#' @param ... Ignored.
#'
#' @return A numerical matrix with dimensions samples x n_dim.
#'
#' @export
predict.ParametricUmapModel <- function(object, newdata, ...) {
  if (is.data.frame(newdata)) {
    newdata <- as.matrix(newdata)
  }
  checkmate::assert_matrix(
    newdata,
    mode = "numeric",
    any.missing = FALSE,
    min.rows = 1,
    ncols = object$params$n_features
  )
  rs_parametric_umap_predict(
    model = object$model$ptr,
    data = newdata
  )
}
