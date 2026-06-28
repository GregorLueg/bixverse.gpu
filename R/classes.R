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
#'
#' @keywords internal
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

# KMeansClusterGPU -------------------------------------------------------------

#' Construct a KMeansClusterGPU object
#'
#' @param centroids Numeric matrix of shape k x features.
#' @param assignments Integer vector of length samples (1-indexed).
#' @param k Integer. Number of clusters.
#' @param metric Character. Distance metric used.
#'
#' @returns A `KMeansClusterGPU` S3 object.
#'
#' @keywords internal
new_kmeans_cluster_gpu <- function(centroids, assignments, k, metric) {
  structure(
    list(
      centroids = centroids,
      assignments = assignments,
      k = k,
      metric = metric
    ),
    class = "KMeansClusterGPU"
  )
}

## getters ---------------------------------------------------------------------

#' Get cluster assignments
#'
#' @param x A `KMeansClusterGPU` object.
#'
#' @returns Integer vector of length samples with cluster assignments
#'   (1-indexed).
#'
#' @export
membership <- function(x) {
  UseMethod("membership")
}

#' @rdname membership
#' @export
membership.KMeansClusterGPU <- function(x) {
  x$assignments
}

#' Get cluster centroids
#'
#' @param x A `KMeansClusterGPU` object.
#'
#' @returns Numeric matrix of shape k x features.
#'
#' @export
get_centroids <- function(x) {
  UseMethod("get_centroids")
}

#' @rdname get_centroids
#'
#' @export
get_centroids.KMeansClusterGPU <- function(x) {
  # checks
  checkmate::assertClass(x, "KMeansCluster")

  x$centroids
}

## primitives ------------------------------------------------------------------

#' Print a KMeansClusterGPU object
#'
#' @param x A `KMeansClusterGPU` object.
#' @param ... Further arguments (ignored).
#'
#' @returns Invisibly returns `x`.
#'
#' @export
#'
#' @keywords internal
print.KMeansClusterGPU <- function(x, ...) {
  n <- length(x$assignments)
  sizes <- tabulate(x$assignments, nbins = x$k)
  cat("KMeansClusterGPU\n")
  cat("  metric:    ", x$metric, "\n")
  cat("  k:         ", x$k, "\n")
  cat("  n:         ", n, "\n")
  cat(
    "  sizes:      min=",
    min(sizes),
    " median=",
    median(sizes),
    " max=",
    max(sizes),
    "\n"
  )
  invisible(x)
}
