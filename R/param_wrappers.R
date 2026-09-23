# param wrappers ---------------------------------------------------------------

## single cells ----------------------------------------------------------------

### scrublet GPU ---------------------------------------------------------------

#' Wrapper function for GPU Scrublet doublet detection parameters
#'
#' @description GPU counterpart to [bixverse::params_scrublet()]. Two
#' differences from the CPU list. The `pca` sub-list is gone: the GPU SVD is
#' always randomised, so `random_svd` and `sparse` have nothing to switch and
#' `no_pcs` is a plain argument. And the kNN block is backend dependent, see
#' `knn_backend`.
#'
#' @param sim_doublet_ratio Numeric. Number of doublets to simulate relative to
#' the number of observed cells. Defaults to `1.5`.
#' @param expected_doublet_rate Numeric in `[0, 1]`. Expected doublet rate,
#' typically 0.05-0.10 depending on cell loading. Defaults to `0.1`.
#' @param stdev_doublet_rate Numeric in `[0, 1]`. Uncertainty in the expected
#' doublet rate. Defaults to `0.02`.
#' @param n_bins_histogram Integer. Histogram bins for the Otsu threshold
#' search. Defaults to `100L`.
#' @param manual_threshold Optional numeric. Fixed doublet score threshold. If
#' `NULL`, Otsu's method picks it.
#' @param no_pcs Integer. Number of principal components. Defaults to `30L`.
#' @param normalisation List. Optional overrides. See
#' [bixverse::params_norm_doublets_defaults()] for the available parameters:
#' `log_transform`, `mean_center`, `normalise_variance`, `target_size`.
#' @param hvg List. Optional overrides. See [bixverse::params_hvg_defaults()]
#' for the available parameters: `min_gene_var_pctl`, `hvg_method`,
#' `loess_span`, `clip_max`, `n_bins`, `binning_strategy`.
#' @param knn_backend String. One of `"gpu"` or `"cpu"`. Picks which nearest
#' neighbour index runs over the combined observed-plus-simulated embedding,
#' and with it which keys `knn` accepts. `"gpu"` is the fast default; `"cpu"`
#' buys the exact CPU indices at the cost of a host round trip on a matrix
#' that is `(1 + sim_doublet_ratio) * n_cells` rows tall.
#' @param knn List. Optional overrides for the kNN block. Validated against
#' [params_knn_gpu_defaults()] when `knn_backend = "gpu"` and against
#' [bixverse::params_knn_defaults()] when `knn_backend = "cpu"`. Unknown keys
#' are an error, not a silent pass-through. Defaults to `list(k = 0L)`, which
#' asks Rust to pick `k`.
#'
#' @details Leave `knn_method` alone unless you have a reason. `"exhaustive"`
#' is the default and is the right answer for Scrublet on the GPU arm.
#'
#' Scrublet queries at a high `k` by construction: the count is taken over an
#' embedding `(1 + sim_doublet_ratio) * n_cells` rows tall, and `k = 0L` then
#' scales `k` by the same factor, so a 20k-cell run searches at `k` around
#' 175. Exhaustive barely notices `k`, since the scan is the cost and `k` only
#' sizes the top-k selection. NN-descent notices a lot: its build degree
#' tracks `k`, so the descent does more work per node as `k` climbs. Measured
#' at 20k cells and 30 PCs, exhaustive took 0.98s against 38.8s for
#' NN-descent at `k = 200`. NN-descent only came out ahead at `k = 10`.
#'
#' `"ivf"` is the one worth trying: it was the quickest of the three across
#' that sweep and holds a Pearson above 0.99 against exhaustive.
#'
#' Recall matters more here than elsewhere. The doublet score is a neighbour
#' count, so a backend that drops neighbours biases every score downwards.
#'
#' @returns A flat named list with all GPU Scrublet parameters.
#'
#' @export
#'
#' @references Wolock, et al., Cell Syst, 2020
params_scrublet_gpu <- function(
  sim_doublet_ratio = 1.5,
  expected_doublet_rate = 0.1,
  stdev_doublet_rate = 0.02,
  n_bins_histogram = 100L,
  manual_threshold = NULL,
  no_pcs = 30L,
  normalisation = list(),
  hvg = list(),
  knn_backend = c("gpu", "cpu"),
  knn = list(k = 0L)
) {
  knn_backend <- match.arg(knn_backend)

  # checks
  checkmate::qassert(sim_doublet_ratio, "N1(0,)")
  checkmate::qassert(expected_doublet_rate, "N1[0,1]")
  checkmate::qassert(stdev_doublet_rate, "N1[0,1]")
  checkmate::qassert(n_bins_histogram, "I1[10,)")
  checkmate::qassert(manual_threshold, c("N1[0,)", "0"))
  checkmate::qassert(no_pcs, "I1[1,)")
  checkmate::assertChoice(knn_backend, c("gpu", "cpu"))
  checkmate::assertList(normalisation)
  checkmate::assertList(hvg)
  checkmate::assertList(knn)

  knn_defaults <- if (knn_backend == "gpu") {
    params_knn_gpu_defaults()
  } else {
    bixverse::params_knn_defaults()
  }

  # the two backends share five key names, so a CPU-only knob silently doing
  # nothing on the GPU arm is the easy mistake here. Catch it where the user
  # types rather than three layers down in Rust.
  unknown_knn <- setdiff(names(knn), names(knn_defaults))
  if (length(unknown_knn) > 0L) {
    stop(sprintf(
      "Unknown kNN parameter(s) for backend '%s': %s. Allowed: %s.",
      knn_backend,
      paste(unknown_knn, collapse = ", "),
      paste(names(knn_defaults), collapse = ", ")
    ))
  }

  knn <- utils::modifyList(knn_defaults, knn, keep.null = TRUE)

  # `"nndescent"` is the package-wide name, Rust only knows `"nndescent_gpu"`
  if (knn_backend == "gpu") {
    knn[["knn_method"]] <- .normalise_gpu_knn_method(knn[["knn_method"]])
  }

  params <- list(
    knn_backend = knn_backend,
    normalisation = utils::modifyList(
      bixverse::params_norm_doublets_defaults(),
      normalisation,
      keep.null = TRUE
    ),
    hvg = utils::modifyList(
      bixverse::params_hvg_defaults(),
      hvg,
      keep.null = TRUE
    ),
    no_pcs = no_pcs,
    sim_doublet_ratio = sim_doublet_ratio,
    expected_doublet_rate = expected_doublet_rate,
    stdev_doublet_rate = stdev_doublet_rate,
    n_bins_hist = n_bins_histogram,
    manual_threshold = manual_threshold,
    knn = knn
  )

  purrr::list_flatten(params, name_spec = "{inner}")
}

### bbknn GPU ------------------------------------------------------------------

#' Wrapper function for the GPU BBKNN parameters
#'
#' @description GPU counterpart to [bixverse::params_sc_bbknn()]. Same BBKNN
#' knobs, but the kNN block is the GPU one, see [params_knn_gpu_defaults()].
#'
#' @details Two keys of the GPU kNN block do nothing here and are rejected
#' rather than silently ignored. `k` is set by `neighbours_within_batch`, and
#' `extract_knn` only applies to a self-query, whereas BBKNN builds one index
#' per batch and queries each with every cell.
#'
#' @param neighbours_within_batch Integer. Number of neighbours to consider
#' per batch. Defaults to `3L`.
#' @param set_op_mix_ratio Numeric. Mixing ratio between union (1.0) and
#' intersection (0.0). Defaults to `1.0`.
#' @param local_connectivity Numeric. UMAP connectivity computation parameter,
#' how many nearest neighbours of each cell are assumed to be fully connected.
#' Defaults to `1.0`.
#' @param trim Optional integer. Trim the neighbours of each cell to these many
#' top connectivities. May help with population independence and improve the
#' tidiness of clustering. If `NULL`, it defaults to
#' `10 * neighbours_within_batch`.
#' @param knn List. Optional overrides for the kNN block. Validated against
#' [params_knn_gpu_defaults()] minus `k` and `extract_knn`. Unknown keys are
#' an error, not a silent pass-through.
#'
#' @returns A flat named list with all GPU BBKNN parameters.
#'
#' @export
#'
#' @references Polański, et al., Bioinformatics, 2020
params_sc_bbknn_gpu <- function(
  neighbours_within_batch = 3L,
  set_op_mix_ratio = 1.0,
  local_connectivity = 1.0,
  trim = NULL,
  knn = list()
) {
  # checks
  checkmate::qassert(neighbours_within_batch, "I1[1,)")
  checkmate::qassert(set_op_mix_ratio, "N1[0,1]")
  checkmate::qassert(local_connectivity, "N1")
  checkmate::qassert(trim, c("0", "I1[1,)"))
  checkmate::assertList(knn)

  knn_defaults <- params_knn_gpu_defaults()
  knn_defaults[c("k", "extract_knn")] <- NULL

  unknown_knn <- setdiff(names(knn), names(knn_defaults))
  if (length(unknown_knn) > 0L) {
    stop(sprintf(
      "Unknown kNN parameter(s) for GPU BBKNN: %s. Allowed: %s.",
      paste(unknown_knn, collapse = ", "),
      paste(names(knn_defaults), collapse = ", ")
    ))
  }

  knn <- utils::modifyList(knn_defaults, knn, keep.null = TRUE)
  knn[["knn_method"]] <- .normalise_gpu_knn_method(knn[["knn_method"]])

  params <- list(
    neighbours_within_batch = neighbours_within_batch,
    set_op_mix_ratio = set_op_mix_ratio,
    local_connectivity = local_connectivity,
    trim = trim,
    knn = knn
  )

  purrr::list_flatten(params, name_spec = "{inner}")
}
