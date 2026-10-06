# ------------------------------------------------------------------------------
# GPU-accelerated single cell embeddings: UMAP, t-SNE and Bonsai trees.
# ------------------------------------------------------------------------------

# gpu umap ---------------------------------------------------------------------

#' Run UMAP on a SingleCells object (GPU)
#'
#' @description
#' GPU-accelerated counterpart to [bixverse::umap_sc()]. Pulls an embedding
#' (defaulting to PCA) off the object, runs [umap_gpu()] on it (GPU kNN plus
#' GPU Adam optimiser by default), and writes the resulting embedding back
#' into `sc_cache$other_embeddings[[slot_name]]`.
#'
#' When `use_knn = TRUE` (the default), the kNN graph already stored on the
#' object is reused via [bixverse::sc_knn_to_nearest_neighbours()]. Otherwise
#' a fresh GPU kNN is built from the chosen embedding.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param use_knn Boolean. Use the kNN graph found in the object. Defaults to
#' `TRUE`. Only reused if the modality lines up; otherwise a fresh GPU kNN is
#' generated.
#' @param embd_to_use String. The embedding to use for UMAP. Must be available
#' in the object for the chosen modality.
#' @param slot_name String. The name of this embedding within the object.
#' Defaults to `"umap"`.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL`, all will be used.
#' @param modality String. On which modality to run UMAP. One of
#' `c("rna", "adt", "wnn")`. The two latter options are only available on
#' `SingleCellsMultiModal`.
#' @param n_dim Integer. Number of UMAP dimensions. Defaults to `2L`.
#' @param k Integer. Number of nearest neighbours. Defaults to `15L`.
#' @param min_dist Numeric. Minimum distance between embedded points. Defaults
#' to `0.5`.
#' @param spread Numeric. Effective scale of embedded points. Defaults to
#' `1.0`.
#' @param knn_method String. GPU (approximate) nearest neighbour method. One
#' of `c("nndescent", "exhaustive", "ivf")`.
#' @param nn_params Named list. GPU kNN parameters, see [params_nn_gpu()].
#' @param umap_params Named list. UMAP (GPU) parameters, see
#' [params_umap_gpu()].
#' @param seed Integer. For reproducibility.
#' @param use_high_precision Optional boolean. Fine-grained fp32 vs fp64
#' control for the optimiser. GPU kNN is always fp32.
#' @param .verbose Boolean or integer. Controls verbosity.
#'
#' @return The object with a `"umap"` embedding added. If the requested
#' embedding is missing, returns the object unchanged with a warning.
#'
#' @seealso [umap_gpu()], [bixverse::umap_sc()], [tsne_gpu_sc()]
#'
#' @export
#'
#' @import bixverse
umap_gpu_sc <- S7::new_generic(
  name = "umap_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    use_knn = TRUE,
    embd_to_use = "pca",
    slot_name = "umap",
    no_embd_to_use = NULL,
    modality = c("rna", "adt", "wnn"),
    n_dim = 2L,
    k = 15L,
    min_dist = 0.5,
    spread = 1.0,
    knn_method = c("nndescent", "exhaustive", "ivf"),
    nn_params = params_nn_gpu(),
    umap_params = params_umap_gpu(),
    seed = 42L,
    use_high_precision = NULL,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

#' @method umap_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(umap_gpu_sc, SingleCells) <- function(
  object,
  use_knn = TRUE,
  embd_to_use = "pca",
  slot_name = "umap",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  k = 15L,
  min_dist = 0.5,
  spread = 1.0,
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  knn_method <- match.arg(knn_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(use_knn, "B1")
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(slot_name, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(n_dim, "I1[1,)")
  checkmate::qassert(k, "I1[2,)")
  checkmate::qassert(min_dist, "N1[0,)")
  checkmate::qassert(spread, "N1[0,)")
  assertNnGpuParams(nn_params)
  assertUmapGpuParams(umap_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  cache_modality <- if (modality == "wnn") "rna" else modality

  # embedding
  available <- get_available_embeddings(object, modality = cache_modality)
  if (!(embd_to_use %in% available)) {
    warning(sprintf(
      "Embedding '%s' not found on the object. Returning object as is.",
      embd_to_use
    ))
    return(object)
  }

  # hard tier: the manifold is written back onto the object, and it is read
  # from `cache_modality` while the kNN comes from `modality`
  assert_sc_state(object, artefacts = embd_to_use, modality = cache_modality)
  if (modality == "wnn" || use_knn) {
    assert_sc_state(object, artefacts = "knn", modality = modality)
  }
  embd <- get_embedding(
    x = object,
    embd_name = embd_to_use,
    modality = cache_modality
  )
  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take]
  }

  # knn
  knn <- if (modality == "wnn") {
    bixverse:::.get_manifoldsr_knn_from_wnn(x = object)
  } else if (use_knn) {
    bixverse:::.get_manifoldsr_knn(x = object, modality = modality)
  } else {
    NULL
  }

  if (.verbose) {
    message("Running GPU UMAP.")
  }

  umap_embd <- umap_gpu(
    data = embd,
    knn = knn,
    n_dim = n_dim,
    k = k,
    min_dist = min_dist,
    spread = spread,
    knn_method = knn_method,
    nn_params = nn_params,
    umap_params = umap_params,
    seed = seed,
    use_high_precision = use_high_precision,
    .verbose = .verbose
  )

  rownames(umap_embd) <- rownames(embd)
  colnames(umap_embd) <- sprintf("umap_%s", seq_len(ncol(umap_embd)))

  object <- set_embedding(
    x = object,
    embd = umap_embd,
    name = slot_name,
    modality = modality,
    from = .manifold_from_gpu(
      embd_to_use = embd_to_use,
      cache_modality = cache_modality,
      modality = modality,
      has_knn = !is.null(knn)
    )
  )

  return(object)
}

# gpu tsne ---------------------------------------------------------------------

#' Run t-SNE on a SingleCells object (GPU)
#'
#' @description
#' GPU-accelerated counterpart to [bixverse::tsne_sc()]. Runs [tsne_gpu()] on
#' an embedding pulled from the object. The kNN runs on the GPU; with the
#' default `"fft_3k_gpu"` the optimiser does too.
#'
#' t-SNE derives the number of neighbours from `perplexity` on the Rust side
#' (the usual `3 * perplexity` convention). To avoid a silent mismatch with
#' the cached kNN, `use_knn` defaults to `FALSE`: every call generates a
#' fresh GPU kNN sized to the requested perplexity. Handy for sweeping
#' perplexities since the kNN is cheap on GPU.
#'
#' @param object `SingleCells` (or `SingleCellsMultiModal`) class.
#' @param use_knn Boolean. Defaults to `FALSE`. Set to `TRUE` to reuse the
#' cached kNN; only sensible when the stored `k` is at least
#' `3 * perplexity`.
#' @param embd_to_use String. The embedding to use for t-SNE. Must be
#' available in the object for the chosen modality.
#' @param slot_name String. The name of this embedding within the object.
#' Defaults to `"tsne"`.
#' @param no_embd_to_use Optional integer. Number of embedding dimensions to
#' use. If `NULL`, all will be used.
#' @param modality String. On which modality to run t-SNE. One of
#' `c("rna", "adt", "wnn")`. The two latter options are only available on
#' `SingleCellsMultiModal`.
#' @param n_dim Integer. Number of t-SNE dimensions. Currently only `2L` is
#' supported. Defaults to `2L`.
#' @param perplexity Numeric. Perplexity parameter. Typical values between 5
#' and 50. Defaults to `20.0`.
#' @param approx_type String. Approximation method. One of `"fft_3k_gpu"`
#' (three-kernel FFT on the GPU, the default), `"bh"` (Barnes-Hut), `"bh_qd"`
#' (quick-and-dirty Barnes-Hut, depth-capped tree), `"fft"` or `"fft_3k"` (CPU
#' FFT interpolation, four or three kernels). The CPU FFT variants are
#' Unix-only. See [tsne_gpu()].
#' @param knn_method String. GPU (approximate) nearest neighbour method. One
#' of `c("ivf", "exhaustive", "nndescent")`. Default is `"ivf"` here, as it
#' deals usually better with the high k in tSNE.
#' @param nn_params Named list. GPU kNN parameters, see [params_nn_gpu()].
#' @param tsne_params Named list. t-SNE (GPU) parameters, see
#' [params_tsne_gpu()].
#' @param seed Integer. For reproducibility.
#' @param use_high_precision Optional boolean. Fine-grained fp32 vs fp64
#' control. GPU kNN is always fp32. Ignored for `"fft_3k_gpu"`.
#' @param .verbose Boolean or integer. Controls verbosity.
#'
#' @return The object with a `"tsne"` embedding added. If the requested
#' embedding is missing, returns the object unchanged with a warning.
#'
#' @seealso [tsne_gpu()], [bixverse::tsne_sc()], [umap_gpu_sc()]
#'
#' @export
#'
#' @import bixverse
tsne_gpu_sc <- S7::new_generic(
  name = "tsne_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    use_knn = FALSE,
    embd_to_use = "pca",
    slot_name = "tsne",
    no_embd_to_use = NULL,
    modality = c("rna", "adt", "wnn"),
    n_dim = 2L,
    perplexity = 20.0,
    approx_type = c("fft_3k_gpu", "bh", "bh_qd", "fft", "fft_3k"),
    knn_method = c("ivf", "exhaustive", "nndescent"),
    nn_params = params_nn_gpu(),
    tsne_params = params_tsne_gpu(),
    seed = 42L,
    use_high_precision = NULL,
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

#' @method tsne_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(tsne_gpu_sc, SingleCells) <- function(
  object,
  use_knn = FALSE,
  embd_to_use = "pca",
  slot_name = "tsne",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  perplexity = 20.0,
  approx_type = c("fft_3k_gpu", "bh", "bh_qd", "fft", "fft_3k"),
  knn_method = c("ivf", "exhaustive", "nndescent"),
  nn_params = params_nn_gpu(),
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
) {
  modality <- match.arg(modality)
  approx_type <- match.arg(approx_type)
  knn_method <- match.arg(knn_method)

  checkmate::assertTRUE(S7::S7_inherits(object, SingleCells))
  checkmate::qassert(use_knn, "B1")
  checkmate::qassert(embd_to_use, "S1")
  checkmate::qassert(slot_name, "S1")
  checkmate::qassert(no_embd_to_use, c("I1", "0"))
  checkmate::qassert(n_dim, "I1[2,2]")
  checkmate::qassert(perplexity, "N1[1,)")
  assertNnGpuParams(nn_params)
  assertTsneGpuParams(tsne_params)
  checkmate::qassert(seed, "I1")
  checkmate::qassert(use_high_precision, c("0", "B1"))
  checkmate::qassert(.verbose, c("B1", "I1[0,2]"))

  if (modality != "rna" && !S7::S7_inherits(object, SingleCellsMultiModal)) {
    stop(sprintf(
      "modality = '%s' is only supported for SingleCellsMultiModal.",
      modality
    ))
  }

  cache_modality <- if (modality == "wnn") "rna" else modality

  # embedding
  available <- get_available_embeddings(object, modality = cache_modality)
  if (!(embd_to_use %in% available)) {
    warning(sprintf(
      "Embedding '%s' not found on the object. Returning object as is.",
      embd_to_use
    ))
    return(object)
  }

  # hard tier: the manifold is written back onto the object, and it is read
  # from `cache_modality` while the kNN comes from `modality`
  assert_sc_state(object, artefacts = embd_to_use, modality = cache_modality)
  if (modality == "wnn" || use_knn) {
    assert_sc_state(object, artefacts = "knn", modality = modality)
  }
  embd <- get_embedding(
    x = object,
    embd_name = embd_to_use,
    modality = cache_modality
  )
  if (!is.null(no_embd_to_use)) {
    to_take <- min(c(no_embd_to_use, ncol(embd)))
    embd <- embd[, 1:to_take]
  }

  # knn - default is regenerate on GPU so perplexity drives k
  knn <- if (modality == "wnn") {
    bixverse:::.get_manifoldsr_knn_from_wnn(x = object)
  } else if (use_knn) {
    bixverse:::.get_manifoldsr_knn(x = object, modality = modality)
  } else {
    NULL
  }

  if (.verbose) {
    message("Running GPU t-SNE.")
  }

  tsne_embd <- tsne_gpu(
    data = embd,
    knn = knn,
    n_dim = n_dim,
    perplexity = perplexity,
    approx_type = approx_type,
    knn_method = knn_method,
    nn_params = nn_params,
    tsne_params = tsne_params,
    seed = seed,
    use_high_precision = use_high_precision,
    .verbose = .verbose
  )

  rownames(tsne_embd) <- rownames(embd)
  colnames(tsne_embd) <- sprintf("tsne_%s", seq_len(ncol(tsne_embd)))

  object <- set_embedding(
    x = object,
    embd = tsne_embd,
    name = slot_name,
    modality = modality,
    from = .manifold_from_gpu(
      embd_to_use = embd_to_use,
      cache_modality = cache_modality,
      modality = modality,
      has_knn = !is.null(knn)
    )
  )

  return(object)
}

# ------------------------------------------------------------------------------
# GPU-accelerated Bonsai:
# - Wraps `rs_sc_bonsai_gpu`, which runs Sanity on the device and the tree
#   search on the CPU.
# - Everything around the Rust call (candidate genes, cells, the timings) is
#   bixverse's own `.bonsai_sc_run()`, so the result is the same `BonsaiTree`
#   S3 object as `bixverse::bonsai_sc()` and its print, plot and relayout
#   methods work unchanged.
# - `SingleCells` and `MetaCells` get a method, as for `bixverse::bonsai_sc()`;
#   the metacells go through `.bonsai_mc_run()` with their in-memory counts.
# ------------------------------------------------------------------------------

# bonsai (gpu) -----------------------------------------------------------------

#' Build a Bonsai tree over the cells, Sanity on the GPU
#'
#' @description GPU counterpart of [bixverse::bonsai_sc()]. Sanity, which turns
#' the raw counts into posterior log fold changes with error bars and is most of
#' the runtime on the CPU, runs on the WGPU backend. Genes are streamed through
#' it in chunks and only the ones passing Bonsai's signal-to-noise filter are
#' kept. The tree search and the layout run on the CPU, as they do in
#' `bixverse`.
#'
#' The device computes in `f32`, so the posteriors match the CPU run to within
#' the resolution of Sanity's variance grid rather than bit for bit, and a gene
#' sitting right at the signal-to-noise threshold can end up on the other side.
#'
#' @param object `SingleCells` or `MetaCells` class.
#' @param hvg Optional integer. Restrict the candidate genes to these. Please
#' provide 1-indexed genes here! If `NULL`, every gene in the object is a
#' candidate.
#' @param bonsai_params List. See [bixverse::params_sc_bonsai()].
#' @param .verbose Boolean or integer. Controls verbosity and returns run times.
#' `FALSE` -> quiet, `TRUE` or `1L` -> normal verbosity, `2L` -> detailed
#' verbosity.
#'
#' @returns A `BonsaiTree` S3 object, see [bixverse::bonsai_sc()].
#'
#' @export
#'
#' @references de Groot, et al., Nat. Biotechnol., 2026; Breda, et al., Nat.
#' Biotechnol., 2021.
bonsai_gpu_sc <- S7::new_generic(
  name = "bonsai_gpu_sc",
  dispatch_args = "object",
  fun = function(
    object,
    hvg = NULL,
    bonsai_params = bixverse::params_sc_bonsai(),
    .verbose = TRUE
  ) {
    assert_gpu()

    S7::S7_dispatch()
  }
)

## SingleCells -----------------------------------------------------------------

#' @method bonsai_gpu_sc SingleCells
#'
#' @export
#'
#' @import bixverse
S7::method(bonsai_gpu_sc, SingleCells) <- function(
  object,
  hvg = NULL,
  bonsai_params = bixverse::params_sc_bonsai(),
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, bixverse::SingleCells))

  bixverse:::.bonsai_sc_run(
    object = object,
    hvg = hvg,
    bonsai_params = bonsai_params,
    runner = rs_sc_bonsai_gpu,
    .verbose = .verbose
  )
}

## MetaCells -------------------------------------------------------------------

#' @method bonsai_gpu_sc MetaCells
#'
#' @export
#'
#' @import bixverse
S7::method(bonsai_gpu_sc, MetaCells) <- function(
  object,
  hvg = NULL,
  bonsai_params = bixverse::params_sc_bonsai(),
  .verbose = TRUE
) {
  # checks
  checkmate::assertTRUE(S7::S7_inherits(object, bixverse::MetaCells))

  bixverse:::.bonsai_mc_run(
    object = object,
    hvg = hvg,
    bonsai_params = bonsai_params,
    runner = rs_mc_bonsai_gpu,
    .verbose = .verbose
  )
}
