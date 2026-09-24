---
name: bixverse-gpu
description: How to USE the bixverse.gpu R package, the GPU (WGPU, any vendor, no CUDA) companion to bixverse and manifoldsR. Covers GPU PCA, Harmony v2, kNN (exhaustive, IVF, NN-descent/CAGRA), fast clustering, BBKNN, Scrublet, SEACells, SCENIC, NMF and NEBULA on SingleCells / SingleCellsSubset / MetaCells objects, plus GPU UMAP, t-SNE, parametric UMAP, k-means and correlations on plain matrices. Use this whenever a task involves bixverse.gpu or any of its functions (calculate_pca_gpu_sc, harmony_v2_gpu_sc, find_neighbours_gpu_sc, umap_gpu, tsne_gpu, parametric_umap, scrublet_gpu_sc, scenic_grn_sc_gpu, generate_seacells_gpu_sc, nebula_gpu_sc, *_nmf_gpu_sc, k_means_cluster_gpu, rs_cor_gpu, params_*_gpu), or when a user wants to speed up a bixverse single cell workflow or a manifoldsR embedding on a GPU. Apply even when the package is not named but the user is running large single cell data through bixverse and asks about GPU acceleration.
---

# Using bixverse.gpu

bixverse.gpu is a set of GPU drop-ins for the slow steps of `bixverse` single
cell workflows and `manifoldsR` embeddings. Rust underneath via extendr, GPU
code on cubecl/burn over WGPU: Metal on macOS, Vulkan on Linux, DX12 or Vulkan
on Windows. It is not a stand-alone package. The objects, the preprocessing and
most of the result classes come from `bixverse`, so load both.

The whole R surface is `lifecycle: experimental` and does shift between
versions. Do not guess at signatures. Check `references/api-index.md` for
whether something exists, then `?fn` for how to call it.

## The three things to know before writing any code

**1. Check the GPU first.** `gpu_available()` returns `TRUE` or `FALSE`. Every
user-facing function hard errors without a usable WGPU adapter; there is no
silent CPU fallback at the function level. `FALSE` means drivers, not the
package.

**2. It is a drop-in, not a parallel universe.** The GPU version of a bixverse
function has `gpu` in its name, takes the same object, and returns the same
thing: the updated `SingleCells`, or bixverse's own result class (`ScrubletRes`,
`MetaCells`, `ScenicGrn`, `ScNebula`, the NMF result classes). Everything
downstream is plain bixverse. Where the knobs are identical the GPU function
reuses bixverse's `params_*()` bundle (`params_scenic()`,
`params_sc_seacells()`, `params_sc_pca()`, `params_nmf_hals()`,
`params_nmf_consensus()`). Where they differ there is a `params_*_gpu()`
bundle.

**3. The GPU does not always win.** Host-device transfer and kernel launch are
a fixed cost. On a few thousand cells the CPU path is often as fast or faster;
the payoff grows with cell count. Results agree with CPU in structure, not bit
for bit (f32 reductions, different tie breaking, randomised sketches). Compare
the two by downstream metrics, never by a diff.

## Where to look

| Task | Read |
|---|---|
| Installing, drivers, Windows, missing FFT t-SNE | `references/install.md` |
| Single cell: PCA, Harmony v2, kNN, fast clustering, BBKNN, UMAP / t-SNE on the object | `references/single-cell.md` |
| Single cell: Scrublet, SEACells, SCENIC, NMF, NEBULA | `references/single-cell.md` |
| Plain matrices: `umap_gpu()`, `tsne_gpu()`, parametric UMAP, kNN graphs, k-means, correlation / covariance | `references/matrix-methods.md` |
| Does function X exist? What's it called? | `references/api-index.md` |

For anything about the bixverse side (loading, QC, HVG, clustering, markers,
the S7 conventions), use the `bixverse` skill. Install it with
`bixverse::install_agent_skill()` if it is not there.

## Smoke test

Synthetic data, no downloads, a couple of seconds. Run it to confirm the
install and the GPU before debugging anything else.

```r
library(bixverse)
library(bixverse.gpu)

stopifnot(gpu_available())

dir_sc <- file.path(tempdir(), "bixverse_gpu_smoke")
dir.create(dir_sc, showWarnings = FALSE)

syn <- generate_single_cell_test_data()

obj <- load_r_data(
  object = SingleCells(dir_data = dir_sc),
  counts = syn$counts,
  obs = syn$obs,
  var = syn$var,
  sc_qc_param = params_sc_min_quality(
    min_unique_genes = 45L, min_lib_size = 300L, min_cells = 500L
  ),
  streaming = 0L
)

obj <- find_hvg_sc(obj, hvg_no = 30L)
obj <- calculate_pca_gpu_sc(obj, no_pcs = 10L)
obj <- find_neighbours_gpu_sc(obj, knn_method = "exhaustive", k = 15L)
obj <- find_clusters_sc(obj, res = 1, name = "clusters")
obj <- umap_gpu_sc(obj, use_knn = TRUE)

obj  # PCA, kNN, sNN and a "umap" embedding should all show up

unlink(dir_sc, recursive = TRUE, force = TRUE)
```

## Traps that bite everywhere

- **Two kNN vocabularies.** `params_nn_gpu()` (used by `find_neighbours_gpu_sc()`,
  `umap_gpu()`, `tsne_gpu()`, `generate_knn_graph_gpu()`) takes `dist_metric`,
  `n_list`, `n_probes`, `n_tree`, `node_degree_final`. The `knn = list(...)`
  blocks inside `params_scrublet_gpu()`, `params_sc_bbknn_gpu()` and
  `params_sc_fast_cluster_gpu()` are validated against
  `params_knn_gpu_defaults()`, which says `ann_dist`, `n_probe`, `graph_k`.
  Unknown keys error, so print `params_knn_gpu_defaults()` rather than copying
  names across.
- **NN-descent is a low-`k` method.** It is the default `knn_method` in most
  functions, and it is the right one at `k = 15`. Above `k` of roughly 30 the
  exhaustive search overtakes it, by more than an order of magnitude at
  `k = 200`. Use `"exhaustive"` or `"ivf"` for high `k` (t-SNE at high
  perplexity, Scrublet).
- **A missing prerequisite warns and returns the object unchanged**, same as in
  bixverse. `calculate_pca_gpu_sc()` without HVGs, or `harmony_v2_gpu_sc()`
  without a PCA, does nothing but warn. Print the object after each step.
- **Not everything has a GPU path.** GRNBoost2 SCENIC is rejected outright
  (use `bixverse::scenic_grn_sc()`). NEBULA has no REML and no `MetaCells`
  method. NMF errors above rank 128. FFT t-SNE does not exist on Windows. All
  of these error clearly; none of them fall back.
- **`rs_*` functions are raw extendr bindings with no input validation.** Use
  the R wrapper. The exceptions are `rs_cor_gpu()` and `rs_cov_gpu()`, which
  have no wrapper and are the intended entry points.
- **Deprecated names still work but warn**: `find_neighbours_cagra_sc()`,
  `generate_cagra_knn_sc()`, `params_sc_cagra()`, `params_sc_ivf()`, and the
  `gpu_method` / `ivf_params` / `dist_metric` arguments on the kNN functions.
  Use `find_neighbours_gpu_sc()` / `generate_gpu_knn_sc()` with `knn_method`
  and `params_nn_gpu()`.
