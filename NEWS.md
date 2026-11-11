# bixverse.gpu 0.7.0

## Features

* GPU PHATE: `phate_gpu()` on plain matrices, `phate_gpu_sc()` on
  `SingleCells`, `params_phate_gpu()` and `rs_phate_gpu()` /
  `rs_phate_from_knn_gpu()`. The kNN search and the randomised SVD and k-means
  of the default spectral landmark selection run on the GPU; diffusion time,
  MDS and the landmark interpolation stay on the CPU. `backend = "mlx"` moves
  the kNN and the spectral landmarks to MLX. Defaults follow manifolds-rs:
  `k = 5L`, a `6 * k + 5` wide search on `knn_method = "ivf"`, 2048 spectral
  landmarks. `phate_gpu_sc()` builds a fresh kNN by default, since the cached one is
  usually far narrower than PHATE wants. New vignette:
  [GPU-accelerated PHATE](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_phate.html).
* `params_umap_gpu(optimiser = "sgd_parallel")`: lock-free (Hogwild!) SGD on
  the CPU for `umap_gpu()` and `umap_gpu_sc()`. Not reproducible run to run
  with more than one thread.

## Breaking changes

* `umap_gpu()` and `umap_gpu_sc()` cut a precomputed kNN wider than `k` down
  to its nearest `k` neighbours, with a warning on stderr. They used every
  neighbour given. Different embeddings whenever the cached graph was wider
  than `k`.
* `umap_gpu()` and `umap_gpu_sc()` with `n_epochs = NULL`: only the SGD
  optimisers drop to 200 epochs above 10k samples, as in manifoldsR 0.6.0.
  `"adam"` now always gets 500 (was 200 above 10k).
* `"sgd"` applies an edge's negative samples as one step (manifolds-rs), so
  different embeddings at the same seed.
* `parametric_umap()` defaults to `min_dist = 0.5` (was `0.1`), matching the
  single default curve manifolds-rs now fits. Different embeddings with the
  defaults.
* Errors from the Rust side reach you as they are: `umap_gpu()`, `tsne_gpu()`
  and `parametric_umap()` no longer wrap them in "... computation failed: ".

* The neighbour graph is a `SingleCellConnectivity` now, following bixverse
  0.8.4, which moved it out of igraph and into Rust. `find_neighbours_gpu_sc()`
  and `bbknn_gpu_sc()` store one, and `find_clusters_gpu_sc()` clusters it.
  `leiden_clustering_gpu()` takes a `SingleCellConnectivity` or an undirected
  igraph; an igraph goes through `bixverse::as_sc_connectivity()` first.
* `rs_leiden_clustering_gpu()` takes one flattened, 1-indexed `edges` vector
  instead of `from` and `to`, the layout `bixverse::rs_graph_to_edges()`
  returns.
* `knn_method = "nndescent"` hands back the NN-descent kNN graph itself: no
  CAGRA index, no beam search. Rows the descent left short are topped up with
  an exact search of just those rows. The old behaviour is
  `knn_method = "cagra"`, now accepted by `generate_gpu_knn_sc()`,
  `find_neighbours_gpu_sc()`, `generate_knn_graph_gpu()`, `rs_gpu_knn()` and the
  GPU kNN blocks of Scrublet, BBKNN and fastMNN. Different neighbours at the
  same seed wherever `"nndescent"` was in use, which includes the defaults of
  `generate_gpu_knn_sc()`, `find_neighbours_gpu_sc()` and
  `generate_knn_graph_gpu()`. BBKNN and fastMNN query one set of cells against
  another, so `"nndescent"` there still builds and beam searches CAGRA.
* `extract_knn` is gone from `params_nn_gpu()` and
  `params_knn_gpu_defaults()`. The deprecated `extract_knn` arguments of
  `generate_knn_graph_gpu()` and the old CAGRA entry points map to
  `"nndescent"` (`TRUE`) or `"cagra"` (`FALSE`).
* `rs_gpu_knn()` runs through bixverse-rs instead of manifolds-rs, on wgpu and
  MLX.
* `harmony_v2_gpu_sc()` is now `harmony_gpu_sc()`, following bixverse 0.9.0
  where `harmony_sc()` runs v2. No alias. The embedding is still
  `"harmony_gpu"`.
* `params_sc_harmony_v2_gpu()` defaults to `use_dynamic_lambda = TRUE`, as on
  the CPU. Different embeddings with the defaults.
* `bbknn_gpu_sc()`: `trim = NULL` now means
  `10 * neighbours_within_batch * n_batches` (was `10 * neighbours_within_batch`)
  and the stored kNN is cut balanced per batch. Different graphs and kNN with
  the defaults.
* `umap_gpu()`, `tsne_gpu()`, `umap_gpu_sc()`, `tsne_gpu_sc()` and their `rs_*`
  functions lose `use_high_precision` and the automatic switch to fp64 above
  100k samples. They always run in fp32 now, with manifolds-rs doing the steps
  prone to catastrophic cancellation in fp64. Runs that used fp64 give different
  embeddings at the same seed.

## Chores

* `ann-search-rs` 0.12.2, `manifolds-rs` 0.9.1 and `bixverse-rs` 0.10.1. UMAP and
  tSNE pick up the manifolds-rs changes: `"nndescent"` builds the kNN graph
  directly there too, spectral initialisation gives different coordinates at
  the same seed, PCA initialisation runs its centring and SVD in fp64, and
  `.verbose` follows one grammar: `1L` prints timed stages and epochs every 50,
  `2L` adds parameters and the kNN internals.
* Needs bixverse >= 0.9.0 and manifoldsR >= 0.6.0. `igraph` and `Matrix` moved from Imports to
  Suggests; only the tests use them.

# bixverse.gpu 0.6.1

## Chores

* Builds without `faer`'s `npy` feature, through `ann-search-rs` 0.11.2,
  `manifolds-rs` 0.7.3 and `bixverse-rs` 0.9.0. This drops `npyz`, `pest` and
  `stacker` from the build tree. The `stacker` build script failed to find
  `windows.h` on the r-universe Windows ARM64 build, where the host is MSVC and
  the target is `aarch64-pc-windows-gnullvm`. No API change.

# bixverse.gpu 0.6.0

## Features

* Exact sparse PCA on the GPU. `calculate_pca_gpu_sc()` runs a block Lanczos
  on the device (WGPU and MLX) for every `svd_solver` except `"randomised"`,
  which keeps the randomised SVD. It matches the CPU Lanczos to f32 precision.
* Fast clustering (`fast_cluster_gpu_sc()`) can build the centroid graph from
  the cells instead of a kNN search over the centroids:
  `params_sc_fast_cluster_gpu(cell_mediated = TRUE, n_anchors = 5L)`. Every
  cell links its `n_anchors` nearest centroids.

## Breaking changes

* `calculate_pca_gpu_sc()` no longer ignores `svd_solver`. The default
  (`"auto"`) is exact and slower than the randomised SVD it used to run. Pass
  `pca_params = params_sc_pca(svd_solver = "randomised")` for the old behaviour.

## Internal

* Moved to bixverse-rs 0.8.1, manifolds-rs 0.7.2, ann-search-rs 0.11.1,
  cubecl 0.11, burn 0.22 and faer 0.24.

# bixverse.gpu 0.5.0

## Features

* Experimental MLX backend for Apple Silicon via mlx-c. Opt in at install
  time: `brew install mlx-c` (or point `MLX_C_PREFIX` at an install prefix),
  then install with `BIXVERSE_GPU_MLX=true`. Check with `mlx_available()`.
  Functions with an MLX path take `backend = c("wgpu", "mlx")`: PCA, kNN
  graphs, Harmony v2, SEACells, NMF (single, stabilised, consensus, k sweep),
  UMAP, tSNE and the `rs_cor_gpu()` / `rs_cov_gpu()` correlations. For UMAP and
  tSNE, `"mlx"` moves the kNN search and the device optimiser (`"adam_gpu"`,
  `"fft_3k_gpu"`); CPU optimisers stay on the CPU. MLX is fp32 only. WGPU stays
  the default and the only portable backend. 
* FFT-accelerated tSNE (`approx_type = "fft"` and `"fft_3k"`) now works on
  Windows too: FFTW was replaced by the pure-Rust rustfft.
* On macOS the CPU GEMMs in the Rust backends go through Apple Accelerate.

## Breaking changes

* The `rs_*` wrappers with an MLX path gained a `backend` argument, inserted
  before `verbose`. Named calls keep working (it defaults to `"wgpu"`);
  positional calls past `seed` need updating.

# bixverse.gpu 0.4.1

## Features

* Wired in the faster NEBULA via [`edge-rs`](https://crates.io/crates/edge-rs).
  This affects also the GPU path because parts of it run on CPU. To get exactly
  the same results, make sure you are on `bixverse` version "≥0.6.2".

# bixverse.gpu 0.4.0

## Features

* Improved speed across various methods:
  - GPU-accelerated kNN searches
  - Harmony (version 2) accelerated on the GPU
  - Randomised SVD
* GPU-accelerated fastMNN wired in via `fast_mnn_gpu_sc()`. Only the neighbour
  searches run on the device. Unlike `bixverse::fast_mnn_sc()`, it does **not**
  regenerate the PCA on batch-aware HVGs: it corrects the PCA already in the
  object. For the batch-aware flavour run `find_hvg_batch_aware_sc()` and
  `calculate_pca_gpu_sc()` first. See the single cell vignette.
* The GPU-accelerated randomised SVD for single cell are is now using 
  N_OVERSAMPLING 20 (instead of 100) and increased iterations from 2 to 4. 
  Across various benchmarks and synthetic data sets, the quality of the trailing 
  PCs has improved with the new thresholds while being faster.
* The GPU-accelerated NEBULA has now a parameter that enables that a minimum of 
  samples similar to the CPU-based version.

## Fixes

* The GPU-accelerated Harmony version could converge wrongly after one 
  iteration for some data sets. This has been fixed via a dampening factor.

# bixverse.gpu 0.3.6

## Features

* Quick-and-dirty Barnes-Hut tSNE optimiser from
  [qdtsne](https://github.com/libscran/qdtsne) wired into `tsne_gpu()` and
  `tsne_gpu_sc()` via `approx_type = "bh_qd"`. Tree depth is set with
  `max_depth` in `params_tsne_gpu()`. Runs on the CPU, like `"bh"`.

# bixverse.gpu 0.3.5

## Features

* GPU-accelerated [Sanity](https://www.nature.com/articles/s41587-026-03220-2) 
  wired in via [sanity-sc-rs](https://crates.io/crates/sanity-sc-rs). This gives
  you the GPU-accelerated equivalent of `bonsai_sc()`:`bonsai_gpu_sc()`.

# bixverse.gpu 0.3.4

## Feature

* GPU-accelerated FFT 3-kernel optimisation for tSNE wired in and the CPU 
  version of the same optimiser.
 
# bixverse.gpu 0.3.3

## Feature

* Faster GPU-accelerated SVD that also fixed a numerical problem in the same
  vein via new `bixverse-rs` release.

# bixverse.gpu 0.3.2

## Feature

* Default GPU-accelerated kNN search for tSNE was set to `"ivf"` – due to the
  usually larger k needed, `"ivf"` scales better than the GPU-accelerated 
  NNDescent approach.

# bixverse.gpu 0.3.1

## Features

* Agent skill for using bixverse.gpu, shipped under `inst/skills/` and
  installed with `install_agent_skill_gpu()`. Covers which GPU function
  replaces which bixverse step, what has no GPU path, and the traps. Same
  layout as `bixverse::install_agent_skill()`, with the API index generated by
  `data-raw/generate_api_index.R`.
* [GPU-accelerated BBKNN](https://academic.oup.com/bioinformatics/article/36/3/964/5545955).
* A GPU-accelerated implementation of [NEBULA](https://www.nature.com/articles/s42003-021-02146-6) 
  for lots and lots of cells.

# bixverse.gpu 0.3.0

## Features

* Take in substantial improvements on the Rust back-ends. The kNN searches
  now return Euclidean distances instead of squared Euclidean distances.

## Deprecated

* The previous interface of `generate_cagra_knn_sc()`, 
  `find_neighbours_cagra_sc()`, `params_sc_cagra()`, `rs_cagra_gpu_knn()`, 
  `rs_exhaustive_gpu_knn()`, `params_sc_ivf()` has been deprecated. These are
  now all wired into `generate_gpu_knn_sc()`.

# bixverse.gpu 0.2.8

## Features

* GPU-accelerated NMF for single cells and meta cells: `nmf_gpu_sc()`,
  `stabilised_nmf_gpu_sc()`, `consensus_nmf_gpu_sc()` and
  `nmf_k_sweep_gpu_sc()`, each dispatching on `SingleCells` and `MetaCells`.

# bixverse.gpu 0.2.7

## Features

* GPU-accelerated Scrublet via `scrublet_gpu_sc()`. The randomised sparse SVD,
  the projection of the simulated doublets and the kNN over the combined
  embedding run on GPU. Full parity with `bixverse::scrublet_sc()`, `group_by`
  included, and the same `ScrubletRes` object comes back.
* `params_scrublet_gpu()` picks between the GPU and the CPU nearest neighbour
  indices through `knn_backend`, with the `knn` block validated against
  whichever one was asked for. `params_knn_gpu_defaults()` holds the GPU-side
  defaults.
* Improvements in terms of kernels taken from `ann-search-rs`.

# bixverse.gpu 0.2.6

## Features

- GPU-accelerated fast clustering for single cell.

# bixverse.gpu 0.2.5

## Features

* GPU-accelerated SEACells: both Frank-Wolfe solves, the B-gradient argmin and
  the per-cell A columns, are executed on GPU, increasing the speed on larger
  data sets.
* Version bumps to get improved GPU performance for the UMAP optimiser from
  `manifolds-rs`.

# bixverse.gpu 0.2.4

## Features

* Update to `bixverse-rs` to enable the new streaming engine.

# bixverse.gpu 0.2.3

## Fix

* Potentially broken `manifoldsR` dependency.

# bixverse.gpu 0.2.2

## Features

* GPU-accelerated SCENIC
* Improvement in the GPU-accelerated kNN searches ported over from 
  `ann-search-rs`.
* Handrolled correlation kernels to increase performance there.

# bixverse.gpu 0.2.1

## Features

* GPU-accelerated UMAP optimisation exposed.
* GPU-accelerated kNN generation for both tSNE and UMAP (longer term also a
  GPU-accelerated tSNE optimiser planned). Additionally, the kNN graphs can
  be used more broadly.
* `umap_gpu_sc()` and `tsne_gpu_sc()` wire the GPU UMAP and t-SNE paths
  directly into the `SingleCells` workflow, mirroring `bixverse::umap_sc()`
  and `bixverse::tsne_sc()`.

# bixverse.gpu 0.2.0

Major release.

## Features

* CPU-backend of parametric UMAP changed to `flex` from burn for better speed.
  Reduces also dependencies and packages and easier installation on Linux.
* GPU-accelerated methods for single cell:
  - kNN graphs are now multi-modal (see bixverse version `"0.4.0"`)
  - sparse, randomised SVD on GPU available
  - GPU-accelerated Harmony (version 2).
* New GPU-accelerated methods:
  - k-means clustering
  - Correlations and co-variances supported on GPU.
* Option addded to save parametric UMAP models

# bixverse.gpu 0.1.3

## Features

* Port to rextendr `0.5.0` and updated Rust <> R APIs and various version 
  updates.

# bixverse.gpu 0.1.2

## Features

* Version bumps on different Rust crates.

# bixverse.gpu 0.1.1

## Features

* Parametric UMAP added; leverages a small neural net that can be trained and
  then used subsequently for repeat predictions.

# bixverse.gpu 0.1.0

## Features

* Package release with GPU-accelerated kNN searches implemented for 
  `SingleCells`.