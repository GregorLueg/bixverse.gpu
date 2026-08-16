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