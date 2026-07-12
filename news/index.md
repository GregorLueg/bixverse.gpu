# Changelog

## bixverse.gpu 0.2.1

### Features

- GPU-accelerated UMAP optimisation exposed.
- GPU-accelerated kNN generation for both tSNE and UMAP (longer term
  also a GPU-accelerated tSNE optimiser planned). Additionally, the kNN
  graphs can be used more broadly.
- [`umap_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu_sc.md)
  and
  [`tsne_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu_sc.md)
  wire the GPU UMAP and t-SNE paths directly into the `SingleCells`
  workflow, mirroring
  [`bixverse::umap_sc()`](https://gregorlueg.github.io/bixverse/reference/umap_sc.html)
  and
  [`bixverse::tsne_sc()`](https://gregorlueg.github.io/bixverse/reference/tsne_sc.html).

## bixverse.gpu 0.2.0

Major release.

### Features

- CPU-backend of parametric UMAP changed to `flex` from burn for better
  speed. Reduces also dependencies and packages and easier installation
  on Linux.
- GPU-accelerated methods for single cell:
  - kNN graphs are now multi-modal (see bixverse version `"0.4.0"`)
  - sparse, randomised SVD on GPU available
  - GPU-accelerated Harmony (version 2).
- New GPU-accelerated methods:
  - k-means clustering
  - Correlations and co-variances supported on GPU.
- Option addded to save parametric UMAP models

## bixverse.gpu 0.1.3

### Features

- Port to rextendr `0.5.0` and updated Rust \<\> R APIs and various
  version updates.

## bixverse.gpu 0.1.2

### Features

- Version bumps on different Rust crates.

## bixverse.gpu 0.1.1

### Features

- Parametric UMAP added; leverages a small neural net that can be
  trained and then used subsequently for repeat predictions.

## bixverse.gpu 0.1.0

### Features

- Package release with GPU-accelerated kNN searches implemented for
  `SingleCells`.
