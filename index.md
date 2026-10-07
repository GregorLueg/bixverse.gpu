# *bixverse.gpu package*

[![r_package](https://img.shields.io/github/r-package/v/GregorLueg/bixverse.gpu?label=R_package&color=orange)](https://github.com/GregorLueg/bixverse.gpu/blob/main/DESCRIPTION)
[![bixverse status
badge](https://gregorlueg.r-universe.dev/bixverse.gpu/badges/version)](https://gregorlueg.r-universe.dev/bixverse.gpu)
[![CI](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml/badge.svg)](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pkgdown](https://img.shields.io/badge/pkgdown-website-1b5e9f?logo=github)](https://gregorlueg.github.io/bixverse.gpu/)
[![extendr](https://img.shields.io/badge/extendr-%5E0.9.0-276DC2)](https://extendr.github.io/extendr/extendr_api/)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

## What this is

GPU-accelerated methods for
[bixverse](https://github.com/GregorLueg/bixverse) and
[manifoldsR](https://github.com/GregorLueg/manifoldsR), written in Rust
on
[cubecl](https://github.com/tracel-ai/cubecl)/[Burn](https://burn.dev)
with the WGPU backend. All you need is a GPU that WGPU can talk to:
Metal on macOS, Vulkan on Linux, DX12 or Vulkan on Windows. No CUDA, no
vendor lock-in, no separate GPU toolchain to install.

Check with
[`gpu_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/gpu_available.md)
after installing. If that returns `TRUE`, you’re good to go. If it
returns `FALSE`, your drivers are the problem, and the [cubecl
book](https://burn.dev/books/cubecl/getting-started/installation.html)
covers the set up per platform.

This is **not** a stand-alone package. Most functions take a `bixverse`
`SingleCells` or `MetaCells` object and slot into the same workflow as
their CPU counterparts. If you only want the GPU kNN searches or a
parametric UMAP, the `rs_` functions and
[`parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/parametric_umap.md)
work on plain matrices.

Heads up: the R-facing API can still shift between versions.
`lifecycle: experimental` covers the whole surface and means it.

## What’s in it

| Domain | What’s in there | Read more |
|----|----|----|
| Single cell core | Sparse randomised PCA, kNN graphs (CAGRA, IVF, exhaustive), Harmony, fastMNN, BBKNN, fast clustering | [single cell](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_single_cell.html) |
| Meta cells | SEACells, both Frank-Wolfe solves on the GPU | [SEACells](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_metacells.html) |
| Regulons | SCENIC with ExtraTrees and random forest learners on the GPU. grnboost2 stays on the CPU, it does not gain much | [SCENIC](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_scenic.html) |
| QC and DGE | Scrublet doublet detection, NEBULA mixed models | [Scrublet](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_scrublet.html), [NEBULA](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_nebula.html) |
| Factorisation | NMF, stabilised and consensus NMF, k sweeps | [NMF](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_nmf.html) |
| Trees | Bonsai, with Sanity running on the GPU; tree search and layout stay on the CPU | [`?bonsai_gpu_sc`](https://gregorlueg.github.io/bixverse.gpu/reference/bonsai_gpu_sc.md) |
| Embeddings | UMAP, tSNE (Barnes-Hut and FFT), parametric UMAP with GPU and CPU backends | [UMAP](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_umap.html), [tSNE](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_tsne.html), [parametric UMAP](https://gregorlueg.github.io/bixverse.gpu/articles/parametric_umap.html) |
| General | k-means, Pearson and Spearman correlations, covariance | [other methods](https://gregorlueg.github.io/bixverse.gpu/articles/other_gpu.html) |

## The “bixverse ecosystem”

- [bixverse](https://github.com/GregorLueg/bixverse) is the parent
  package. Single cell, bulk, gene set enrichment, ontologies, graphs.
  Everything here plugs into its classes.
- [bixverse.plots](https://github.com/GregorLueg/bixverse.plots) for
  plotting, with a large number of single cell helpers.
- [manifoldsR](https://github.com/GregorLueg/manifoldsR) for manifold
  learning. The GPU UMAP, tSNE and parametric UMAP here sit next to its
  CPU versions.

## Installation

On the GPU side there is nothing extra to install beyond working
drivers; whatever your OS ships is what WGPU picks up. The CPU path of
the neural net methods used to go through ndarray with OpenBLAS (Linux)
or Accelerate (Mac). It now runs on Burn’s
[flex](https://github.com/tracel-ai/burn/pull/4761) backend, so no BLAS
set up either.

The easy route is r-universe. You get a pre-built binary, so no Rust
toolchain and no compile:

``` r

install.packages(
  "bixverse.gpu",
  repos = c("https://gregorlueg.r-universe.dev", "https://cloud.r-project.org")
)
```

### From source

Building from source needs Rust on your system. Install guide
[here](https://www.rust-lang.org/tools/install), and the rextendr guys
have written a lot of further help on the Rust set up
[here](https://extendr.github.io/rextendr/index.html).

1.  In the terminal, install
    [Rust](https://www.rust-lang.org/tools/install)

&nbsp;

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

2.  In R, install
    [rextendr](https://extendr.github.io/rextendr/index.html):

&nbsp;

    install.packages("rextendr")

3.  Install bixverse.gpu. Keep the r-universe repo in the list:
    `bixverse` and `manifoldsR` live there, not on CRAN.

``` r

options(repos = c("https://gregorlueg.r-universe.dev", getOption("repos")))
devtools::install_github("https://github.com/GregorLueg/bixverse.gpu")
```

### Windows

Windows works. WGPU was never the problem there, DX12 and Vulkan are
both well covered, and the h5 dependency (for reading h5ad files) turned
out to be a dull `MAX_PATH` issue rather than a cross-compile one:
`R CMD INSTALL` builds in a deep temp directory, and the HDF5 CMake
build pushed object paths past the 260 character limit. The build now
puts the cargo target directory in `~/.bixverse-gpu-cargo`, which stays
clear of it. Same fix as in
[bixverse](https://github.com/GregorLueg/bixverse).

One thing is still missing on Windows: the FFT-accelerated tSNE. FFTW
does not come along for the ride, so `tsne_gpu(approx_type = "fft")`
errors there. Barnes-Hut (`approx_type = "bh"`, the default) works
everywhere.

### Older Intel Macs

Likely not a good time. A Mac without a native GPU that Metal can drive
properly, such as a macOS VM with a paravirtualised GPU, may advertise
subgroup (plane) operations and then silently drop every kernel that
uses them. The macos-15-intel GitHub runners do exactly that. NEBULA and
Bonsai check for this and refuse to run; other methods pass CI on those
runners, but treat results on such a machine with suspicion and compare
against the CPU version in `bixverse`.

## Where to start

The [package website](https://gregorlueg.github.io/bixverse.gpu/) is the
main entry point. The [single cell
article](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_single_cell.html)
is the obvious first stop: it runs the standard workflow with the GPU
versions swapped in. What changed in each release: [the
changelog](https://gregorlueg.github.io/bixverse.gpu/news/index.html).

Working with an LLM coding agent?
[`install_agent_skill_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/install_agent_skill_gpu.md)
drops a skill into your agent set up so it stops guessing at the API.

## Roadmap

**Implemented**

GPU-based kNN graph generation (for single cells)

k-means clustering on GPU

Sparse, randomised SVD for single cells

GPU-accelerated Harmony, fastMNN and BBKNN batch correction (single
cells)

GPU-accelerated correlations (Spearman and Pearson)

GPU-accelerated UMAP and tSNE embedding generation

SCENIC with GPU-accelerated ExtraTrees and random forest learners

GPU-accelerated SEACells meta cell generation

Scrublet with GPU acceleration

GPU-accelerated NMF

GPU-accelerated NEBULA

Sanity processing with GPU acceleration before Bonsai

**General**

~~More vignettes on some of the implemented functions.~~ Got a bit
better.

If you have some other ideas, please feel free to open an issue.

*Last update to the read-me: 02.10.2026*
