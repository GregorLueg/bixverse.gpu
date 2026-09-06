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

## Intro

GPU-accelerated algorithms via the WGPU backend on cubecl. All you need
is a GPU that WGPU can talk to: Metal on macOS, Vulkan on Linux, DX12 or
Vulkan on Windows. No CUDA, no vendor lock-in, no separate GPU toolchain
to install. Check with
[`gpu_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/gpu_available.md)
after installing; if that returns `TRUE`, everything in the package will
run. If it returns `FALSE`, your drivers are the problem, and the
[cubecl
book](https://burn.dev/books/cubecl/getting-started/installation.html)
covers the set up per platform.

The package is designed to support the [bixverse
package](https://github.com/GregorLueg/bixverse). Additionally, also
provides some neural net-based versions of embedding methods for
[manifoldsR](https://github.com/GregorLueg/manifoldsR) + a
GPU-accelerated version of the Adam optimiser for UMAP.

Heads up: the package is being refactored at the moment, so the R-facing
API can shift between versions. `lifecycle: experimental` covers the
whole surface and means it.

## Usage

### Installation

On the GPU side there is nothing extra to install beyond working
drivers, whatever your OS ships is what WGPU picks up. Previously, the
CPU-based versions of neural net acceleration where running through
ndarray and accelerated via OpenBLAS (Linux) or Accelerate (Mac). This
has been now replaced with the
[flex](https://github.com/tracel-ai/burn/pull/4761) framework.

The easy route is r-universe. You get a pre-built binary, so no Rust
toolchain and no compile:

``` r

install.packages(
  "bixverse.gpu",
  repos = c("https://gregorlueg.r-universe.dev", "https://cloud.r-project.org")
)
```

### From source

You will need Rust on your system to install the package from source. An
installation guide is provided
[here](https://www.rust-lang.org/tools/install). There is a bunch of
further help written
[here](https://extendr.github.io/rextendr/index.html) by the rextendr
guys in terms of Rust set up. (`bixverse.gpu` as `bixverse` both use
rextendr to interface with Rust.)

#### Setting up Rust

Steps for installation:

1.  In the terminal, install
    [Rust](https://www.rust-lang.org/tools/install)

&nbsp;

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

2.  In R, install
    [rextendr](https://extendr.github.io/rextendr/index.html):

&nbsp;

    install.packages("rextendr")

3.  Finally install bixverse.gpu:

&nbsp;

    devtools::install_github("https://github.com/GregorLueg/bixverse.gpu")

#### Windows support

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

### How to use the package.

The package website can be found
[here](https://gregorlueg.github.io/bixverse.gpu/). This package is
**not** a stand-alone package, but designed to support the `bixverse`
with GPU-accelerated methods. If you are however interesting in just
using the GPU-accelerated kNN searches, feel free to use the respective
`rs_` functions for that. Or if you want to train a neural network for
UMAP, the package also provides what you need.

### Roadmap:

Current roadmap (subject to change and interest):

**GPU-related things:**

GPU-based kNN graph generation (for single cells)

k-means clustering on GPU

Sparse, randomised SVD for single cells

GPU-accelerated Harmony batch correction (single cells)

GPU-accelerated correlations (Spearman and Pearson)

GPU-accelerate UMAP embedding generation.

SCENIC has now working GPU acceleration for ExtraTrees and the RF
learner. The grnboost2 version does not make too much sense and is left
on the CPU.

GPU-accelerated SEACells meta cell generation (both Frank-Wolfe solves).

Scrublet with GPU-acceleration.

GPU-accelerated NMF.

GPU-accelerated BBKNN batch correction (single cells)

**General:**

~~More vignettes on some of the implemented functions.~~ Got a bit
better.

If you have some other ideas, please feel free to make an issue.
