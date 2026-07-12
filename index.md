# *bixverse.gpu package*

![r_package](https://img.shields.io/badge/R_package-0.2.1-orange)[![CI](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml/badge.svg)](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pkgdown](https://img.shields.io/badge/pkgdown-website-1b5e9f?logo=github)](https://gregorlueg.github.io/bixverse.gpu/)
[![extendr](https://img.shields.io/badge/extendr-%5E0.9.0-276DC2)](https://extendr.github.io/extendr/extendr_api/)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

![bixverse.plots logo](reference/figures/bixverse_gpu_logo.png)

## Intro

GPU-accelerated algorithms (via the WGPU backend on cubecl which allows
you to run the code here on any GPU). You will need to set up your wgpu
however, please check the [cubecl
book](https://burn.dev/books/cubecl/getting-started/installation.html).
I do not have access to an Nvidia GPU, but I will aim also to allow for
conditional compiling to that backend if cuda is detected (future
problem). The package is designed to support the [bixverse
package](https://github.com/GregorLueg/bixverse). Additionally, also
provides some neural net-based versions of embedding methods for
[manifoldsR](https://github.com/GregorLueg/manifoldsR).

## Usage

### Installation

You will need Rust on your system to have the package working. An
installation guide is provided
[here](https://www.rust-lang.org/tools/install). There is a bunch of
further help written
[here](https://extendr.github.io/rextendr/index.html) by the rextendr
guys in terms of Rust set up. (bixverse.gpu as bixverse both use
rextendr to interface with Rust.) Additionally, in this special case,
you will also need the GPU drivers set up properly on your system.
Please refer to the [cubecl book](https://burn.dev/books/cubecl/) in
terms of how to ensure wgpu runs on your respective system. Previously,
the CPU-based versions of neural net acceleration where running through
ndarray and accelerated via OpenBLAS (Linux) or Accelerate (Mac). This
has been now replaced with the
[flex](https://github.com/tracel-ai/burn/pull/4761) framework.

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

If you are using Windows, I am sorry, the tool chain is just very, very
painful… I really tried to make this work and maybe there are some hacks
in terms of compiling everything to make this work, but I cannot
guarantee proper behaviour here due to the dependency with h5 (for
reading in h5ad files). If you know how to make this work without
several hacks in an easy way, please contact me!

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

GPU-accelerated NMF.

GPU-accelerated BBKNN batch correction (single cells)

GPU-accelerate UMAP embedding generation.

**General:**

~~More vignettes on some of the implemented functions.~~ Got a bit
better.

If you have some other ideas, please feel free to make an issue.
