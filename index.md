# *bixverse.gpu package*

![r_package](https://img.shields.io/badge/R_package-0.1.2-orange)[![CI](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml/badge.svg)](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pkgdown](https://img.shields.io/badge/pkgdown-website-1b5e9f?logo=github)](https://gregorlueg.github.io/bixverse.gpu/)

![bixverse.plots logo](reference/figures/bixverse_gpu_logo.png)

## Intro

GPU-accelerated algorithms (via the WGPU backend on cubecl which allows
you to run the code here on any GPU). You will need to set up your wgpu
however, please check the [CubeCL
book](https://burn.dev/books/cubecl/getting-started/installation.html).
I do not have access to an Nvidia GPU, but I will aim also to allow for
conditional compiling to that backend if Cuda is detected (future
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
Please refer to the [CubeCL book](https://burn.dev/books/cubecl/) in
terms of how to ensure WGPU runs on your respective system. On a Unix
system you also need OpenBLAS set up for the Ndarray backend; on MacOS
it will just use the Accelerate framework.

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
with GPU-accelerated methods (at the moment with focus on single cell
support). If you are however interesting in just using the
GPU-accelerated kNN searches, feel free to use the respective `rs_`
functions for that. Or if you want to train a neural network for UMAP,
the package also provides what you need.

### Roadmap:

Currently, the main focus was on acceleration of kNN graph generation,
which is quite ubiquitous in single cell. Other areas of (personal)
interest could be GPU-accelerated k-means (Harmony could benefit here)
or specific sparse matrix multiplications on the GPU for large data
sets.
