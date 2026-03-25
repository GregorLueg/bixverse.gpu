# *bixverse.gpu package*

![r_package](https://img.shields.io/badge/R_package-0.1.0-orange) 
[![CI](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml/badge.svg)](https://github.com/GregorLueg/bixverse.gpu/actions/workflows/R-cmd-check.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pkgdown](https://img.shields.io/badge/pkgdown-website-1b5e9f?logo=github)](https://gregorlueg.github.io/bixverse.gpu/)

</br>

<img src="man/figures/bixverse_gpu_logo.png" width="128" height="128" alt="bixverse.plots logo">

</br>

## Intro

GPU-accelerated algorithms (via the WGPU backend on cubecl which allows you
to run the code here on any GPU). You will need to set up your wgpu however, 
please check the [CubeCL book](https://burn.dev/books/cubecl/getting-started/installation.html).
I do not have access to an Nvidia GPU, but I will aim also to allow for 
conditional compiling to that backend if Cuda is detected (future problem).
The package is designed to support the 
[bixverse package](https://github.com/GregorLueg/bixverse). 

## Usage

### Installation

You will need Rust on your system to have the package working. An installation
guide is provided [here](https://www.rust-lang.org/tools/install). There is a
bunch of further help written [here](https://extendr.github.io/rextendr/index.html)
by the rextendr guys in terms of Rust set up. (bixverse.gpu as bixverse both 
use rextendr to interface with Rust.) Additionally, in this special case, you
will also need the GPU

Steps for installation:

1. In the terminal, install [Rust](https://www.rust-lang.org/tools/install)

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. In R, install [rextendr](https://extendr.github.io/rextendr/index.html):

```
install.packages("rextendr")
```

3. Finally install bixverse:

```
devtools::install_github("https://github.com/GregorLueg/bixverse.gpu")
```

If you are using Windows, I am sorry, the tool chain is too much of a pain. The
package will not work there.

### How to use the package.

The package website can be found 
[here](https://gregorlueg.github.io/bixverse.gpu/). This package is **not** a
stand-alone package, but designed to support the `bixverse` with GPU-accelerated
methods (at the moment with focus on single cell support).
