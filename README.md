# *bixverse.gpu package*

![r_package](https://img.shields.io/badge/R_package-0.0.1.0-orange) 
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

