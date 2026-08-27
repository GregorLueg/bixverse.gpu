# Maximum rank the GPU HALS kernels support

The sweep kernels tier their workgroup width by rank and stop at 128,
where the width is already down to one SIMD group on Apple Silicon.

## Usage

``` r
NMF_GPU_MAX_RANK
```
