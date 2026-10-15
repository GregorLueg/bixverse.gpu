# Check whether the MLX backend was compiled in

The MLX backend (Apple Silicon only, via mlx-c) is opt-in at install
time: set `BIXVERSE_GPU_MLX=true` and have mlx-c installed
(`brew install mlx-c`, or `MLX_C_PREFIX` pointing at an install prefix).

## Usage

``` r
rs_mlx_available()
```

## Value

Boolean. `TRUE` when the package was built with the MLX backend.
