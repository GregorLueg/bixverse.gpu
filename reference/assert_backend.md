# Assert that a device backend is usable

The `backend` counterpart of
[`assert_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/assert_gpu.md):
`"wgpu"` needs a WGPU adapter, `"mlx"` needs a build with the MLX
backend.

## Usage

``` r
assert_backend(backend)
```

## Arguments

- backend:

  String. One of `c("wgpu", "mlx")`.

## Value

Invisibly `TRUE`. Called for the error.
