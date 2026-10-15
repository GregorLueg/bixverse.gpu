# Is the MLX backend available

Reports whether this build carries the MLX backend. MLX is Apple Silicon
only and opt-in at install time, since mlx-c is a system library the
package cannot ship: install mlx-c (`brew install mlx-c`, or point
`MLX_C_PREFIX` at an install prefix) and reinstall with
`BIXVERSE_GPU_MLX=true`. Functions with an MLX path take
`backend = "mlx"`.

## Usage

``` r
mlx_available()
```

## Value

Boolean. `TRUE` when the package was built with the MLX backend.

## Examples

``` r
if (mlx_available()) {
  # ... backend = "mlx"
}
```
