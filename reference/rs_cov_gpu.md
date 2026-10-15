# GPU-accelerated covariance calculations

**\[experimental\]** GPU-accelerated pairwise column co-variance
calculation.

## Usage

``` r
rs_cov_gpu(x, backend = "wgpu", verbose)
```

## Arguments

- x:

  Numerical matrix. The matrix for which to calculate the column
  pairwise covariance matrix.

- backend:

  String. `"wgpu"` or `"mlx"` (Apple Silicon, needs a build with the MLX
  backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).

- verbose:

  Boolean. Controls verbosity of the function.

## Value

The covariance matrix
