# GPU-accelerated correlation calculations

**\[experimental\]** GPU-accelerated pairwise column correlations. Has
the options of Pearson and Spearman correlation coefficient
calculations.

## Usage

``` r
rs_cor_gpu(x, spearman, backend = "wgpu", verbose)
```

## Arguments

- x:

  Numerical matrix. The matrix for which to calculate the column
  pairwise correlation matrix.

- spearman:

  Boolean. Shall the Spearman correlation be calculated instead of
  Pearson.

- backend:

  String. `"wgpu"` or `"mlx"` (Apple Silicon, needs a build with the MLX
  backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).

- verbose:

  Boolean. Controls verbosity of the function.

## Value

The correlation matrix
