# GPU-accelerated correlation calculations

**\[experimental\]** GPU-accelerated pairwise column correlations. Has
the options of Pearson and Spearman correlation coefficient
calculations.

## Usage

``` r
rs_cor_gpu(x, spearman, verbose)
```

## Arguments

- x:

  Numerical matrix. The matrix for which to calculate the column
  pairwise correlation matrix.

- spearman:

  Boolean. Shall the Spearman correlation be calculated instead of
  Pearson.

- verbose:

  Boolean. Controls verbosity of the function.

## Value

The correlation matrix
