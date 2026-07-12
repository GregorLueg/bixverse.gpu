# Default parameters for GPU k-means

Default parameters for GPU k-means

## Usage

``` r
params_kmeans_gpu(
  k_means_iter = 50L,
  k_means_init = NULL,
  metric = c("euclidean", "cosine"),
  fixed = FALSE,
  quantise = FALSE
)
```

## Arguments

- k_means_iter:

  Integer. Number of k-means iterations.

- k_means_init:

  Optional character. Initialisation method. One of `"random"`,
  `"parallel"`, or `"plusplus"`. If `NULL`, determined on the Rust side.

- metric:

  String. One of `c("euclidean", "cosine")`.

- fixed:

  Logical. Shall the algorithm be run for a fixed number of iterations,
  without checking for convergence.

- quantise:

  Logical. Whether to quantise data to `fp16` before clustering. This
  can improve performance in circumstances where it is memory bound.

## Value

A list with the parameters.
