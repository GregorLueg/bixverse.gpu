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

  Integer. Number of k-means iterations. Defaults to `50L`.

- k_means_init:

  String or `NULL`. Initialisation method. One of `"random"`,
  `"parallel"`, or `"plusplus"`. If `NULL`, determined on the Rust side.
  Defaults to `NULL`.

- metric:

  String. The distance metric. One of `c("euclidean", "cosine")`.
  Defaults to `"euclidean"`.

- fixed:

  Boolean. Shall the algorithm be run for a fixed number of iterations,
  without checking for convergence. Defaults to `FALSE`.

- quantise:

  Boolean. Whether to quantise data to `fp16` before clustering. This
  can improve performance in circumstances where it is memory bound.
  Defaults to `FALSE`.

## Value

A named list with the following elements:

- k_means_iter - Integer. Number of k-means iterations. Defaults to
  `50L`.

- k_means_init - String or `NULL`. Initialisation method. One of
  `"random"`, `"parallel"`, or `"plusplus"`. If `NULL`, determined on
  the Rust side. Defaults to `NULL`.

- metric - String. The distance metric. One of
  `c("euclidean", "cosine")`. Defaults to `"euclidean"`.

- fixed - Boolean. Shall the algorithm be run for a fixed number of
  iterations, without checking for convergence. Defaults to `FALSE`.

- quantise - Boolean. Whether to quantise data to `fp16` before
  clustering. This can improve performance in circumstances where it is
  memory bound. Defaults to `FALSE`.
