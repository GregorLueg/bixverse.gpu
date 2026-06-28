# Default parameters for GPU k-means

Default parameters for GPU k-means

## Usage

``` r
params_kmeans_gpu(
  k_means_iter = 50L,
  k_means_init = NULL,
  metric = c("euclidean", "cosine"),
  fixed = TRUE,
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

  Logical. Whether cluster centres are fixed after initialisation.

- quantise:

  Logical. Whether to quantise data to f16 before clustering.

## Value

A list with the parameters.
