# Fold the deprecated fast clustering arguments into the current ones

Fold the deprecated fast clustering arguments into the current ones

## Usage

``` r
.resolve_deprecated_fc_args(
  fc_params,
  n_seeds,
  n_centroids,
  snn,
  grid_search,
  no_seeds
)
```

## Arguments

- fc_params:

  List. Current argument.

- n_seeds:

  Integer. Current argument.

- n_centroids:

  Deprecated. Superseded by `fc_params$n_centroids`.

- snn:

  Deprecated. Superseded by `fc_params$knn_sym`.

- grid_search:

  Deprecated. Superseded by `n_seeds`.

- no_seeds:

  Deprecated. Superseded by `n_seeds`.

## Value

A list with the resolved `fc_params` and `n_seeds`.
