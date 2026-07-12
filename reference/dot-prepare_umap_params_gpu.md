# Internal helper to prepare the UMAP parameters (GPU version)

Internal helper to prepare the UMAP parameters (GPU version)

## Usage

``` r
.prepare_umap_params_gpu(
  n,
  min_dist,
  spread,
  knn_method,
  nn_params,
  umap_params,
  .verbose = TRUE
)
```

## Arguments

- n:

  Integer. Number of samples in the data set

- min_dist:

  Numeric. Minimum distance between embedded points.

- spread:

  Numeric. Effective scale of embedded points.

- knn_method:

  String. Method to use to generate the kNN graph.

- nn_params:

  Named list. The nearest neighbour search parameters (GPU).

- umap_params:

  Named list. The UMAP-specific parameters (GPU).

- .verbose:

  Boolean. Controls verbosity

## Value

Returns the list of final parameters.
