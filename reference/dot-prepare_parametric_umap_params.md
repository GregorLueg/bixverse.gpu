# Internal helper to prepare parametric UMAP parameters

Internal helper to prepare parametric UMAP parameters

## Usage

``` r
.prepare_parametric_umap_params(
  min_dist,
  spread,
  knn_method,
  nn_params,
  parametric_umap_params
)
```

## Arguments

- min_dist:

  Numeric. Minimum distance between embedded points.

- spread:

  Numeric. Effective scale of embedded points.

- knn_method:

  String. Approximate nearest neighbour method.

- nn_params:

  Named list. Nearest neighbour parameters, see
  [`manifoldsR::params_nn()`](https://gregorlueg.github.io/manifoldsR/reference/params_nn.html).

- parametric_umap_params:

  Named list. Parametric UMAP parameters.

## Value

Returns the merged list of final parameters.
