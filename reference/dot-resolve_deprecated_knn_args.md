# Fold the deprecated kNN arguments into the current ones

Fold the deprecated kNN arguments into the current ones

## Usage

``` r
.resolve_deprecated_knn_args(
  knn_method,
  nn_params,
  k,
  gpu_method,
  ivf_params,
  dist_metric,
  fn
)
```

## Arguments

- knn_method:

  String. Current argument.

- nn_params:

  List. Current argument.

- k:

  Integer. Current argument.

- gpu_method:

  Deprecated. Superseded by `knn_method`.

- ivf_params:

  Deprecated. Superseded by `nn_params`.

- dist_metric:

  Deprecated. Superseded by `params_nn_gpu(dist_metric)`.

- fn:

  String. Name of the calling function, for the warning text.

## Value

A list with the resolved `knn_method`, `nn_params` and `k`.
