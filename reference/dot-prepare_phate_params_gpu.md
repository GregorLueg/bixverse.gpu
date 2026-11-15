# Internal helper to prepare the PHATE parameters (GPU version)

Internal helper to prepare the PHATE parameters (GPU version)

## Usage

``` r
.prepare_phate_params_gpu(knn_method, nn_params, phate_params)
```

## Arguments

- knn_method:

  String. Method to use to generate the kNN graph.

- nn_params:

  Named list. The nearest neighbour search parameters (GPU).

- phate_params:

  Named list. The PHATE-specific parameters (GPU).

## Value

Returns the list of final parameters.
