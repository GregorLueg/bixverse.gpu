# Internal helper to prepare the t-SNE parameters (GPU version)

Internal helper to prepare the t-SNE parameters (GPU version)

## Usage

``` r
.prepare_tsne_params_gpu(knn_method, nn_params, tsne_params)
```

## Arguments

- knn_method:

  String. Method to use to generate the kNN graph.

- nn_params:

  Named list. The nearest neighbour search parameters (GPU).

- tsne_params:

  Named list. The t-SNE-specific parameters (GPU).

## Value

Returns the list of final parameters.
