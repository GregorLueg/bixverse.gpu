# Generate a k-nearest neighbour graph (GPU-accelerated)

This function generates a kNN graph based on a given numeric matrix.
Three different GPU-accelerated versions are available

- `"exhaustive"` - Exact nearest neighbour search via GPU.

- `"ivf"` - Inverted file index that leverages k-means clustering and
  probing a few of the clusters via GPU-accelerated distance
  calculations.

- `"nndescent"` - A CAGRA style nearest neighbour search on the GPU.

## Usage

``` r
generate_knn_graph_gpu(
  data,
  k,
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  seed = 42L,
  extract_knn = lifecycle::deprecated(),
  .verbose = TRUE
)
```

## Arguments

- data:

  Numeric matrix. The embedding or feature matrix to compute neighbours
  on. Rows are observations, columns are features.

- k:

  Integer. The number of nearest neighbours to compute.

- knn_method:

  Character. The algorithm to use for nearest neighbour search. One of
  `c("exhaustive", "ivf", "nndescent")`. Defaults to `"nndescent"`

- nn_params:

  List. Output of
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- seed:

  Integer. For reproducibility. Defaults to `42L`.

- extract_knn:

  **\[deprecated\]** Use the `extract_knn` field of
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md)
  instead.

- .verbose:

  Boolean. Controls verbosity.

## Value

A nearest neighbours class object with 1-indexed neighbour indices and
distances. Euclidean distances are true L2, not squared.
