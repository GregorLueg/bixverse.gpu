# Generate a k-nearest neighbour graph (GPU-accelerated)

This function generates a kNN graph based on a given numeric matrix.
Four different GPU-accelerated versions are available

- `"nndescent"` - The NN-descent kNN graph itself, no index and no
  search. Rows the descent left short are topped up with an exact
  search.

- `"cagra"` - The NN-descent graph pruned into a CAGRA graph and beam
  searched.

- `"exhaustive"` - Exact nearest neighbour search via GPU.

- `"ivf"` - Inverted file index that clusters the data with k-means and
  probes a few of the clusters at query time.

## Usage

``` r
generate_knn_graph_gpu(
  data,
  k,
  knn_method = c("nndescent", "cagra", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  seed = 42L,
  extract_knn = lifecycle::deprecated(),
  backend = c("wgpu", "mlx"),
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
  `c("nndescent", "cagra", "exhaustive", "ivf")`. Defaults to
  `"nndescent"`.

- nn_params:

  List. Output of
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- seed:

  Integer. For reproducibility. Defaults to `42L`.

- extract_knn:

  **\[deprecated\]** Use `knn_method = "nndescent"` (graph only) or
  `knn_method = "cagra"` (beam search) instead.

- backend:

  String. One of `c("wgpu", "mlx")`. `"mlx"` runs the search on Apple
  Silicon via MLX and needs a build with the MLX backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md).

- .verbose:

  Boolean. Controls verbosity.

## Value

A nearest neighbours class object with 1-indexed neighbour indices and
distances. Euclidean distances are true L2, not squared.
