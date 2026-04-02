# Parametric UMAP

Performs parametric UMAP dimensionality reduction using a neural network
encoder trained on the GPU via wgpu.

## Usage

``` r
parametric_umap(
  data,
  n_dim = 2L,
  k = 15L,
  min_dist = 0.1,
  spread = 1,
  knn_method = c("hnsw", "annoy", "nndescent", "balltree", "exhaustive"),
  nn_params = manifoldsR::params_nn(),
  parametric_umap_params = params_parametric_umap(),
  use_gpu = TRUE,
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- data:

  Numerical matrix or data frame. The data to embed of shape samples x
  features. Will be coerced to a matrix.

- n_dim:

  Integer. Number of embedding dimensions. Defaults to `2L`.

- k:

  Integer. Number of nearest neighbours. Defaults to `15L`.

- min_dist:

  Numeric. Minimum distance between embedded points. Defaults to `0.1`.

- spread:

  Numeric. Effective scale of embedded points. Defaults to `1.0`.

- knn_method:

  Character. Approximate nearest neighbour algorithm. One of `"hnsw"`,
  `"annoy"`, `"nndescent"`, `"balltree"`, or `"exhaustive"`. Defaults to
  `"hnsw"`.

- nn_params:

  Named list. Nearest neighbour parameters, see
  [`manifoldsR::params_nn()`](https://gregorlueg.github.io/manifoldsR/reference/params_nn.html).

- parametric_umap_params:

  Named list. Parametric UMAP parameters, see
  [`params_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_parametric_umap.md).

- use_gpu:

  Boolean. Shall the neural net be trained on GPU via the `wgpu`
  backend. On smaller datasets, the CPU can be faster (via the
  `ndarray`) backend due to kernel launch overhead. data sets, the CPU
  will be faster via the Ndarray.

- seed:

  Integer. Random seed for reproducibility. Defaults to `42L`.

- .verbose:

  Logical. Controls verbosity. Defaults to `TRUE`.

## Value

A `ParametricUmapModel` object containing the embedding matrix and the
trained encoder model.
