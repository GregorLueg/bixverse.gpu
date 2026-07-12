# Rust-based UMAP (GPU)

Performs UMAP dimensionality reduction on the input data. This function
provides a user-friendly interface with input validation before calling
the Rust implementation. Leverages GPU-accelerated kNN searches and in
the default setting also uses a GPU-accelerated Adam optimiser for the
embedding.

## Usage

``` r
umap_gpu(
  data,
  knn = NULL,
  n_dim = 2L,
  k = 15L,
  min_dist = 0.5,
  spread = 1,
  knn_method = c("nndescent", "exhaustive", "ivf"),
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
)
```

## Arguments

- data:

  Numerical matrix or data frame. The data to embed of shape samples x
  features. Will be coerced to a matrix.

- knn:

  Optional `NearestNeighbours` class. If provided, UMAP will skip the
  k-nearest neighbour graph generation and use this one. Defaults to
  `NULL`. See
  [`manifoldsR::new_nearest_neighbour()`](https://gregorlueg.github.io/manifoldsR/reference/new_nearest_neighbour.html)
  for details.

- n_dim:

  Integer. Number of dimensions in the embedding space. Defaults to
  `2L`.

- k:

  Integer. Number of nearest neighbours to consider for manifold
  approximation. Larger values result in more global structure being
  preserved. Defaults to `15L`.

- min_dist:

  Numeric. Minimum distance between points in the embedding. Controls
  how tightly points are packed. Smaller values result in more clustered
  embeddings. Must be \>= 0. Defaults to `0.5`. If you use SGD, consider
  reducing this!

- spread:

  Numeric. Effective scale of embedded points. Determines the scale at
  which embedded points will be spread out. Defaults to `1.0`.

- knn_method:

  Character. (Approximate) Nearest neighbour method to use. One of
  `"exhaustive"`, `"ivf"` or `"nndescent_gpu"`. These are
  GPU-accelerated methods.

- umap_params:

  Named list. UMAP (GPU) algorithm parameters, see
  [`params_umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_umap_gpu.md).

- seed:

  Integer. Random seed for reproducibility. Defaults to `42L`.

- use_high_precision:

  Optional boolean. Gives fine-grained control over `fp32` vs `fp64`
  usage. The GPU calculations will be forced into `fp32`.

- .verbose:

  Logical. Controls verbosity. Defaults to `TRUE`.

- nn_params_gpu:

  Named list. Nearest neighbour search parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

## Value

A numerical matrix with dimensions samples x n_dim containing the UMAP
embedding.
