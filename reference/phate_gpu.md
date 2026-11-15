# Rust-based PHATE (GPU)

Performs PHATE dimensionality reduction on the input data. The kNN
search runs on the GPU, and so do the randomised SVD and k-means of the
default spectral landmark selection. The diffusion time, the powering of
the operator, MDS and the landmark interpolation stay on the CPU.

## Usage

``` r
phate_gpu(
  data,
  knn = NULL,
  n_dim = 2L,
  k = 5L,
  knn_method = c("ivf", "exhaustive", "nndescent"),
  nn_params = params_nn_gpu(),
  phate_params = params_phate_gpu(),
  seed = 42L,
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
)
```

## Arguments

- data:

  Numerical matrix or data frame. The data to embed of shape samples x
  features. Will be coerced to a matrix.

- knn:

  Optional `NearestNeighbours` class. If provided, PHATE will skip the
  k-nearest neighbour graph generation and use this one. Should be
  `6 * k + 5` wide; a narrower graph gives a warning. Defaults to
  `NULL`. See
  [`manifoldsR::new_nearest_neighbour()`](https://gregorlueg.github.io/manifoldsR/reference/new_nearest_neighbour.html)
  for details.

- n_dim:

  Integer. Number of dimensions in the embedding space. Defaults to
  `2L`.

- k:

  Integer. Number of nearest neighbours that set the kernel bandwidth.
  Defaults to `5L`.

- knn_method:

  Character. GPU-accelerated (approximate) nearest neighbour method to
  use. One of `"ivf"`, `"exhaustive"` or `"nndescent"`. Defaults to
  `"ivf"`: the `6 * k + 5` wide search is past the width where
  NN-descent keeps up.

- nn_params:

  Named list. Nearest neighbour search parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- phate_params:

  Named list. PHATE (GPU) algorithm parameters, see
  [`params_phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_phate_gpu.md).

- seed:

  Integer. Random seed for reproducibility. Defaults to `42L`.

- backend:

  String. One of `c("wgpu", "mlx")`. `"mlx"` runs the kNN search and the
  spectral landmark selection on MLX (Apple Silicon, needs a build with
  the MLX backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).

- .verbose:

  Logical. Controls verbosity. Defaults to `TRUE`.

## Value

A numerical matrix with dimensions samples x n_dim containing the PHATE
embedding.

## Details

The kNN search runs `6 * k + 5` wide: neighbours past `k` join the
kernel while their affinity clears `thresh` (see
[`params_phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_phate_gpu.md)).
A precomputed graph should be that wide as well.

## References

Moon, et al., Nat. Biotechnol., 2019
