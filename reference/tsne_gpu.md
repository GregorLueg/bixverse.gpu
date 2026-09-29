# Rust-based t-SNE (GPU)

Performs t-SNE dimensionality reduction on the input data. The kNN
search runs on the GPU. With the default `"fft_3k_gpu"` the optimiser
runs on the GPU as well, so the whole embedding stays on the device; the
other approximations optimise on the CPU.

## Usage

``` r
tsne_gpu(
  data,
  knn = NULL,
  n_dim = 2L,
  perplexity = 20,
  approx_type = c("fft_3k_gpu", "bh", "fft", "fft_3k"),
  knn_method = c("ivf", "exhaustive", "nndescent"),
  nn_params = params_nn_gpu(),
  tsne_params = params_tsne_gpu(),
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

  Optional `NearestNeighbours` class. If provided, t-SNE will skip the
  k-nearest neighbour graph generation and use this one. Defaults to
  `NULL`. See
  [`manifoldsR::new_nearest_neighbour()`](https://gregorlueg.github.io/manifoldsR/reference/new_nearest_neighbour.html)
  for details.

- n_dim:

  Integer. Number of dimensions in the embedding space. Currently only
  `2L` is supported. Defaults to `2L`.

- perplexity:

  Numeric. Perplexity parameter, related to the number of nearest
  neighbours used in manifold learning. Typical values are between 5
  and 50. Defaults to `20.0`.

- approx_type:

  Character. Approximation method for computing repulsive forces. One of
  `"fft_3k_gpu"` (three-kernel FFT interpolation on the GPU, the
  default), `"bh"` for Barnes-Hut, `"fft"` for FFT-accelerated
  interpolation or `"fft_3k"` for its three-kernel variant (one forward
  and three inverse FFTs per epoch instead of four each). The last three
  run on the CPU; `"fft"` and `"fft_3k"` are only available on Unix
  systems.

- knn_method:

  Character. GPU-accelerated (approximate) nearest neighbour method to
  use. One of `"nndescent"`, `"exhaustive"`, or `"ivf"`.

- nn_params:

  Named list. Nearest neighbour search parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).
  tSNE uses higher k usually, hence, \`"ivf"“ is the default here.

- tsne_params:

  Named list. t-SNE (GPU) algorithm parameters, see
  [`params_tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_tsne_gpu.md).

- seed:

  Integer. Random seed for reproducibility. Defaults to `42L`.

- use_high_precision:

  Optional boolean. Gives fine-grained control over `fp32` vs `fp64`
  usage. The GPU kNN calculations will be forced into `fp32`. Ignored
  with a warning for `"fft_3k_gpu"`, which always runs in `fp32`.

- .verbose:

  Logical. Controls verbosity. Defaults to `TRUE`.

## Value

A numerical matrix with dimensions samples x n_dim containing the t-SNE
embedding.

## Details

The number of neighbours is derived from `perplexity` on the Rust side
following the usual `3 * perplexity` convention.
