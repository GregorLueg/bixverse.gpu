# PHATE implementation from a pre-computed kNN graph

**\[experimental\]** Wraps the GPU PHATE in manifolds-rs. This version
takes a pre-computed kNN graph, please see
[`new_nearest_neighbour()`](https://gregorlueg.github.io/manifoldsR/reference/new_nearest_neighbour.html).
Neighbours past `k` join the kernel while their affinity clears
`thresh`, so the graph should be `6 * k + 5` wide. Spectral landmark
selection still runs on the GPU.

## Usage

``` r
rs_phate_from_knn_gpu(
  embd,
  knn_data,
  n_dim,
  k,
  phate_params,
  seed,
  backend = "wgpu",
  verbose
)
```

## Arguments

- embd:

  Numerical matrix. The data to use to generate the embeddings. Should
  be of dimensions samples x features.

- knn_data:

  `NearestNeighbours` class from R.

- n_dim:

  Integer. Number of PHATE dimensions to return.

- k:

  Integer. Number of neighbours that set the kernel bandwidth.

- phate_params:

  Named list. The merged GPU kNN and PHATE parameters plus `knn_method`.
  Missing keys fall back to the manifolds-rs defaults.

- seed:

  Integer. Seed for reproducibility.

- backend:

  String. `"wgpu"` or `"mlx"`. `"mlx"` runs the spectral landmark
  selection on MLX (Apple Silicon, needs a build with the MLX backend,
  see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).

- verbose:

  Integer. If `0L` -\> silent or `1L` for normal verbosity; `2L` for
  detailed verbosity.

## Value

The PHATE embeddings.
