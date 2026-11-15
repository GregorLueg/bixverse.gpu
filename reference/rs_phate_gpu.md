# PHATE implementation

**\[experimental\]** Wraps the GPU PHATE in manifolds-rs. The kNN search
(`6 * k + 5` wide) and, for spectral landmarks, the randomised SVD and
k-means run on the GPU. The diffusion time, the powering, MDS and the
landmark interpolation stay on the CPU.

## Usage

``` r
rs_phate_gpu(embd, n_dim, k, phate_params, seed, backend = "wgpu", verbose)
```

## Arguments

- embd:

  Numerical matrix. The data to use to generate the embeddings. Should
  be of dimensions samples x features.

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

  String. `"wgpu"` or `"mlx"`. `"mlx"` runs the kNN search and the
  spectral landmark selection on MLX (Apple Silicon, needs a build with
  the MLX backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).

- verbose:

  Integer. If `0L` -\> silent or `1L` for normal verbosity; `2L` for
  detailed verbosity.

## Value

The PHATE embeddings.
