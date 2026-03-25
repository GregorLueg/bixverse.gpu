# Generate an IVF-GPU-accelerated kNN graph

Builds an IVF index over the provided embedding matrix and queries each
vector against it to produce a kNN graph. Runs on the wgpu backend.

## Usage

``` r
rs_ivf_gpu_knn(embd, ivf_params, seed, verbose)
```

## Arguments

- embd:

  Numeric matrix of embeddings, cells x features.

- ivf_params:

  A named list with the parameters, see
  [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md)

- seed:

  Integer. Random seed for reproducibility.

- verbose:

  Logical. Whether to print progress messages.

## Value

A named list with:

- `indices` - Integer matrix of shape cells x k with 0-based neighbour
  indices.

- `dist` - Numeric matrix of shape cells x k with distances to the
  neighbours.

- `dist_metric` - Character. The distance metric used.
