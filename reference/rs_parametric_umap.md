# Parametric UMAP implementation

Trains a neural network encoder to learn a mapping from the input space
to a low-dimensional embedding that preserves the UMAP graph structure.
Supports both GPU (wgpu) and CPU (NdArray) backends. For small to medium
data sets (fewer than ~10k samples or narrow hidden layers), the CPU
backend is typically faster owing to GPU kernel dispatch overhead.

## Usage

``` r
rs_parametric_umap(
  data,
  n_dim,
  k,
  min_dist,
  spread,
  parametric_params,
  seed,
  verbose,
  use_gpu
)
```

## Arguments

- data:

  Numerical matrix. Data of dimensions samples x features.

- n_dim:

  Integer. Number of embedding dimensions.

- k:

  Integer. Number of nearest neighbours for graph construction.

- min_dist:

  Numeric. Minimum distance between embedded points.

- spread:

  Numeric. Effective scale of embedded points.

- parametric_params:

  Named list. Merged parametric UMAP parameters containing nearest
  neighbour, graph, and training configuration.

- seed:

  Integer. Seed for reproducibility.

- verbose:

  Boolean. Controls verbosity.

- use_gpu:

  Logical. If `TRUE`, trains on the wgpu backend. If `FALSE`, trains on
  the CPU via NdArray. Defaults to `TRUE`.

## Value

A named list with two elements: `embedding` (numerical matrix of
dimensions samples x n_dim) and `model` (external pointer to the trained
encoder for use with `rs_parametric_umap_predict`).
