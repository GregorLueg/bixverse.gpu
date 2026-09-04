# Generate a GPU-accelerated kNN graph

**\[experimental\]** Builds a kNN graph from an embedding matrix on the
wgpu backend. Three searches are available: an exact brute-force scan,
an IVF index that probes a subset of Voronoi cells, and a CAGRA-style
NNDescent graph that is either beam searched or handed back as the
descent left it.

Euclidean distances come back as true L2, not squared.

## Usage

``` r
rs_gpu_knn(embd, k, knn_method, nn_params, seed, verbose)
```

## Arguments

- embd:

  Numeric matrix of embeddings, cells x features.

- k:

  Integer. Number of neighbours to return, self excluded.

- knn_method:

  String. One of `c("nndescent", "exhaustive", "ivf")`.

- nn_params:

  A named list with the parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md)

- seed:

  Integer. Random seed for reproducibility.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A named list with:

- `indices` - Integer matrix of shape cells x k with 0-based neighbour
  indices.

- `dist` - Numeric matrix of shape cells x k with distances to the
  neighbours.

- `dist_metric` - Character. The distance metric used.
