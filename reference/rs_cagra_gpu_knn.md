# Generate a CAGRA-style GPU-accelerated kNN graph

Builds a kNN graph from an embedding matrix using the CAGRA algorithm on
the wgpu backend. Supports two retrieval modes: direct extraction from
the NNDescent graph, or beam search over the pruned CAGRA graph.

## Usage

``` r
rs_cagra_gpu_knn(embd, cagra_params, extract_knn, seed, verbose)
```

## Arguments

- embd:

  Numeric matrix of embeddings, cells x features.

- cagra_params:

  A named list with the parameters, see
  [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md)

- extract_knn:

  Logical. If `TRUE`, extracts the kNN graph directly from the NNDescent
  result (faster, slightly lower precision). If `FALSE`, runs beam
  search over the pruned CAGRA graph (slower, higher precision).

- seed:

  Integer. Random seed for reproducibility.

- verbose:

  Logical. Whether to print progress messages.

## Value

A named list with:

- `indices` - Integer matrix of shape cells x k_query with 0-based
  neighbour indices.

- `dist` - Numeric matrix of shape cells x k_query with distances to the
  neighbours.

- `dist_metric` - Character. The distance metric used.
