# Wrapper function to generate GPU nearest neighbour parameters

Wrapper function to generate GPU nearest neighbour parameters

## Usage

``` r
params_nn_gpu(
  dist_metric = c("euclidean", "cosine"),
  n_list = NULL,
  n_probes = NULL,
  node_degree_final = NULL,
  k_build = NULL,
  n_tree = NULL,
  delta = 0.001,
  rho = NULL,
  beam_width = NULL,
  max_beam_iters = NULL,
  n_entry_points = NULL
)
```

## Arguments

- dist_metric:

  String. The distance metric to use. One of `c("euclidean", "cosine")`.
  Defaults to `"euclidean"`.

- n_list:

  Integer or `NULL`. IVF GPU: Number of clusters to use. If `NULL`, will
  default to `sqrt(n)`. Defaults to `NULL`.

- n_probes:

  Integer or `NULL`. IVF GPU: Number of clusters to probe. If `NULL`,
  will default to `sqrt(n_list)`. Defaults to `NULL`.

- node_degree_final:

  Integer or `NULL`. CAGRA GPU: Final node degree of the navigational
  graph. If `NULL`, defaults to `30` on the Rust side. Ignored by UMAP,
  tSNE and PHATE. Defaults to `NULL`.

- k_build:

  Integer or `NULL`. NNDescent and CAGRA GPU: Working degree of the
  descent. If `NULL`, defaults to `1.5 * k` on the Rust side. For CAGRA
  it cannot be smaller than `node_degree_final`. Defaults to `NULL`.

- n_tree:

  Integer or `NULL`. NNDescent and CAGRA GPU: Number of trees seeding
  the descent. Automatically if `NULL`. Defaults to `NULL`.

- delta:

  Numeric. NNDescent and CAGRA GPU: Early termination parameter for the
  descent. Defaults to `0.001`.

- rho:

  Numeric or `NULL`. NNDescent and CAGRA GPU: Sample rate for the
  descent. Defaults to `NULL`.

- beam_width:

  Integer or `NULL`. CAGRA GPU: Beam width for beam search. If not
  provided will be set to `max(c(k, node_degree_final, 16L)) * 2`.
  Ignored by UMAP, tSNE and PHATE. Defaults to `NULL`.

- max_beam_iters:

  Integer or `NULL`. CAGRA GPU: Maximum number of beam search
  iterations. If not provided, defaults to `3 * beam_width`. Ignored by
  UMAP, tSNE and PHATE. Defaults to `NULL`.

- n_entry_points:

  Integer or `NULL`. CAGRA GPU: Number of entry points for beam search.
  If not provided, defaults to `8L`. Ignored by UMAP, tSNE and PHATE.
  Defaults to `NULL`.

## Value

A named list with the following elements:

- dist_metric - String. The distance metric to use. One of
  `c("euclidean", "cosine")`. Defaults to `"euclidean"`.

- n_list - Integer or `NULL`. IVF GPU: Number of clusters to use. If
  `NULL`, will default to `sqrt(n)`. Defaults to `NULL`.

- n_probes - Integer or `NULL`. IVF GPU: Number of clusters to probe. If
  `NULL`, will default to `sqrt(n_list)`. Defaults to `NULL`.

- node_degree_final - Integer or `NULL`. CAGRA GPU: Final node degree of
  the navigational graph. If `NULL`, defaults to `30` on the Rust side.
  Ignored by UMAP, tSNE and PHATE. Defaults to `NULL`.

- k_build - Integer or `NULL`. NNDescent and CAGRA GPU: Working degree
  of the descent. If `NULL`, defaults to `1.5 * k` on the Rust side. For
  CAGRA it cannot be smaller than `node_degree_final`. Defaults to
  `NULL`.

- n_tree - Integer or `NULL`. NNDescent and CAGRA GPU: Number of trees
  seeding the descent. Automatically if `NULL`. Defaults to `NULL`.

- delta - Numeric. NNDescent and CAGRA GPU: Early termination parameter
  for the descent. Defaults to `0.001`.

- rho - Numeric or `NULL`. NNDescent and CAGRA GPU: Sample rate for the
  descent. Defaults to `NULL`.

- beam_width - Integer or `NULL`. CAGRA GPU: Beam width for beam search.
  If not provided will be set to
  `max(c(k, node_degree_final, 16L)) * 2`. Ignored by UMAP, tSNE and
  PHATE. Defaults to `NULL`.

- max_beam_iters - Integer or `NULL`. CAGRA GPU: Maximum number of beam
  search iterations. If not provided, defaults to `3 * beam_width`.
  Ignored by UMAP, tSNE and PHATE. Defaults to `NULL`.

- n_entry_points - Integer or `NULL`. CAGRA GPU: Number of entry points
  for beam search. If not provided, defaults to `8L`. Ignored by UMAP,
  tSNE and PHATE. Defaults to `NULL`.
