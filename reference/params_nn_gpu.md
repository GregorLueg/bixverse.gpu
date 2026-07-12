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
  refine_sweeps = 0L,
  delta = 0.001,
  rho = NULL,
  beam_width = NULL,
  max_beam_iters = NULL,
  n_entry_points = NULL
)
```

## Arguments

- dist_metric:

  Character. The distance metric to use. Defaults to `"euclidean"`.

- n_list:

  Optional integer. IVF GPU: Number of clusters to use. If `NULL`, will
  default to `sqrt(n)`.

- n_probes:

  Optional integer. IVF GPU: Number of clusters to probe. If `NULL`,
  will default to `sqrt(n_list)`.

- node_degree_final:

  Optional integer. Final node degree of the CAGRA navigational graph.
  If `NULL`, defaults to `30` on the Rust side.

- k_build:

  Optional integer. Number of k-neighbours during the NNDescent build
  phase before CAGRA pruning. If `NULL`, defaults to
  `1.5 * node_degree_final` on the Rust side. (Cannot be smaller than
  `node_degree_final`)

- n_tree:

  Optional integer. CAGRA GPU: Number of trees for graph build.
  Automatically if `NULL`.

- refine_sweeps:

  Integer. Number of refinement sweeps during graph generation.

- delta:

  Float. CAGRA GPU: Early termination parameter for NN descent. Defaults
  to `0.001`.

- rho:

  Optional float. CAGRA GPU: Sample rate parameter for NN descent.
  Defaults to `0.5` if not provided.

- beam_width:

  Optional integer. CAGRA GPU: Beam width for beam search. If not
  provided will be set to `max(c(k, node_degree_final, 16L)) * 2`.

- max_beam_iters:

  Optional integer. CAGRA GPU: Maximum number of beam search iterations.
  If not provided, defaults to `3 * beam_width`.

- n_entry_points:

  Optional integer. CAGRA GPU: Number of entry points for beam search.
  If not provided, defaults to `8L`.

## Value

A list with the GPU nearest neighbour parameters.
