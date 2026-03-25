# Default parameters for CAGRA-style kNN search

Default parameters for CAGRA-style kNN search

## Usage

``` r
params_sc_cagra(
  k_query = 15L,
  ann_dist = "cosine",
  k = NULL,
  k_build = NULL,
  refine_sweeps = 0L,
  max_iters = NULL,
  n_trees = NULL,
  delta = 0.001,
  rho = NULL,
  beam_width = NULL,
  max_beam_iters = NULL,
  n_entry_points = NULL
)
```

## Arguments

- k_query:

  Integer. Number of neighbours to identify.

- ann_dist:

  Character. Distance metric to use. One of `"euclidean"` or `"cosine"`.

- k:

  Optional integer. Final node degree of the CAGRA navigational graph.
  If `NULL`, defaults to `30` on the Rust side.

- k_build:

  Optional integer. Number of k-neighbours during the NNDescent build
  phase before CAGRA pruning. If `NULL`, defaults to `1.5 * k` on the
  Rust side.

- refine_sweeps:

  Integer. Number of refinement sweeps during graph generation.

- max_iters:

  Optional integer. Maximum iterations for the NNDescent rounds. If
  `NULL`, determined automatically.

- n_trees:

  Optional integer. Number of trees to use in the initial
  GPU-accelerated forest. If `NULL`, determined automatically.

- delta:

  Numeric. Termination criterion for the NNDescent iterations.

- rho:

  Optional numeric. Sampling rate during NNDescent iterations. If
  `NULL`, determined automatically.

- beam_width:

  Optional integer. Beam width during querying. If `NULL`, determined
  automatically.

- max_beam_iters:

  Optional integer. Maximum beam iterations. If `NULL`, determined
  automatically.

- n_entry_points:

  Optional integer. Number of entry points into the graph. If `NULL`,
  determined automatically.

## Value

A list with the parameters.
