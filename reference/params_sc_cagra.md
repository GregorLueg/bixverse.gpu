# Default parameters for CAGRA-style kNN search (deprecated)

**\[deprecated\]**

The CAGRA, IVF and exhaustive GPU searches share one parameter wrapper
now, see
[`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

## Usage

``` r
params_sc_cagra(
  k = 15L,
  ann_dist = "euclidean",
  node_degree_final = NULL,
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

- k:

  Integer. Number of neighbours. Carried on the returned list so the
  deprecated generics can still read it.

- ann_dist:

  Character. One of `"euclidean"` or `"cosine"`.

- node_degree_final:

  Optional integer. Final node degree of the CAGRA navigational graph.

- k_build:

  Optional integer. Node degree during the NNDescent build phase before
  CAGRA pruning.

- refine_sweeps:

  Integer. Ignored, the knob is gone.

- max_iters:

  Optional integer. Ignored, the knob is gone.

- n_trees:

  Optional integer. Number of trees in the initial forest.

- delta:

  Numeric. Termination criterion for the NNDescent iterations.

- rho:

  Optional numeric. Sampling rate during NNDescent iterations.

- beam_width:

  Optional integer. Beam width during querying.

- max_beam_iters:

  Optional integer. Maximum beam iterations.

- n_entry_points:

  Optional integer. Number of entry points.

## Value

A list with the parameters, as returned by
[`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).
