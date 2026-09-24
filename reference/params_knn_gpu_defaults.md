# Default parameters for the GPU nearest neighbour backends

GPU sibling of
[`bixverse::params_knn_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_knn_defaults.html).
The GPU indices take a different knob set: there is no Annoy and no HNSW
on the device, so what survives is exhaustive, IVF and NN-descent.

## Usage

``` r
params_knn_gpu_defaults()
```

## Value

A named list with the following elements:

- k - Integer. Number of neighbours. `0L` hands the choice to Rust,
  which uses `sqrt(n_cells) * 0.5` and then adjusts for the simulated
  doublets. Defaults to `15L`.

- knn_method - String. The GPU index to use. One of
  `c("exhaustive", "ivf", "nndescent")`. Defaults to `"exhaustive"`.

- ann_dist - String. Manhattan is not supported by the GPU kernels. One
  of `c("euclidean", "cosine")`. Defaults to `"euclidean"`.

- n_list - Integer or `NULL`. IVF only. Number of clusters. `NULL` gives
  `sqrt(n)`. Defaults to `NULL`.

- n_probe - Integer or `NULL`. IVF only. Clusters to probe. `NULL` gives
  `sqrt(n_list)`. Defaults to `NULL`.

- graph_k - Integer or `NULL`. NN-descent only. Node degree of the graph
  after pruning. `NULL` gives 30, widened to cover `k` when
  `extract_knn` is set. Defaults to `NULL`.

- k_build - Integer or `NULL`. NN-descent only. Build degree before
  pruning. `NULL` gives `max(k, floor(1.5 * k))`. Defaults to `NULL`.

- n_tree - Integer or `NULL`. NN-descent only. Trees seeding the
  descent. Defaults to `NULL`.

- delta - Numeric. NN-descent only. Termination criterium for the
  descent. Defaults to `0.001`.

- rho - Numeric or `NULL`. NN-descent only. Sampling rate for the
  descent. Defaults to `NULL`.

- refine_knn - Integer or `NULL`. NN-descent only. 2-hop refinement
  sweeps after the descent. Buys graph quality at a linear cost. `NULL`
  gives 0. Defaults to `NULL`.

- beam_width - Integer or `NULL`. NN-descent only. Beam width when
  querying. Ignored when `extract_knn` is set. Defaults to `NULL`.

- max_beam_iters - Integer or `NULL`. NN-descent only. Beam search
  iterations. Ignored when `extract_knn` is set. Defaults to `NULL`.

- n_entry_points - Integer or `NULL`. NN-descent only. Entry points when
  querying. Ignored when `extract_knn` is set. Defaults to `NULL`.

- extract_knn - Boolean. NN-descent only. Hand back the graph the
  descent built instead of beam searching over it. Defaults to `FALSE`.

## Details

NN-descent builds a CAGRA graph and, with `extract_knn = TRUE`, hands
that graph back rather than beam searching over it. Note that this saves
the query, not the build: the descent itself dominates, and its build
degree tracks `k`. NN-descent is therefore a low-`k` tool on the GPU.
Above `k` of roughly 30 both exhaustive and IVF beat it, and by
`k = 200` they beat it by more than an order of magnitude.
