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
  `c("exhaustive", "ivf", "nndescent", "cagra")`. Defaults to
  `"exhaustive"`.

- ann_dist - String. Manhattan is not supported by the GPU kernels. One
  of `c("euclidean", "cosine")`. Defaults to `"euclidean"`.

- n_list - Integer or `NULL`. IVF only. Number of clusters. `NULL` gives
  `sqrt(n)`. Defaults to `NULL`.

- n_probe - Integer or `NULL`. IVF only. Clusters to probe. `NULL` gives
  `sqrt(n_list)`. Defaults to `NULL`.

- graph_k - Integer or `NULL`. CAGRA only. Node degree of the graph
  after pruning. `NULL` gives 30. Defaults to `NULL`.

- k_build - Integer or `NULL`. NN-descent and CAGRA. Working degree of
  the descent. `NULL` gives `max(k, floor(1.5 * k))`. Defaults to
  `NULL`.

- n_tree - Integer or `NULL`. NN-descent and CAGRA. Trees seeding the
  descent. Defaults to `NULL`.

- delta - Numeric. NN-descent and CAGRA. Termination criterium for the
  descent. Defaults to `0.001`.

- rho - Numeric or `NULL`. NN-descent and CAGRA. Sampling rate for the
  descent. Defaults to `NULL`.

- refine_knn - Integer or `NULL`. NN-descent and CAGRA. 2-hop refinement
  sweeps after the descent. Buys graph quality at a linear cost. `NULL`
  gives 0. Defaults to `NULL`.

- beam_width - Integer or `NULL`. CAGRA only, which includes NN-descent
  on a cross-query. Beam width when querying. Defaults to `NULL`.

- max_beam_iters - Integer or `NULL`. CAGRA only, which includes
  NN-descent on a cross-query. Beam search iterations. Defaults to
  `NULL`.

- n_entry_points - Integer or `NULL`. CAGRA only, which includes
  NN-descent on a cross-query. Entry points when querying. Defaults to
  `NULL`.

## Details

`"nndescent"` hands back the kNN graph the descent built, with no index
and no search; rows the descent left short are topped up with an exact
search of just those rows. Where the query is a different set of cells
(BBKNN, fastMNN) it builds and beam searches a CAGRA graph instead.
`"cagra"` always does the latter, self-queries included. The descent
itself dominates either way, and its build degree tracks `k`, so both
are low-`k` tools on the GPU. Above `k` of roughly 30 exhaustive and IVF
beat them.
