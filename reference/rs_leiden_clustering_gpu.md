# GPU-accelerated Leiden and Louvain clustering

**\[experimental\]** GPU twin of
[`bixverse::rs_leiden_clustering`](https://gregorlueg.github.io/bixverse/reference/rs_leiden_clustering.html).
Level-0 local moving runs on the wgpu backend; coarser levels,
refinement and aggregation stay on the CPU. Graphs below 50,000 nodes,
or with a node of degree above 512, run fully on the CPU with coloured
local moving. On one device the result depends on the seed only.

## Usage

``` r
rs_leiden_clustering_gpu(edges, weights, n_nodes, leiden_params)
```

## Arguments

- edges:

  Integer vector. Flattened edge pairs (1-indexed!),
  `c(from_1, to_1, from_2, to_2, ...)`, as
  [`bixverse::rs_graph_to_edges()`](https://gregorlueg.github.io/bixverse/reference/rs_graph_to_edges.html)
  returns them.

- weights:

  Numeric vector. One weight per pair.

- n_nodes:

  Integer. Number of nodes in the graph.

- leiden_params:

  List. Named list with the optional fields `resolution`, `beta`,
  `n_iterations` (negative: until stable), `min_gain`, `refine` (`TRUE`
  for Leiden, `FALSE` for Louvain), `local_moving` (coarser levels only)
  and `seed`.

## Value

Integer vector with the 1-based community of each node; `1` is the
largest community.
