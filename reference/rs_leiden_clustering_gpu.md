# GPU-accelerated Leiden and Louvain clustering

**\[experimental\]** GPU twin of
[`bixverse::rs_leiden_clustering`](https://gregorlueg.github.io/bixverse/reference/rs_leiden_clustering.html).
Level-0 local moving runs on the wgpu backend; coarser levels,
refinement and aggregation stay on the CPU. Graphs below 50,000 nodes,
or with a node of degree above 512, run fully on the CPU with coloured
local moving. On one device the result depends on the seed only.

## Usage

``` r
rs_leiden_clustering_gpu(from, to, weights, n_nodes, leiden_params)
```

## Arguments

- from:

  Integer vector. 1-based source node of each edge.

- to:

  Integer vector. 1-based target node of each edge. Same length as
  `from`.

- weights:

  Numeric vector. Edge weights, same length as `from`.

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
