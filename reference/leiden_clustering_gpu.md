# GPU-accelerated Leiden or Louvain community detection on an igraph

GPU counterpart of
[`bixverse::leiden_clustering()`](https://gregorlueg.github.io/bixverse/reference/leiden_clustering.html).
The level-0 local moving runs on the GPU; the coarser levels, the Leiden
refinement and the aggregation stay on the CPU. Graphs below 50,000
nodes, or with a node of degree above 512, run fully on the CPU with
coloured local moving, as the kernel launches cost more than they save
there. On one device the partition depends on `seed` only, but it is not
identical to the CPU result: the GPU accepts moves with an order-free
rule instead of the CPU's sequential re-check. Communities are numbered
by size, `1` being the largest.

## Usage

``` r
leiden_clustering_gpu(
  graph,
  resolution = 1,
  algorithm = c("leiden", "louvain"),
  beta = 0.01,
  n_iterations = NULL,
  min_gain = 0,
  local_moving = c("queue", "coloured"),
  seed = 42L
)
```

## Arguments

- graph:

  igraph. Undirected. If it has an edge attribute `weight`, it is used,
  otherwise all edges get weight 1.

- resolution:

  Numeric. Modularity resolution parameter. Higher values give more
  communities.

- algorithm:

  String. One of `c("leiden", "louvain")`.

- beta:

  Numeric. Randomness of the Leiden refinement merges. Ignored for
  Louvain.

- n_iterations:

  Optional integer. Number of full iterations, each starting from the
  previous partition. A negative value iterates until the partition is
  stable. If `NULL`, `5L` for Leiden and `2L` for Louvain.

- min_gain:

  Numeric. With a negative `n_iterations`, also stop once an iteration
  gains no more than this in modularity.

- local_moving:

  String. One of `c("queue", "coloured")`. Applies to the coarser levels
  only; level 0 always runs the coloured moves on the GPU.

- seed:

  Integer. Random seed.

## Value

Integer vector with the community of each node (1-based, `1` is the
largest community).

## References

Traag, Waltman & van Eck, Sci Rep, 2019; Blondel, et al., J Stat Mech,
2008
