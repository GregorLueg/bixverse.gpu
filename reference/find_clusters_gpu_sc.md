# Graph-based clustering of cells on the neighbour graph (GPU)

GPU counterpart of
[`bixverse::find_clusters_sc()`](https://gregorlueg.github.io/bixverse/reference/find_clusters_sc.html).
Runs Leiden or Louvain via
[`leiden_clustering_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/leiden_clustering_gpu.md)
on the neighbour graph and adds a column to the obs table. Only the
level-0 local moving runs on the GPU, and only on graphs of at least
50,000 cells; smaller graphs run on the CPU. `0` is the largest cluster.

## Usage

``` r
find_clusters_gpu_sc(
  object,
  cluster_algorithm = c("leiden", "louvain"),
  res = 1,
  name = "leiden_clustering",
  modality = c("rna", "adt", "wnn"),
  seed = 42L,
  ...
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- cluster_algorithm:

  String. One of `c("leiden", "louvain")`.

- res:

  Numeric. The modularity resolution parameter, see
  [`leiden_clustering_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/leiden_clustering_gpu.md).

- name:

  String. The name to add to the obs table.

- modality:

  String. Which modality's neighbour graph to use. One of
  `c("rna", "adt", "wnn")`. The two latter options are only available
  for multi-modal objects.

- seed:

  Integer. For reproducibility.

- ...:

  Further parameters for
  [`leiden_clustering_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/leiden_clustering_gpu.md):
  `n_iterations`, `beta`, `min_gain` and `local_moving`.

## Value

The object with the added clustering in the obs table.
