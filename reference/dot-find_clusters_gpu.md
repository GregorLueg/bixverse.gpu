# Run the GPU graph clustering and write it to the obs table

Run the GPU graph clustering and write it to the obs table

## Usage

``` r
.find_clusters_gpu(object, cluster_algorithm, res, name, modality, seed, ...)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class.

- cluster_algorithm:

  String. One of `c("leiden", "louvain")`.

- res:

  Numeric. The modularity resolution parameter.

- name:

  String. The name of the obs column.

- modality:

  String. One of `c("rna", "adt", "wnn")`.

- seed:

  Integer. For reproducibility.

- ...:

  Passed to
  [`leiden_clustering_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/leiden_clustering_gpu.md).

## Value

The object with the added clustering in the obs table.
