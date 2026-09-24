# Run BBKNN on the GPU

GPU counterpart of
[`bixverse::bbknn_sc()`](https://gregorlueg.github.io/bixverse/reference/bbknn_sc.html),
implementing the batch-balanced k-nearest neighbour algorithm from
Polański, et al. One nearest neighbour index is built per batch and
queried by every cell, so each cell gets `neighbours_within_batch`
neighbours from every batch. The UMAP connectivity calculations that
reduce spurious connections then run on the CPU, shared with the CPU
implementation.

## Usage

``` r
bbknn_gpu_sc(
  object,
  batch_column,
  no_neighbours_to_keep = 5L,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  bbknn_params = params_sc_bbknn_gpu(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- batch_column:

  String. The column with the batch information in the obs data of the
  class.

- no_neighbours_to_keep:

  Integer. Maximum number of neighbours to keep from the BBKNN
  algorithm. Generating neighbours per batch can produce a lot of them,
  so this keeps the top `no_neighbours_to_keep`. Defaults to `5L`.

- embd_to_use:

  String. The embedding to use. Atm, the only option is `"pca"`.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL` all
  will be used.

- bbknn_params:

  List. Output of
  [`params_sc_bbknn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_bbknn_gpu.md).

- seed:

  Integer. Random seed.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

The object with the added kNN matrix based on BBKNN and the graph based
on the returned connectivities of the algorithm.

## Details

Only the per-batch searches move to the device, and they are the part
that scales with batch count: BBKNN builds one index per batch and
queries each with all cells, so the work grows as `n_cells * n_batches`.
With a handful of batches on a small object the CPU is fine. The GPU
starts to matter once you have many samples.

Results match the CPU path exactly with `knn_method = "exhaustive"`,
since both are exact and recompute distances against the same embedding.
The approximate backends break ties differently and will not.

## References

Polański, et al., Bioinformatics, 2020
