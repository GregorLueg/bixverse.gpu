# GPU: BBKNN batch correction

**\[experimental\]** GPU equivalent of
[`bixverse::rs_bbknn`](https://gregorlueg.github.io/bixverse/reference/rs_bbknn.html),
implementing the BBKNN algorithm from Polański, et al. One nearest
neighbour index is built per batch on the WGPU backend and queried by
every cell, so each cell gets `neighbours_within_batch` neighbours from
every batch. The UMAP connectivity calculations that follow stay on the
CPU and are shared with the CPU implementation.

## Usage

``` r
rs_bbknn_gpu(embd, batch_labels, bbknn_params, seed, verbose)
```

## Arguments

- embd:

  Numerical matrix. The embedding matrix used to generate the BBKNN
  results. Usually PCA. Rows represent cells.

- batch_labels:

  Integer vector. These represent to which batch a given cell belongs.
  Needs to be 0-indexed!

- bbknn_params:

  List. Parameter list, see
  [`params_sc_bbknn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_bbknn_gpu.md).

- seed:

  Integer. Seed for reproducibility purposes.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list of two lists representing the sparse matrix representation of the
distances and the connectivities. Each of them contains

- data - The values of the sparse matrix.

- indptr - The index pointers. 0-indexed.

- indices - The column indices. 0-indexed.

- nrow - Number of rows.

- ncol - Number of columns.

- cs_type - The sparse format, `"csr"` here.

## References

Polański, et al., Bioinformatics, 2020
