# GPU: SCENIC gene-regulatory network inference (disk-backed)

**\[experimental\]** GPU equivalent of
[`bixverse::rs_scenic_grn`](https://gregorlueg.github.io/bixverse/reference/rs_scenic_grn.html).
Reads the target genes from disk in chunks and dispatches per-batch tree
fits to the WGPU backend. GBM is not supported on GPU; use the CPU
version.

## Usage

``` r
rs_scenic_grn_gpu(
  f_path_genes,
  cell_indices,
  gene_indices,
  tf_indices,
  scenic_params,
  wave_byte_budget,
  seed,
  verbose
)
```

## Arguments

- f_path_genes:

  String. Path to the `counts_genes.bin` file.

- cell_indices:

  Integer vector. 0-indexed positions of cells to include.

- gene_indices:

  Integer vector. 0-indexed positions of target genes.

- tf_indices:

  Integer vector. 0-indexed positions of TF predictors.

- scenic_params:

  Named list. See
  [`bixverse::params_scenic()`](https://gregorlueg.github.io/bixverse/reference/params_scenic.html).

- wave_byte_budget:

  Double. VRAM ceiling for per-wave histogram + cumulative tensors
  (bytes). Default 4 GiB.

- seed:

  Integer. Random seed.

- verbose:

  Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.

## Value

A gene x TF importance matrix.
