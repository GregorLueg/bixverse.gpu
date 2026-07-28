# GPU: SCENIC GRN inference (streaming; bounded host memory)

**\[experimental\]** Streaming GPU equivalent of
[`bixverse::rs_scenic_grn_streaming`](https://gregorlueg.github.io/bixverse/reference/rs_scenic_grn_streaming.html).
Reads targets in I/O chunks and dispatches each in-chunk batch serially
to the GPU. Peak host memory is bounded to one chunk of sparse columns.
GBM is not supported on GPU; use the CPU version.

## Usage

``` r
rs_scenic_grn_streaming_gpu(
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
