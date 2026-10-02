# GPU: Bonsai tree from single cell counts

**\[experimental\]** GPU equivalent of
[`bixverse::rs_sc_bonsai`](https://gregorlueg.github.io/bixverse/reference/rs_sc_bonsai.html).
Sanity runs on the WGPU backend, streamed over chunks of genes and
keeping only the ones that pass Bonsai's ingest filters. The tree search
and the layout run on the CPU.

## Usage

``` r
rs_sc_bonsai_gpu(
  f_path_gene,
  f_path_cell,
  cell_indices,
  gene_indices,
  bonsai_params,
  verbose
)
```

## Arguments

- f_path_gene:

  String. Path to the `counts_genes.bin` file.

- f_path_cell:

  String. Path to the `counts_cells.bin` file. Supplies the library
  sizes.

- cell_indices:

  Integer. The cell indices to use. (0-indexed!) Sets the leaf order.

- gene_indices:

  Integer. The candidate genes. (0-indexed!)

- bonsai_params:

  List. Parameter list, see
  [`bixverse::params_sc_bonsai()`](https://gregorlueg.github.io/bixverse/reference/params_sc_bonsai.html).

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

The same list as
[`bixverse::rs_sc_bonsai()`](https://gregorlueg.github.io/bixverse/reference/rs_sc_bonsai.html):
`parent` (0-indexed, `-1` for the root), `branch`, `x`, `y`, `n_leaves`,
`loglik`, `steps`, `timings` and `genes_used` (0-indexed).

## References

de Groot, et al., Nat Biotechnol, 2026; Breda, et al., Nat Biotechnol,
2021.
