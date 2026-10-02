# GPU: Bonsai tree from metacell counts

**\[experimental\]** GPU equivalent of
[`bixverse::rs_mc_bonsai`](https://gregorlueg.github.io/bixverse/reference/rs_mc_bonsai.html).
Sanity runs on the WGPU backend over the metacells' aggregated raw
counts, the tree search and the layout on the CPU. Every metacell is a
leaf.

## Usage

``` r
rs_mc_bonsai_gpu(sparse_data, gene_indices, bonsai_params, verbose)
```

## Arguments

- sparse_data:

  List. The raw metacell counts, see
  [`bixverse::mc_counts_to_list()`](https://gregorlueg.github.io/bixverse/reference/mc_counts_to_list.html)
  with `assay = "raw"`.

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
[`bixverse::rs_mc_bonsai()`](https://gregorlueg.github.io/bixverse/reference/rs_mc_bonsai.html).

## References

de Groot, et al., Nat Biotechnol, 2026; Breda, et al., Nat Biotechnol,
2021.
