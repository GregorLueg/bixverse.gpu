# GPU: SEACells meta cell generation

**\[experimental\]** GPU equivalent of
[`bixverse::rs_get_seacells`](https://gregorlueg.github.io/bixverse/reference/rs_get_seacells.html).
Both Frank-Wolfe solves, the B-gradient argmin and the per-cell A
columns, are dispatched to the WGPU backend. The kNN graph, the kernel
matrix, the RSS evaluation and the aggregation into pseudo-bulk counts
all stay on the CPU.

## Usage

``` r
rs_seacells_gpu(
  f_path,
  embd,
  cells_to_keep,
  cells_to_use,
  knn_data,
  seacells_params,
  target_size,
  seed,
  verbose
)
```

## Arguments

- f_path:

  String. Path to the `counts_cells.bin` file.

- embd:

  Numeric matrix. Cells x components embedding, one row per QC-passing
  cell.

- cells_to_keep:

  Optional integer vector. 0-indexed original row indices the embedding
  was built from, in embedding row order.

- cells_to_use:

  Optional integer vector. 0-indexed original row indices to narrow the
  run to. Forces a kNN rebuild on that subset.

- knn_data:

  Optional list. Precomputed kNN graph with `indices`, `dist`,
  `dist_metric` and `k`. Ignored when `cells_to_use` is set.

- seacells_params:

  Named list. See
  [`bixverse::params_sc_seacells()`](https://gregorlueg.github.io/bixverse/reference/params_sc_seacells.html).

- target_size:

  Double. Library target size the meta cells are normalised to.

- seed:

  Integer. Random seed.

- verbose:

  Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.

## Value

A list with the cell assignments, the aggregated meta cell counts in
compressed sparse form, the RSS history and the archetype cell indices.

## References

Persad, et al., Nat. Biotechnol., 2023.
