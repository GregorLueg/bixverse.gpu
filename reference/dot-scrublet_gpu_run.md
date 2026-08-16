# Run GPU Scrublet on a set of cells

GPU sibling of `bixverse:::.scrublet_run()`. Resolves streaming, calls
[`rs_sc_scrublet_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_sc_scrublet_gpu.md)
and stamps the result with the `ScrubletRes` class plus the
`cell_indices` attribute that every downstream method reads.

## Usage

``` r
.scrublet_gpu_run(
  object,
  cells_to_use,
  scrublet_params,
  seed,
  streaming,
  return_combined_pca,
  return_pairs,
  .verbose
)
```

## Arguments

- object:

  `SingleCells` class from `bixverse`.

- cells_to_use:

  Integer vector of 0-indexed cell indices.

- scrublet_params:

  List. Output of
  [`params_scrublet_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_scrublet_gpu.md).

- seed:

  Integer. Random seed.

- streaming:

  Optional boolean. Shall the counts be streamed during HVG selection.
  If `NULL`, resolved from the cell count.

- return_combined_pca:

  Boolean. Shall the combined PCA of observed cells and simulated
  doublets be returned.

- return_pairs:

  Boolean. Shall the parent indices of the simulated doublets be
  returned.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

A `ScrubletRes` S3 object.
