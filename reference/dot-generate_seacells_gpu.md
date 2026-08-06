# Shared implementation of the GPU SEACells generation

Body behind both
[`generate_seacells_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_seacells_gpu_sc.md)
methods. Pulls the embedding and the cached kNN graph off the object,
hands them to
[`rs_seacells_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_seacells_gpu.md)
and wraps the result into a
[`bixverse::MetaCells()`](https://gregorlueg.github.io/bixverse/reference/MetaCells.html).

## Usage

``` r
.generate_seacells_gpu(
  object,
  seacell_params,
  embd_to_use,
  no_embd_to_use,
  cells_to_use,
  regenerate_knn,
  target_size,
  seed,
  .verbose
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- seacell_params:

  List. Output of
  [`bixverse::params_sc_seacells()`](https://gregorlueg.github.io/bixverse/reference/params_sc_seacells.html).
  A list with the following items:

  - n_sea_cells - Number of SEA cells to detect.

  - max_fw_iters - Maximum iterations for the Frank-Wolfe algorithm per
    matrix update.

  - convergence_epsilon - Convergence threshold. Algorithm stops when
    RSS change \< epsilon \* RSS(0).

  - max_iter - Maximum iterations to run SEACells for.

  - min_iter - Minimum iterations to run SEACells for.

  - greedy_threshold - Maximum number of cells before defaulting to
    rapid random selection of archetypes.

  - graph_building - Graph building method.

  - pruning - Boolean. Shall small values be pruned during the Frank-
    Wolfe iterations.

  - pruning_threshold - The threshold below which pruning shall be
    applied during Frank-Wolfe iterations.

  - n_landmarks - Optional integer. Number of landmarks for the Nystroem
    archetype initialisation.

  - knn - List of kNN parameters. See
    [`bixverse::params_knn_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_knn_defaults.html)
    for available parameters and their defaults.

- embd_to_use:

  String. The embedding to use. Atm, the only option is `"pca"`.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL` all
  will be used.

- cells_to_use:

  Optional string. Names of the cells to use for the generation of the
  SEACells. Forces a kNN rebuild on that subset.

- regenerate_knn:

  Boolean. Shall a kNN graph be regenerated. If not, the internal one
  will be used.

- target_size:

  Numeric. The library target size to normalise the meta cells to.

- seed:

  Integer. Seed for reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

A
[`bixverse::MetaCells()`](https://gregorlueg.github.io/bixverse/reference/MetaCells.html).
