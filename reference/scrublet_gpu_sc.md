# Doublet detection with Scrublet on the GPU

GPU counterpart of
[`bixverse::scrublet_sc()`](https://gregorlueg.github.io/bixverse/reference/scrublet_sc.html).
Three stages run on the WGPU backend: the randomised sparse SVD of the
observed cells, the projection of the simulated doublets into that PC
space, and the kNN over the combined embedding. HVG selection, doublet
simulation, the kNN classifier and the Otsu threshold stay on the CPU,
so the speedup tracks how much of the run the SVD and the kNN own. That
share grows with cell count: the combined embedding is
`(1 + sim_doublet_ratio) * n_cells` rows tall and an exhaustive kNN over
it is quadratic.

## Usage

``` r
scrublet_gpu_sc(
  object,
  scrublet_params = params_scrublet_gpu(),
  seed = 42L,
  streaming = NULL,
  cells_to_use = NULL,
  group_by = NULL,
  return_combined_pca = FALSE,
  return_pairs = FALSE,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` class from `bixverse`.

- scrublet_params:

  List. Output of
  [`params_scrublet_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_scrublet_gpu.md).

- seed:

  Integer. Random seed.

- streaming:

  Optional boolean. Shall the counts be streamed during HVG selection.
  If `NULL`, resolved from the cell count.

- cells_to_use:

  Optional character vector. Names of the cells to run on. The returned
  object covers exactly these cells.

- group_by:

  Optional string. Column in the obs table to run the method per level
  of, typically a sample identifier.

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

A `ScrubletRes` S3 object, identical in shape to the CPU one, with the
following items:

- predicted_doublets - Boolean vector indicating which observed cells
  were predicted as doublets (TRUE = doublet, FALSE = singlet).

- doublet_scores_obs - Numerical vector with the likelihood of being a
  doublet for the observed cells.

- doublet_scores_sim - Numerical vector with the likelihood of being a
  doublet for the simulated cells.

- doublet_errors_obs - Numerical vector with the standard errors of the
  scores for the observed cells.

- z_scores - Z-scores for the observed cells. Represents:
  `score - threshold / error`.

- threshold - Used threshold.

- detected_doublet_rate - Fraction of cells that are called as doublet.

- detectable_doublet_fraction - Fraction of simulated doublets with
  scores above the threshold.

- overall_doublet_rate - Estimated overall doublet rate.

- pca - Optional PCA embeddings across the original cells and simulated
  doublets.

- pair_1 - Optional index of the parent cell 1 of the simulated
  doublets.

- pair_2 - Optional index of the parent cell 2 of the simulated
  doublets.

The 0-indexed cell indices are attached as the `cell_indices` attribute.
Grouped runs additionally carry `grouped` and `group_by_col` attributes
and a `cell_groups` element.

## Details

Scores do not match the CPU bit for bit. The SVD is randomised on both
sides but draws a different sketch, and the GPU indices break neighbour
ties differently. Expect a correlation around 0.99 rather than equality,
and a handful of borderline calls to flip because Otsu's threshold is a
step function of the histogram bins.

## References

Wolock, et al., Cell Syst, 2020
