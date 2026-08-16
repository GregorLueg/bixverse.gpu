# GPU: Scrublet doublet detection

**\[experimental\]** GPU equivalent of
[`bixverse::rs_sc_scrublet`](https://gregorlueg.github.io/bixverse/reference/rs_sc_scrublet.html).
The PCA of the observed cells, the projection of the simulated doublets
and the kNN over the combined embedding run on the WGPU backend. HVG
selection, doublet simulation, scoring and the Otsu threshold stay on
the CPU. Which nearest neighbour index runs is decided by the
`knn_backend` element of `scrublet_params`.

## Usage

``` r
rs_sc_scrublet_gpu(
  f_path_gene,
  f_path_cell,
  cells_to_keep,
  scrublet_params,
  seed,
  verbose,
  streaming,
  return_combined_pca,
  return_pairs
)
```

## Arguments

- f_path_gene:

  String. Path to the `counts_genes.bin` file.

- f_path_cell:

  String. Path to the `counts_cells.bin` file.

- cells_to_keep:

  Integer vector. The indices (0-indexed!) of the cells to include in
  this analysis.

- scrublet_params:

  List. Parameter list, see
  [`params_scrublet_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_scrublet_gpu.md).

- seed:

  Integer. Seed for reproducibility purposes.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

- streaming:

  Boolean. Shall the data be streamed for the HVG calculations.

- return_combined_pca:

  Boolean. Shall the generated PCA be returned.

- return_pairs:

  Boolean. Shall the parents of the simulated cells be returned.

## Value

A list with

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

## References

Wolock, et al., Cell Syst, 2020
