# Run NEBULA on single cells on the GPU

GPU counterpart of
[`bixverse::nebula_sc()`](https://gregorlueg.github.io/bixverse/reference/nebula_sc.html).
Stage two of NEBULA, the per-gene penalised fits, runs on the WGPU
backend in `f32` and is finished on the host in `f64`. Cell ordering,
gene batching, the dispersion shrinkage and the Wald test are the CPU
code. Expect estimates close to the CPU ones, not identical to them.

REML is not implemented on the device, so
[`params_nebula_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nebula_gpu.md)
does not carry it. Everything else, the design handling, the result
class and the downstream code, is identical to the CPU version.

## Usage

``` r
nebula_gpu_sc(
  object,
  subject_col,
  design,
  coef = NULL,
  contrast = NULL,
  genes_to_use = NULL,
  offset = NULL,
  nebula_params = params_nebula_gpu(),
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- subject_col:

  String. The column in the obs table holding the subject (donor)
  identifier. This is what the random effect is over.

- design:

  Formula. The experimental design, evaluated against the obs table,
  e.g. `~ condition` or `~ condition + age`. Include the intercept.

- coef:

  Optional integer or character. Which coefficient of the design the
  Wald test reports, as a 1-based column position or a column name.
  Defaults to the last column.

- contrast:

  Optional numeric vector. One weight per design column. Mutually
  exclusive with `coef`.

- genes_to_use:

  Optional character vector. The genes to fit. Defaults to every gene in
  the object, which is usually too many.

- offset:

  Optional numeric vector. Strictly positive scaling factor per cell,
  aligned to the cells that survive the design. Defaults to `NULL`,
  which uses the library sizes.

- nebula_params:

  A list, see
  [`params_nebula_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nebula_gpu.md).
  The list has the following parameters:

  - nebula_method - String. One of `c("ln", "hl")`.

  - min_sigma, max_sigma - Numeric. Bounds on the subject-level
    overdispersion.

  - min_phi, max_phi - Numeric. Bounds on the cell-level overdispersion.

  - cutoff_cell - Numeric. When to refit both overdispersions.

  - kappa - Numeric. When to trust the stage-one subject overdispersion.

  - cpc - Numeric. Minimum mean count per cell for a gene to be tested.

  - mincp - Integer. Minimum number of cells expressing a gene.

  - eps - Numeric. Optimiser stopping tolerance.

  - gene_batch_size - Integer. Genes read and fitted per batch.

  - shrink_dispersion - Boolean. Empirical Bayes shrinkage of the
    cell-level overdispersions.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

A `ScNebula` class, see `bixverse:::new_sc_nebula_res()`, with

- results - data.table. One row per gene that survived NEBULA's
  expression filter, with the Wald test and both overdispersions.

- coefficients - Numeric matrix of genes x coefficients.

- se - Numeric matrix of genes x coefficients.

- params - List. The parameters the run used.

## References

He, et al., Commun Biol, 2021
