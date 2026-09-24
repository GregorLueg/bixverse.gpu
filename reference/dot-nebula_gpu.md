# GPU NEBULA implementation

Shared body of the
[`nebula_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/nebula_gpu_sc.md)
methods. Mirrors the CPU
[`bixverse::nebula_sc()`](https://gregorlueg.github.io/bixverse/reference/nebula_sc.html)
method step for step and only swaps the Rust call.

## Usage

``` r
.nebula_gpu(
  object,
  subject_col,
  design,
  coef,
  contrast,
  genes_to_use,
  offset,
  nebula_params,
  .verbose
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

A `ScNebula` class.
