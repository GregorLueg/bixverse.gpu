# Wrapper function for parameters for GPU NEBULA

GPU counterpart to
[`bixverse::params_nebula()`](https://gregorlueg.github.io/bixverse/reference/params_nebula.html).
Same knobs and defaults, minus `reml`: the device fit does not implement
it. Stage two of NEBULA, the per-gene penalised fits, runs in `f32` on
the device and is finished on the host in `f64`, so the estimates sit
close to the CPU ones rather than on them.

## Usage

``` r
params_nebula_gpu(
  nebula_method = c("ln", "hl"),
  min_sigma = 1e-04,
  min_phi = 1e-04,
  max_sigma = 10,
  max_phi = 1000,
  cutoff_cell = 20,
  kappa = 800,
  cpc = 0.005,
  mincp = 5L,
  eps = 1e-06,
  gene_batch_size = 1000L,
  shrink_dispersion = TRUE
)
```

## Arguments

- nebula_method:

  String. Which variant to run. NEBULA downgrades `"ln"` to `"hl"` below
  30 cells per subject, as the R package does. One of `c("ln", "hl")`.
  Defaults to `"ln"`.

- min_sigma:

  Numeric. Lower bound on the subject-level overdispersion. Defaults to
  `1e-04`.

- min_phi:

  Numeric. Lower bound on the cell-level overdispersion. Defaults to
  `1e-04`.

- max_sigma:

  Numeric. Upper bound on the subject-level overdispersion. Defaults to
  `10.0`.

- max_phi:

  Numeric. Upper bound on the cell-level overdispersion. Defaults to
  `1000.0`.

- cutoff_cell:

  Numeric. Refit both overdispersions when the product of the cells per
  subject and the estimated `phi` falls below this. Defaults to `20.0`.

- kappa:

  Numeric. Threshold on NEBULA's `kappa_obs` above which the
  subject-level overdispersion from stage one is trusted as is. Defaults
  to `800.0`.

- cpc:

  Numeric. Drop a gene whose mean count per cell is at most this.
  Defaults to `0.005`.

- mincp:

  Integer. Drop a gene expressed in fewer than this many cells. Defaults
  to `5L`.

- eps:

  Numeric. Absolute stopping tolerance for the optimiser. Defaults to
  `1e-06`.

- gene_batch_size:

  Integer. Genes read and fitted per batch. Bounds how much of the store
  is resident at once and changes nothing about the answer, since NEBULA
  is gene-independent. Defaults to `1000L`.

- shrink_dispersion:

  Boolean. Shrink the cell-level overdispersions towards an empirical
  Bayes prior once the sweep is done. Defaults to `TRUE`.

## Value

A named list with the following elements:

- nebula_method - String. Which variant to run. NEBULA downgrades `"ln"`
  to `"hl"` below 30 cells per subject, as the R package does. One of
  `c("ln", "hl")`. Defaults to `"ln"`.

- min_sigma - Numeric. Lower bound on the subject-level overdispersion.
  Defaults to `1e-04`.

- min_phi - Numeric. Lower bound on the cell-level overdispersion.
  Defaults to `1e-04`.

- max_sigma - Numeric. Upper bound on the subject-level overdispersion.
  Defaults to `10.0`.

- max_phi - Numeric. Upper bound on the cell-level overdispersion.
  Defaults to `1000.0`.

- cutoff_cell - Numeric. Refit both overdispersions when the product of
  the cells per subject and the estimated `phi` falls below this.
  Defaults to `20.0`.

- kappa - Numeric. Threshold on NEBULA's `kappa_obs` above which the
  subject-level overdispersion from stage one is trusted as is. Defaults
  to `800.0`.

- cpc - Numeric. Drop a gene whose mean count per cell is at most this.
  Defaults to `0.005`.

- mincp - Integer. Drop a gene expressed in fewer than this many cells.
  Defaults to `5L`.

- eps - Numeric. Absolute stopping tolerance for the optimiser. Defaults
  to `1e-06`.

- gene_batch_size - Integer. Genes read and fitted per batch. Bounds how
  much of the store is resident at once and changes nothing about the
  answer, since NEBULA is gene-independent. Defaults to `1000L`.

- shrink_dispersion - Boolean. Shrink the cell-level overdispersions
  towards an empirical Bayes prior once the sweep is done. Defaults to
  `TRUE`.

## References

He, et al., Commun Biol, 2021
