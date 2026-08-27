# Sweep k for consensus NMF on the GPU over single cell or meta cell data

GPU counterpart of
[`bixverse::nmf_k_sweep_sc()`](https://gregorlueg.github.io/bixverse/reference/nmf_k_sweep_sc.html).
Runs the consensus step across a range of ranks and reports stability
against reconstruction error, keeping no factors. Pick the last `k`
before stability falls away while the error curve is still coming down,
then fit there with
[`consensus_nmf_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/consensus_nmf_gpu_sc.md).

## Usage

``` r
nmf_k_sweep_gpu_sc(
  object,
  k_range,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  nmf_consensus_params = bixverse::params_nmf_consensus(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `MetaCells` class from `bixverse`.

- k_range:

  Integer vector. The ranks to evaluate. Every entry at least 2 and at
  most 128, see
  [NMF_GPU_MAX_RANK](https://gregorlueg.github.io/bixverse.gpu/reference/NMF_GPU_MAX_RANK.md).

- cell_ids:

  Optional character. Cell ids (or meta cell ids) to restrict the NMF
  to. If `NULL`, uses
  [`bixverse::get_cells_to_keep()`](https://gregorlueg.github.io/bixverse/reference/get_cells_to_keep.html)
  for `SingleCells` and all meta cells for `MetaCells`.

- gene_ids:

  Optional character. Gene ids to restrict the NMF to. If `NULL`, uses
  [`bixverse::get_hvg()`](https://gregorlueg.github.io/bixverse/reference/get_hvg.html)
  on the object.

- preprocessing:

  String. One of `c("none", "sd", "sqrt_sd")`.

- use_second_layer:

  Boolean. If `TRUE`, runs NMF on the normalised counts (recommended);
  if `FALSE`, on the raw counts.

- nmf_hals_params:

  List, see
  [`bixverse::params_nmf_hals()`](https://gregorlueg.github.io/bixverse/reference/params_nmf_hals.html).
  The `nmf_init` field is ignored, restarts always use random
  initialisation.

- nmf_consensus_params:

  List, see
  [`bixverse::params_nmf_consensus()`](https://gregorlueg.github.io/bixverse/reference/params_nmf_consensus.html).

- n_runs:

  Integer. Number of restarts. At least 2.

- seed:

  Integer. Random seed for initialisation.

- .verbose:

  Boolean or integer. Controls verbosity. `FALSE` -\> quiet, `TRUE` or
  `1L` -\> normal verbosity, `2L` -\> detailed verbosity.

## Value

An `NmfKSweepResult`, which is a data.table with one row per `k`.

## Details

This is the shape the GPU path is really for. The counts upload once and
serve all `length(k_range) * n_runs` solves, where the CPU pays full
memory traffic over the matrix for every one of them. The scratch is
sized once at the largest rank in `k_range`.

It is a diagnostic, so it leaves the object alone and hands the result
back directly. [`plot()`](https://rdrr.io/r/graphics/plot.default.html)
on it gives you the two curves.

## References

Kotliar et al., eLife, 2019
