# Run stabilised (multi-run) NMF on the GPU over single cell or meta cell data

GPU counterpart of
[`bixverse::stabilised_nmf_sc()`](https://gregorlueg.github.io/bixverse/reference/stabilised_nmf_sc.html).
Runs `n_runs` HALS NMF with random initialisations seeded by `seed + i`.
The `nmf_init` field in `nmf_hals_params` is ignored; random init is
always used.

The counts upload once and serve every restart, but the restarts
themselves run one after the other on the single device, where the CPU
version spreads them across cores. On a small matrix the CPU can still
win.

## Usage

``` r
stabilised_nmf_gpu_sc(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  n_runs = 30L,
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `MetaCells` class from `bixverse`.

- k:

  Integer. Number of latent factors per run. At most 128, see
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

- n_runs:

  Integer. Number of random restarts.

- seed:

  Integer. Random seed for initialisation.

- .verbose:

  Boolean or integer. Controls verbosity. `FALSE` -\> quiet, `TRUE` or
  `1L` -\> normal verbosity, `2L` -\> detailed verbosity.

## Value

A `StabilisedNmfResult` object, the same class
[`bixverse::stabilised_nmf_sc()`](https://gregorlueg.github.io/bixverse/reference/stabilised_nmf_sc.html)
returns.
