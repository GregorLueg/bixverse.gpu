# Run single-run NMF on the GPU over single cell or meta cell data

GPU counterpart of
[`bixverse::nmf_sc()`](https://gregorlueg.github.io/bixverse/reference/nmf_sc.html).
Runs one HALS NMF on a chosen subset of cells and genes. The counts are
uploaded once and the whole HALS loop runs on the WGPU backend; the
NNDSVD initialisation stays on the CPU.

For `SingleCells` the counts are streamed from the Rust binary files,
for `MetaCells` the in-memory sparse counts are used. Params, result
class and downstream code are identical to the CPU version.

A single run is here for parity rather than speed. The GPU pays off when
the same matrix serves many solves, so reach for
[`consensus_nmf_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/consensus_nmf_gpu_sc.md)
or
[`nmf_k_sweep_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/nmf_k_sweep_gpu_sc.md)
if you want the speed-up.

## Usage

``` r
nmf_gpu_sc(
  object,
  k,
  cell_ids = NULL,
  gene_ids = NULL,
  preprocessing = "none",
  use_second_layer = TRUE,
  nmf_hals_params = bixverse::params_nmf_hals(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `MetaCells` class from `bixverse`.

- k:

  Integer. Number of latent factors to return. At most 128, see
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

- seed:

  Integer. Random seed for initialisation.

- .verbose:

  Boolean or integer. Controls verbosity. `FALSE` -\> quiet, `TRUE` or
  `1L` -\> normal verbosity, `2L` -\> detailed verbosity.

## Value

An `NmfResult` object, the same class
[`bixverse::nmf_sc()`](https://gregorlueg.github.io/bixverse/reference/nmf_sc.html)
returns.
