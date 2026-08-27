# Run consensus NMF on the GPU over single cell or meta cell data

GPU counterpart of
[`bixverse::consensus_nmf_sc()`](https://gregorlueg.github.io/bixverse/reference/consensus_nmf_sc.html).
Runs `n_runs` HALS restarts on the device, pools their components, drops
unstable ones by local density, k-means clusters the survivors and
refits the partner factor against the per-cluster median.

Prefer this over
[`stabilised_nmf_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/stabilised_nmf_gpu_sc.md),
which picks the lowest-loss restart. Use
[`nmf_k_sweep_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/nmf_k_sweep_gpu_sc.md)
first if you do not already know `k`.

## Usage

``` r
consensus_nmf_gpu_sc(
  object,
  k,
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

- k:

  Integer. Number of latent factors. At least 2, at most 128, see
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

A `ConsensusNmfResult` object, the same class
[`bixverse::consensus_nmf_sc()`](https://gregorlueg.github.io/bixverse/reference/consensus_nmf_sc.html)
returns.

## Details

Only the restarts move to the GPU. The pooling, the density filter, the
k-means and the silhouette all run on the CPU, shared with the bixverse
implementation, so the speed-up tracks how much of the run the solves
own.

The restart factors are dense and all held at once, so budget for
`n_runs` times `k` times the cell count on top of the counts themselves.

If the density filter leaves fewer than `k` components, or a cluster
comes out empty, the run errors rather than returning a partial answer.
Raise `density_threshold` (2 switches the filter off) or increase
`n_runs`.

## References

Kotliar et al., eLife, 2019
