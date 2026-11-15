# Run PHATE on a SingleCells object (GPU)

GPU-accelerated counterpart to
[`bixverse::phate_sc()`](https://gregorlueg.github.io/bixverse/reference/phate_sc.html).
Runs
[`phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/phate_gpu.md)
on an embedding pulled from the object. The kNN search and the spectral
landmark selection run on the GPU, the diffusion and MDS on the CPU.

PHATE searches `6 * k + 5` neighbours (35 at the default `k = 5L`), far
wider than the cached kNN usually is. So unlike
[`bixverse::phate_sc()`](https://gregorlueg.github.io/bixverse/reference/phate_sc.html),
`use_knn` defaults to `FALSE` and every call builds a fresh GPU kNN of
the right width. A cached graph that is too narrow gives a warning.

## Usage

``` r
phate_gpu_sc(
  object,
  use_knn = FALSE,
  embd_to_use = "pca",
  slot_name = "phate",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  k = 5L,
  knn_method = c("ivf", "exhaustive", "nndescent"),
  nn_params = params_nn_gpu(),
  phate_params = params_phate_gpu(),
  seed = 42L,
  backend = c("wgpu", "mlx"),
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- use_knn:

  Boolean. Defaults to `FALSE`. Set to `TRUE` to reuse the cached kNN;
  only sensible when the stored `k` is at least `6 * k + 5`.

- embd_to_use:

  String. The embedding to use for PHATE. Must be available in the
  object for the chosen modality.

- slot_name:

  String. The name of this embedding within the object. Defaults to
  `"phate"`.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL`,
  all will be used.

- modality:

  String. On which modality to run PHATE. One of
  `c("rna", "adt", "wnn")`. The two latter options are only available on
  `SingleCellsMultiModal`.

- n_dim:

  Integer. Number of PHATE dimensions. Defaults to `2L`.

- k:

  Integer. Number of nearest neighbours that set the kernel bandwidth.
  Defaults to `5L`.

- knn_method:

  String. GPU (approximate) nearest neighbour method. One of
  `c("ivf", "exhaustive", "nndescent")`. Defaults to `"ivf"`, as the
  `6 * k + 5` wide search is past the width where NN-descent keeps up.

- nn_params:

  Named list. GPU kNN parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- phate_params:

  Named list. PHATE (GPU) parameters, see
  [`params_phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_phate_gpu.md).

- seed:

  Integer. For reproducibility.

- backend:

  String. One of `c("wgpu", "mlx")`. `"mlx"` moves the kNN search and
  the spectral landmark selection to MLX, see
  [`phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/phate_gpu.md).

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

The object with a `"phate"` embedding added. If the requested embedding
is missing, returns the object unchanged with a warning.

## See also

[`phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/phate_gpu.md),
[`bixverse::phate_sc()`](https://gregorlueg.github.io/bixverse/reference/phate_sc.html),
[`umap_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu_sc.md)
