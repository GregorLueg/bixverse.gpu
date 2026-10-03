# Run t-SNE on a SingleCells object (GPU)

GPU-accelerated counterpart to
[`bixverse::tsne_sc()`](https://gregorlueg.github.io/bixverse/reference/tsne_sc.html).
Runs
[`tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu.md)
on an embedding pulled from the object. The kNN runs on the GPU; with
the default `"fft_3k_gpu"` the optimiser does too.

t-SNE derives the number of neighbours from `perplexity` on the Rust
side (the usual `3 * perplexity` convention). To avoid a silent mismatch
with the cached kNN, `use_knn` defaults to `FALSE`: every call generates
a fresh GPU kNN sized to the requested perplexity. Handy for sweeping
perplexities since the kNN is cheap on GPU.

## Usage

``` r
tsne_gpu_sc(
  object,
  use_knn = FALSE,
  embd_to_use = "pca",
  slot_name = "tsne",
  no_embd_to_use = NULL,
  modality = c("rna", "adt", "wnn"),
  n_dim = 2L,
  perplexity = 20,
  approx_type = c("fft_3k_gpu", "bh", "bh_qd", "fft", "fft_3k"),
  knn_method = c("ivf", "exhaustive", "nndescent"),
  nn_params = params_nn_gpu(),
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  use_high_precision = NULL,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- use_knn:

  Boolean. Defaults to `FALSE`. Set to `TRUE` to reuse the cached kNN;
  only sensible when the stored `k` is at least `3 * perplexity`.

- embd_to_use:

  String. The embedding to use for t-SNE. Must be available in the
  object for the chosen modality.

- slot_name:

  String. The name of this embedding within the object. Defaults to
  `"tsne"`.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL`,
  all will be used.

- modality:

  String. On which modality to run t-SNE. One of
  `c("rna", "adt", "wnn")`. The two latter options are only available on
  `SingleCellsMultiModal`.

- n_dim:

  Integer. Number of t-SNE dimensions. Currently only `2L` is supported.
  Defaults to `2L`.

- perplexity:

  Numeric. Perplexity parameter. Typical values between 5 and 50.
  Defaults to `20.0`.

- approx_type:

  String. Approximation method. One of `"fft_3k_gpu"` (three-kernel FFT
  on the GPU, the default), `"bh"` (Barnes-Hut), `"bh_qd"`
  (quick-and-dirty Barnes-Hut, depth-capped tree), `"fft"` or `"fft_3k"`
  (CPU FFT interpolation, four or three kernels). The CPU FFT variants
  are Unix-only. See
  [`tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu.md).

- knn_method:

  String. GPU (approximate) nearest neighbour method. One of
  `c("ivf", "exhaustive", "nndescent")`. Default is `"ivf"` here, as it
  deals usually better with the high k in tSNE.

- nn_params:

  Named list. GPU kNN parameters, see
  [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

- tsne_params:

  Named list. t-SNE (GPU) parameters, see
  [`params_tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_tsne_gpu.md).

- seed:

  Integer. For reproducibility.

- use_high_precision:

  Optional boolean. Fine-grained fp32 vs fp64 control. GPU kNN is always
  fp32. Ignored for `"fft_3k_gpu"`.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

The object with a `"tsne"` embedding added. If the requested embedding
is missing, returns the object unchanged with a warning.

## See also

[`tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu.md),
[`bixverse::tsne_sc()`](https://gregorlueg.github.io/bixverse/reference/tsne_sc.html),
[`umap_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu_sc.md)
