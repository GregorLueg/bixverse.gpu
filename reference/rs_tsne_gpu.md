# tSNE implementation

**\[experimental\]** Wraps the tSNE implementation in manifolds-rs. The
kNN search runs on the GPU. `"fft_3k_gpu"` runs the optimiser on the GPU
as well; `"bh"`, `"fft"` and `"fft_3k"` keep it on the CPU.

## Usage

``` r
rs_tsne_gpu(
  embd,
  n_dim,
  perplexity,
  approx_type,
  tsne_params,
  seed,
  use_high_precision,
  verbose
)
```

## Arguments

- embd:

  Numerical matrix. The data to use to generate the embeddings. Should
  be of dimensions samples x features.

- n_dim:

  Integer. Number of tSNE dimensions to return. Needs to be two, others
  are not supported.

- perplexity:

  Numeric. The tSNE perplexity parameter.

- approx_type:

  String. One of `c("fft_3k_gpu", "bh", "fft", "fft_3k")`. Which
  repulsive-force approximation to use. `"fft"` and `"fft_3k"` need FFTW
  and are not available on Windows.

- tsne_params:

  Named list. List that contains all of the key parameters for the tSNE
  generation.

- seed:

  Integer. Seed for reproducibility.

- use_high_precision:

  Optional logical. Controls `fp32` vs `fp64`. If `NULL` will use
  sensible default thresholding. Ignored for `"fft_3k_gpu"`, which
  always runs in `fp32`.

- verbose:

  Integer. If `0L` -\> silent or `1L` for normal verbosity; `2L` for
  detailed verbosity.

## Value

The tSNE embeddings.
