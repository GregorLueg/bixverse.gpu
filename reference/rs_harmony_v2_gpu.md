# Harmony batch correction in Rust (version 2, GPU-accelerated)

**\[experimental\]** This function implements the GPU-accelerated
version 2 Harmony algorithm from Patikas, et al., 2026. Only a single
batch covariate is supported.

## Usage

``` r
rs_harmony_v2_gpu(pca, harmony_params, batch_labels, seed, verbose)
```

## Arguments

- pca:

  Numerical matrix, i.e., the PCA matrix you want to correct.

- harmony_params:

  List. The parameters for the Harmony (v2) GPU algorithm.

- batch_labels:

  List. Must contain exactly one element: a 0-indexed integer vector
  representing the batch effects you wish to regress out.

- seed:

  Integer. Seed for reproducibility purposes.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

The batch-corrected Harmony (v2) embedding space.
