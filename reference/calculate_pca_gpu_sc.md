# GPU-accelerated PCA for single cell

This function will run sparse, randomised SVD while running several of
the large matrix multiplications on GPU for improved speed. This also
means you will have to provide the necessary VRAM for your data set.
This version only works on the `"rna"` modality.

## Usage

``` r
calculate_pca_gpu_sc(
  object,
  no_pcs,
  pca_params = bixverse::params_sc_pca(),
  hvg = NULL,
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` class

- no_pcs:

  Integer. Number of PCs to calculate.

- pca_params:

  Named list. Controls the parameters to be used for the PCA calculation
  which is single cell-specific, see
  [`params_sc_pca()`](https://gregorlueg.github.io/bixverse/reference/params_sc_pca.html).

- hvg:

  Optional integer. If you want to provide your own HVG genes.
  Otherwise, the function will default to what is found in
  [`bixverse::get_hvg()`](https://gregorlueg.github.io/bixverse/reference/get_hvg.html).
  Please provide 1-indexed genes here! If you provide these, the
  internal HVG will be overwritten.

- seed:

  Integer. Controls reproducibility. Only relevant if
  `randomised_svd = TRUE`.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

The function will add the PCA factors, loadings and singular values to
the object cache in memory.
