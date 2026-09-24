# Wrapper function to generate UMAP parameters (GPU version)

Wrapper function to generate UMAP parameters (GPU version)

## Usage

``` r
params_umap_gpu(
  local_connectivity = 1,
  bandwidth = 1e-05,
  mix_weight = 1,
  lr = 1,
  n_epochs = NULL,
  neg_sample_rate = 5L,
  gamma = 1,
  optimiser = c("adam_gpu", "adam_parallel", "sgd", "adam"),
  init = c("spectral", "pca", "random"),
  randomised = FALSE
)
```

## Arguments

- local_connectivity:

  Numeric. Number of nearest neighbours assumed to be at distance zero.
  Defaults to `1.0`.

- bandwidth:

  Numeric. Convergence tolerance for smooth kNN distance binary search.
  Defaults to `1e-05`.

- mix_weight:

  Numeric. Balance between fuzzy union and directed graph during
  symmetrisation. Defaults to `1.0`.

- lr:

  Numeric. Learning rate. Defaults to `1.0`.

- n_epochs:

  Integer or `NULL`. Number of optimisation epochs. Defaults to `NULL`.

- neg_sample_rate:

  Integer. Number of negative samples per positive sample. Defaults to
  `5L`.

- gamma:

  Numeric. Repulsion strength. Defaults to `1.0`.

- optimiser:

  String. The optimiser for the embedding. One of
  `c("adam_gpu", "adam_parallel", "sgd", "adam")`. Defaults to
  `"adam_gpu"`.

- init:

  String. Embedding initialisation method. One of
  `c("spectral", "pca", "random")`. Defaults to `"spectral"`.

- randomised:

  Boolean. Use randomised SVD for PCA initialisation. Defaults to
  `FALSE`.

## Value

A named list with the following elements:

- local_connectivity - Numeric. Number of nearest neighbours assumed to
  be at distance zero. Defaults to `1.0`.

- bandwidth - Numeric. Convergence tolerance for smooth kNN distance
  binary search. Defaults to `1e-05`.

- mix_weight - Numeric. Balance between fuzzy union and directed graph
  during symmetrisation. Defaults to `1.0`.

- lr - Numeric. Learning rate. Defaults to `1.0`.

- n_epochs - Integer or `NULL`. Number of optimisation epochs. Defaults
  to `NULL`.

- neg_sample_rate - Integer. Number of negative samples per positive
  sample. Defaults to `5L`.

- gamma - Numeric. Repulsion strength. Defaults to `1.0`.

- optimiser - String. The optimiser for the embedding. One of
  `c("adam_gpu", "adam_parallel", "sgd", "adam")`. Defaults to
  `"adam_gpu"`.

- init - String. Embedding initialisation method. One of
  `c("spectral", "pca", "random")`. Defaults to `"spectral"`.

- randomised - Boolean. Use randomised SVD for PCA initialisation.
  Defaults to `FALSE`.
