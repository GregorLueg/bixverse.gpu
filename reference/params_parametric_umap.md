# Wrapper function to generate parametric UMAP parameters

Wrapper function to generate parametric UMAP parameters

## Usage

``` r
params_parametric_umap(
  local_connectivity = 1,
  bandwidth = 1e-05,
  mix_weight = 1,
  hidden_layers = c(128L, 64L, 32L),
  lr = 0.001,
  corr_weight = 0,
  n_epochs = 500L,
  batch_size = 256L,
  neg_sample_rate = 5L
)
```

## Arguments

- local_connectivity:

  Numeric. Number of nearest neighbours assumed to be at distance zero.
  Defaults to `1.0`.

- bandwidth:

  Numeric. Convergence tolerance for smooth kNN distance binary search.
  Defaults to `1e-5`.

- mix_weight:

  Numeric. Balance between fuzzy union and directed graph during
  symmetrisation. Defaults to `1.0`.

- hidden_layers:

  Integer vector. Hidden layer sizes for the MLP encoder. Defaults to
  `c(128L, 64L, 32L)`.

- lr:

  Numeric. Learning rate for the neural network optimiser. Defaults to
  `0.001`.

- corr_weight:

  Numeric. Coefficient for the negative Pearson correlation loss that
  encourages similar distances in embedding and original space. Defaults
  to `0.0`.

- n_epochs:

  Integer. Number of training epochs. Defaults to `500L`.

- batch_size:

  Integer. Training batch size. Defaults to `256L`.

- neg_sample_rate:

  Integer. Number of negative samples per positive edge. Defaults to
  `5L`.

## Value

A list with the parametric UMAP parameters.
