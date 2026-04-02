# Predict new data using a trained parametric UMAP model

Runs forward inference through the trained encoder network. The
prediction automatically uses whichever backend (GPU or CPU) the model
was trained on.

## Usage

``` r
rs_parametric_umap_predict(model, data)
```

## Arguments

- model:

  External pointer to the trained parametric UMAP model, as returned by
  `rs_parametric_umap`.

- data:

  Numerical matrix. New data of dimensions samples x features. The
  number of features must match the training data.

## Value

Numerical matrix of dimensions samples x n_dim with the predicted
embeddings.
