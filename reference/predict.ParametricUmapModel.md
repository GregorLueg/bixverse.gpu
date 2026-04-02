# Predict embeddings for new data using a trained parametric UMAP model

Predict embeddings for new data using a trained parametric UMAP model

## Usage

``` r
# S3 method for class 'ParametricUmapModel'
predict(object, newdata, ...)
```

## Arguments

- object:

  A `ParametricUmapModel` object.

- newdata:

  Numerical matrix or data frame. New data of shape samples x features.
  Must have the same number of features as the training data.

- ...:

  Ignored.

## Value

A numerical matrix with dimensions samples x n_dim.
