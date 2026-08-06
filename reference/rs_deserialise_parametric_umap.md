# Deserialises raw bytes to a trained UMAP model.

**\[experimental\]** Deserialises a trained parametric UMAP model from
raw bytes and returns an R object.

## Usage

``` r
rs_deserialise_parametric_umap(bytes)
```

## Arguments

- bytes:

  The raw byte sequence. The leading byte tags the backend the model was
  trained on: `0` for wgpu, `1` for the flex CPU backend.

## Value

An external pointer to the restored model, for use with
`rs_parametric_umap_predict`.
