# Parents of a GPU manifold embedding (UMAP, t-SNE)

Builds the `from` argument for
[`bixverse::set_embedding()`](https://gregorlueg.github.io/bixverse/reference/set_embedding.html)
so the resulting embedding joins the provenance chain rather than being
recorded as a root. The source embedding is read from `cache_modality`
while the result is written under `modality`, and the two differ for
`"wnn"`, so both parents are spelled out modality qualified.

Kept local rather than reaching for the equivalent `bixverse` internal:
four lines are not worth coupling to an unexported function.

## Usage

``` r
.manifold_from_gpu(embd_to_use, cache_modality, modality, has_knn)
```

## Arguments

- embd_to_use:

  String. Name of the source embedding.

- cache_modality:

  String. Modality the source embedding was read from.

- modality:

  String. Modality the result is written to.

- has_knn:

  Boolean. Whether a cached kNN fed the manifold.

## Value

Character vector of parent artefact names.
