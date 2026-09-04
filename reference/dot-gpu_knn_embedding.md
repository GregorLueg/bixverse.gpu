# Pull an embedding out of a single cell object for a GPU kNN search

Pull an embedding out of a single cell object for a GPU kNN search

## Usage

``` r
.gpu_knn_embedding(object, embd_to_use, cells_to_use, no_embd_to_use, modality)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- embd_to_use:

  String. The embedding to use.

- cells_to_use:

  Optional string vector. Cell names to include.

- no_embd_to_use:

  Optional integer. Number of dimensions to keep.

- modality:

  String. One of `c("rna", "adt")`.

## Value

The embedding matrix, or `NULL` if the embedding is not in the object.
