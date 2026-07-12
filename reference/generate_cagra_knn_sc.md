# Generate CAGRA GPU kNN data for single cells

This function generates a `SingleCellNearestNeighbour` object using the
CAGRA (CUDA-Accelerated Graph Retrieval Approximation) algorithm via the
`bixverse.gpu` package. CAGRA first builds a dense NNDescent graph, then
prunes it into a sparser navigational graph optimised for beam-search
traversal. Two retrieval modes are available: direct extraction from the
NNDescent graph (`extract_knn = TRUE`), which is faster but slightly
less precise, or beam search over the pruned CAGRA graph
(`extract_knn = FALSE`), which is slower but yields higher recall. This
function is the CAGRA counterpart of
[`generate_knn_sc()`](https://gregorlueg.github.io/bixverse/reference/generate_knn_sc.html).

## Usage

``` r
generate_cagra_knn_sc(
  object,
  embd_to_use = "pca",
  cells_to_use = NULL,
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  cagra_params = params_sc_cagra(),
  extract_knn = TRUE,
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- embd_to_use:

  String. The embedding to use. Whichever you choose, it needs to be
  part of the object for the selected modality.

- cells_to_use:

  Optional string vector. Cell names to include. If `NULL` all cells in
  the object will be used.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL` all
  will be used.

- modality:

  String. One of `c("rna", "adt")`. You can only use `"adt"` on
  `SingleCellsMultiModal` class.

- cagra_params:

  List. Output of
  [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md).

- extract_knn:

  Logical. If `TRUE`, extracts the kNN graph directly from the NNDescent
  result (faster, slightly lower precision). If `FALSE`, runs beam
  search over the pruned CAGRA graph (slower, higher precision).

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

Initialised `sc_knn` with the kNN data.
