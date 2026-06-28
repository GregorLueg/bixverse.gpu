# Find CAGRA GPU-accelerated neighbours for single cells

This function generates kNN data using the CAGRA (CUDA-Accelerated Graph
Retrieval Approximation) algorithm on the wgpu backend via the
`bixverse.gpu` package. CAGRA first builds a dense NNDescent graph, then
prunes it into a sparser navigational graph optimised for beam-search
traversal. Two retrieval modes are available: direct extraction from the
NNDescent graph (`extract_knn = TRUE`), which is faster but slightly
less precise, or beam search over the pruned CAGRA graph
(`extract_knn = FALSE`), which is slower but yields higher recall.
Subsequently, the kNN data is used to generate an sNN igraph for
downstream clustering.

## Usage

``` r
find_neighbours_cagra_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  cagra_params = params_sc_cagra(),
  extract_knn = FALSE,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- embd_to_use:

  String. The embedding to use.

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
  result. If `FALSE`, runs beam search over the pruned CAGRA graph. The
  extraction is faster, but creates a lower quality kNN graph.

- snn_params:

  List. Output of
  [`bixverse::params_sc_neighbours()`](https://rdrr.io/pkg/bixverse/man/params_sc_neighbours.html).
  The kNN graph-related parameters will be ignored.

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean. Controls verbosity.

## Value

The object with added kNN matrix and sNN graph in the selected modality
slot.

## Note

Euclidean distance calculates the squared Euclidean distance for speed.
