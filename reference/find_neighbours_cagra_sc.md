# Find CAGRA GPU-accelerated neighbours for single cells

This function generates kNN data using the CAGRA (CUDA-Accelerated Graph
Retrieval Approximation) algorithm on the wgpu backend via the
`bixverse.gpu` package. CAGRA first builds a dense NNDescent graph, then
prunes it into a sparser navigational graph optimised for beam-search
traversal. Two retrieval modes are available: direct extraction from the
NNDescent graph (`extract_knn = TRUE`), which is faster but slightly
less precise, or beam search over the pruned CAGRA graph
(`extract_knn = FALSE`), which is slower but yields higher recall. CAGRA
tends to perform well on high-dimensional embeddings and very large data
sets. Subsequently, the kNN data is used to generate an sNN igraph for
downstream clustering. As with
[`find_neighbours_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_gpu_sc.md),
this function lives in a separate package so that users without GPU
hardware are not required to install the GPU dependencies.

## Usage

``` r
find_neighbours_cagra_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  cagra_params = params_sc_cagra(),
  extract_knn = TRUE,
  snn_params = params_sc_neighbours(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` class.

- embd_to_use:

  String. The embedding to use. Whichever you choose, it needs to be
  part of the object.

- no_embd_to_use:

  Optional integer. Number of embedding dimensions to use. If `NULL` all
  will be used.

- cagra_params:

  List. Output of
  [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md).
  A list with the following items:

  - k_query - Integer. Number of nearest neighbours to return in the
    final result.

  - ann_dist - String. Distance metric; one of
    `c("euclidean", "cosine")`.

  - k - Optional integer. Final node degree of the pruned CAGRA
    navigational graph. Controls the sparsity of the search graph;
    higher values improve recall but increase memory usage. If `NULL`,
    defaults to `30`.

  - k_build - Optional integer. Number of neighbours during the
    NNDescent build phase before CAGRA pruning. If `NULL`, defaults to
    `1.5 * k`.

  - refine_sweeps - Integer. Number of refinement sweeps during graph
    construction. More sweeps improve graph quality at the cost of build
    time.

  - max_iters - Optional integer. Maximum iterations for the NNDescent
    rounds. If `NULL`, determined automatically.

  - n_trees - Optional integer. Number of trees in the initial
    GPU-accelerated random projection forest used to seed NNDescent. If
    `NULL`, determined automatically.

  - delta - Numeric. Early-stopping criterion for NNDescent; iterations
    terminate when fewer than `delta` fraction of neighbours change.

  - rho - Optional numeric. Sampling rate during NNDescent iterations.
    Lower values speed up construction at the cost of graph quality. If
    `NULL`, determined automatically.

  - beam_width - Optional integer. Beam width during graph search.
    Larger beams improve recall but slow down querying. If `NULL`,
    determined automatically.

  - max_beam_iters - Optional integer. Maximum beam search iterations.
    If `NULL`, determined automatically.

  - n_entry_points - Optional integer. Number of entry points into the
    CAGRA graph. If `NULL`, determined automatically.

- extract_knn:

  Logical. If `TRUE`, extracts the kNN graph directly from the NNDescent
  result (faster, slightly lower precision). If `FALSE`, runs beam
  search over the pruned CAGRA graph (slower, higher precision).

- snn_params:

  List. Output of
  [`bixverse::params_sc_neighbours()`](https://rdrr.io/pkg/bixverse/man/params_sc_neighbours.html).
  Controls sNN graph construction. The relevant items are:

  - full_snn - Boolean. Whether to generate edges between all cells
    rather than only between neighbours.

  - pruning - Numeric. Weights below this threshold are set to 0 in the
    sNN graph.

  - snn_similarity - String. One of `c("rank", "jaccard")`. Defines how
    the sNN edge weights are calculated.

- seed:

  Integer. For reproducibility.

- .verbose:

  Boolean. Controls verbosity.

## Value

The object with added kNN matrix and sNN graph.
