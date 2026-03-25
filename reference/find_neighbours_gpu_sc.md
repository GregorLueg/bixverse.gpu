# Find GPU-accelerated neighbours for single cells (exhaustive / IVF)

This function generates kNN data using GPU-accelerated algorithms via
the `bixverse.gpu` package. Two methods are available: `"exhaustive"`
performs an exact brute-force search on the GPU, which is precise but
scales quadratically; `"ivf"` builds an inverted file index that
partitions the embedding space into Voronoi cells and probes only a
subset at query time, trading a small amount of precision for
considerably faster search on larger data sets. Subsequently, the kNN
data is used to generate an sNN igraph for downstream clustering. This
function lives in a separate package from the CPU-based
[`bixverse::find_neighbours_sc()`](https://rdrr.io/pkg/bixverse/man/find_neighbours_sc.html)
so that users without GPU hardware do not need to install the GPU
dependencies.

## Usage

``` r
find_neighbours_gpu_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  gpu_method = c("exhaustive", "ivf"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "cosine",
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

- gpu_method:

  String. One of `c("exhaustive", "ivf")`. `"exhaustive"` computes exact
  nearest neighbours via brute-force on the GPU. `"ivf"` builds an
  inverted file index for approximate search.

- ivf_params:

  List. Output of
  [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md).
  Only used when `gpu_method = "ivf"`. A list with the following items:

  - k - Integer. Number of nearest neighbours to identify.

  - ann_dist - String. Distance metric; one of
    `c("euclidean", "cosine")`.

  - nlist - Optional integer. Number of clusters to partition the index
    into. Controls the granularity of the Voronoi partitioning. If
    `NULL`, defaults to `sqrt(n)` on the Rust side.

  - nprobe - Optional integer. Number of clusters to probe at query
    time. Higher values improve recall at the cost of speed. If `NULL`,
    defaults to `sqrt(nlist)`.

  - nquery - Optional integer. Number of query vectors processed per GPU
    batch. If `NULL`, defaults to 100,000.

  - max_iters - Optional integer. Maximum k-means iterations during
    index construction. If `NULL`, defaults to 30.

  - seed - Integer. Seed for k-means initialisation.

- k:

  Integer. Number of neighbours. Only used when
  `gpu_method = "exhaustive"`.

- dist_metric:

  String. One of `c("euclidean", "cosine")`. Only used when
  `gpu_method = "exhaustive"`.

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
