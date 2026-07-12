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
[`find_neighbours_sc()`](https://gregorlueg.github.io/bixverse/reference/find_neighbours_sc.html)
so that users without GPU hardware do not need to install the GPU
dependencies.

## Usage

``` r
find_neighbours_gpu_sc(
  object,
  embd_to_use = "pca",
  no_embd_to_use = NULL,
  modality = c("rna", "adt"),
  gpu_method = c("ivf", "exhaustive"),
  ivf_params = params_sc_ivf(),
  k = 15L,
  dist_metric = "euclidean",
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

- gpu_method:

  String. One of `c("exhaustive", "ivf")`.

- ivf_params:

  List. Output of
  [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md).
  Only used when `gpu_method = "ivf"`.

- k:

  Integer. Number of neighbours. Only used when
  `gpu_method = "exhaustive"`.

- dist_metric:

  String. One of `c("euclidean", "cosine")` for the distance metric to
  use. This is used specifically only for `gpu_method = "exhaustive"`.

- snn_params:

  List. Output of
  [`bixverse::params_sc_neighbours()`](https://gregorlueg.github.io/bixverse/reference/params_sc_neighbours.html).
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
