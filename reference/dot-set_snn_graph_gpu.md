# Build an sNN igraph from kNN data and attach it to a single cell object

Build an sNN igraph from kNN data and attach it to a single cell object

## Usage

``` r
.set_snn_graph_gpu(object, knn_data, snn_params, modality, .verbose)
```

## Arguments

- object:

  `SingleCells` (or `SingleCellsMultiModal`) class.

- knn_data:

  Initialised `sc_knn` with the kNN data.

- snn_params:

  List. Output of
  [`bixverse::params_sc_neighbours()`](https://gregorlueg.github.io/bixverse/reference/params_sc_neighbours.html).

- modality:

  String. One of `c("rna", "adt")`.

- .verbose:

  Boolean or integer. Controls verbosity.

## Value

The object with the sNN graph in the selected modality slot.
