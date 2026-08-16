# Default parameters for the GPU nearest neighbour backends

GPU sibling of
[`bixverse::params_knn_defaults()`](https://gregorlueg.github.io/bixverse/reference/params_knn_defaults.html).
The GPU indices take a much smaller knob set: there is no Annoy, no
NN-descent and no HNSW on the device, so only the exhaustive and IVF
parameters survive.

## Usage

``` r
params_knn_gpu_defaults()
```

## Value

A named list with the following parameters:

- k - Number of neighbours. `0L` hands the choice to Rust, which uses
  `sqrt(n_cells) * 0.5` and then adjusts for the simulated doublets.

- knn_method - One of `"exhaustive"` or `"ivf"`.

- ann_dist - One of `"euclidean"` or `"cosine"`. Manhattan is not
  supported by the GPU kernels.

- n_list - IVF only. Number of clusters. `NULL` gives `sqrt(n)`.

- n_probe - IVF only. Clusters to probe. `NULL` gives `sqrt(n_list)`.
