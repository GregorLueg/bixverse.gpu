# Default parameters for IVF-GPU kNN search

Default parameters for IVF-GPU kNN search

## Usage

``` r
params_sc_ivf(
  k = 15L,
  ann_dist = "cosine",
  nlist = NULL,
  nprobe = NULL,
  nquery = NULL,
  max_iters = NULL,
  seed = 42L
)
```

## Arguments

- k:

  Integer. Number of neighbours to identify.

- ann_dist:

  Character. Distance metric to use. One of `"euclidean"` or `"cosine"`.

- nlist:

  Optional integer. Number of clusters to partition the index into. If
  `NULL`, defaults to `sqrt(n)`.

- nprobe:

  Optional integer. Number of clusters to probe at query time. If
  `NULL`, defaults to `sqrt(nlist)`.

- nquery:

  Optional integer. Number of query vectors processed per GPU batch. If
  `NULL`, defaults to 100,000.

- max_iters:

  Optional integer. Maximum k-means iterations during index build. If
  `NULL`, defaults to 30.

- seed:

  Integer. Seed for k-means initialisation.

## Value

A list with the parameters.
