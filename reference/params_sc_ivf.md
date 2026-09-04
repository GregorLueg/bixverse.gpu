# Default parameters for IVF-GPU kNN search (deprecated)

**\[deprecated\]**

The CAGRA, IVF and exhaustive GPU searches share one parameter wrapper
now, see
[`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).

## Usage

``` r
params_sc_ivf(
  k = 15L,
  ann_dist = "euclidean",
  nlist = NULL,
  nprobe = NULL,
  nquery = NULL,
  max_iters = NULL,
  seed = 42L
)
```

## Arguments

- k:

  Integer. Number of neighbours. Carried on the returned list so the
  deprecated generics can still read it.

- ann_dist:

  Character. One of `"euclidean"` or `"cosine"`.

- nlist:

  Optional integer. Number of clusters to partition the index into.

- nprobe:

  Optional integer. Number of clusters to probe at query time.

- nquery:

  Optional integer. Ignored, the knob is gone.

- max_iters:

  Optional integer. Ignored, the knob is gone.

- seed:

  Integer. Ignored. `seed` is an argument of the calling function.

## Value

A list with the parameters, as returned by
[`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md).
