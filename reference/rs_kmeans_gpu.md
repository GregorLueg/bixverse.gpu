# GPU-accelerated k-means

**\[experimental\]** A GPU-accelerated k-means version leveraging the
wgpu backend via cubecl.

## Usage

``` r
rs_kmeans_gpu(data, dist, n_centroids, kmeans_params, seed, verbose)
```

## Arguments

- data:

  Numeric matrix. Samples x features.

- dist:

  String. Distance metric to use. One of `c("euclidean", "cosine")`.

- n_centroids:

  Integer. Number of clusters, centroids to identify.

- kmeans_params:

  Named list. Contains specific parameters for the GPU- accelerated
  k-means.

- seed:

  Integer. Seed for reproducibility.

- verbose:

  Boolean. Controls verbosity of the function.

## Value

A list with

- centoids - The centroids matrix (centroids x features)

- assignments - The cluster assignments of the data. (1-indexed.)
