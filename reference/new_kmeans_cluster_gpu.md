# Construct a KMeansClusterGPU object

Construct a KMeansClusterGPU object

## Usage

``` r
new_kmeans_cluster_gpu(centroids, assignments, k, metric)
```

## Arguments

- centroids:

  Numeric matrix of shape k x features.

- assignments:

  Integer vector of length samples (1-indexed).

- k:

  Integer. Number of clusters.

- metric:

  Character. Distance metric used.

## Value

A `KMeansClusterGPU` S3 object.
