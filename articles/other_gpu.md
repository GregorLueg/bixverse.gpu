# Other GPU-accelerated methods

## Intro

Beyond the single cell stuff, `bixverse.gpu` has a small collection of
general-purpose GPU kernels that keep showing up: k-means, and Pearson
correlation / covariance on wide matrices. This vignette runs each one
against its CPU counterpart and gives a rough sense of when the GPU path
actually pays off.

Everything here runs on [cubecl with the wgpu
backend](https://github.com/tracel-ai/cubecl), so no CUDA-only
dependencies.

``` r

library(bixverse)
library(bixverse.gpu)
library(manifoldsR)
library(data.table)
#> Warning: package 'data.table' was built under R version 4.5.2
library(ggplot2)
#> Warning: package 'ggplot2' was built under R version 4.5.2
```

> **Note**
>
> Vignettes were built locally on a MacBook Pro M1 Max. The GH runners
> were just too slow and do not have proper GPU support. This gives an
> idea of speed on a decent, but older machine.

## GPU-accelerated k-means

k-means keeps coming back everywhere: quantisation, initialisation for
other clustering methods, IVF index construction, Harmony soft
assignments. It’s a natural GPU workload because every iteration is
dominated by a distance kernel and a reduction.

[`k_means_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/k_means_cluster_gpu.md)
is Lloyd’s on the GPU. Knobs in
[`params_kmeans_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_kmeans_gpu.md):

- `k_means_init`: two variants exist on the GPU path, `"random"` (fast
  random selection) and `"parallel"` (also accepted as `"plusplus"`;
  this is k-means\|\|, the parallel version of k-means++). `NULL` picks
  `"random"` when `n_centroids > 200`, `"parallel"` otherwise.
- `metric`: `"euclidean"` or `"cosine"`.
- `fixed`: run for a fixed number of iterations, skip the convergence
  check.
- `quantise`: hold the data buffer at fp16 on the GPU. Centroids and
  accumulators stay at fp32. Halves the data-buffer memory and boosts
  bandwidth on the assignment kernels. Needs `shader-f16` on the wgpu
  adapter.

10k points in 32D across 100 well-separated clusters:

``` r

set.seed(42L)

cluster_data <- manifold_synthetic_data(
  type = "clusters",
  n_samples = 10000L,
  dim = 32L,
  parameters = params_clusters(n_clusters = 100L)
)
```

### CPU baseline

CPU reference is
[`manifoldsR::kmeans_cluster()`](https://gregorlueg.github.io/manifoldsR/reference/kmeans_cluster.html),
which is Lloyd’s along two orthogonal acceleration axes: **Hamerly’s
bounds** (per-point upper/lower distance bounds that prune most distance
calculations each iteration) and a **GEMM assignment path** (batch
point-to-centroid distances through a `faer` matmul when dim and
centroid count are big enough). The four resulting paths
(`ParallelLloyd`, `GemmLloyd`, `HamerlySimd`, `HamerlyGemm`) are
auto-selected from the data shape. Hamerly falls back to plain Lloyd’s
under Cosine (no triangle inequality).

Here we force both off. Comparison is against plain parallel SIMD
Lloyd’s, same solution the accelerated paths converge to, just with a
bigger distance-calculation budget.

``` r

km_cpu <- kmeans_cluster(
  data = cluster_data$data,
  k = 100L,
  method = "full",
  kmeans_params = params_kmeans(
    init = "random",
    metric = "euclidean",
    use_hamerly = FALSE,
    use_gemm = FALSE
  ),
  seed = 42L,
  .verbose = FALSE
)
```

### GPU version

Same interface, bump `k_means_iter` so both runs get a comparable
budget:

``` r

km_gpu <- k_means_cluster_gpu(
  data = cluster_data$data,
  k = 100L,
  kmeans_params = params_kmeans_gpu(
    k_means_iter = 100L,
    k_means_init = "random",
    metric = "euclidean",
    quantise = FALSE
  ),
  seed = 42L,
  .verbose = FALSE
)
```

k-means is a local optimiser sensitive to init, so the two won’t be
bit-identical. Assignments should agree well though. ARI is the standard
measure:

``` r

ari_cpu_gpu <- calc_ari(km_cpu$assignments, km_gpu$assignments)
cat(sprintf("ARI (CPU vs GPU): %.3f\n", ari_cpu_gpu))
#> ARI (CPU vs GPU): 1.000
```

### Quantised GPU version

For memory-bound workloads (very wide feature matrices, very large N),
`quantise = TRUE` keeps the data buffer at `fp16` on the GPU while
centroids and accumulators stay at `fp32`. Frees up bandwidth on your
GPU. Accuracy loss is (usually) small and easy to check:

``` r

km_gpu_q <- k_means_cluster_gpu(
  data = cluster_data$data,
  k = 100L,
  kmeans_params = params_kmeans_gpu(
    k_means_iter = 100L,
    k_means_init = "random",
    metric = "euclidean",
    quantise = TRUE
  ),
  seed = 42L,
  .verbose = FALSE
)

ari_gpu_q <- calc_ari(km_gpu$assignments, km_gpu_q$assignments)
cat(sprintf("ARI (GPU fp32 vs GPU fp16): %.3f\n", ari_gpu_q))
#> ARI (GPU fp32 vs GPU fp16): 0.998
```

Tracks the full-precision run closely at a fraction of the memory.

### When to use the GPU version

On small data (a few thousand points, low dim) the CPU tends to win.
Launch overhead and host-device transfers dominate. GPU pays off when:

- N is large (roughly \> 50k),
- k is large (hundreds to thousands of centroids),
- Features are moderate to high dimensional.

## GPU-accelerated correlation and covariance

Pairwise Pearson correlation over a wide matrix looks trivial until you
notice it’s a symmetric `X^T X` after centring and scaling. Textbook
BLAS-3 GEMM. GPUs love this; modern CPU BLAS with SIMD also loves this,
which is why the crossover point isn’t always where you’d expect.

`bixverse` already ships
[`rs_cor()`](https://gregorlueg.github.io/bixverse/reference/rs_cor.html),
which dispatches to [`faer`](https://github.com/sarah-quinones/faer-rs)
under the hood. faer is a Rust BLAS with strong SIMD support, and a
generic GEMM on the GPU only beat it on very large data sets.
`bixverse-rs` now ships handrolled Gram kernels for correlation and
covariance that actually beat `faer`, and `bixverse.gpu` exposes them as
[`rs_cor_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cor_gpu.md)
and
[`rs_cov_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cov_gpu.md)
for when you need your correlations fast on large data sets.

### Correlation

25k samples, 2.5k features. Base R vs CPU vs GPU:

``` r

set.seed(42L)

n_samples <- 25000L
n_features <- 2500L

random_data <- matrix(
  data = rnorm(n_samples * n_features),
  nrow = n_samples,
  ncol = n_features
)
```

Base R [`cor()`](https://rdrr.io/r/stats/cor.html):

``` r

# Base R will be slow...
t_base <- system.time({
  cor_base <- cor(random_data)
})
cat(sprintf("base R cor():   %.2fs\n", t_base[["elapsed"]]))
#> base R cor():   75.41s
```

faer-backed CPU (via `bixverse`):

``` r

t_cpu <- system.time({
  cor_cpu <- rs_cor(random_data, spearman = FALSE)
})
cat(sprintf("rs_cor() CPU:   %.2fs\n", t_cpu[["elapsed"]]))
#> rs_cor() CPU:   0.87s
```

And GPU-accelerated:

``` r

t_gpu <- system.time({
  cor_gpu <- rs_cor_gpu(random_data, spearman = FALSE, verbose = FALSE)
})
cat(sprintf("rs_cor_gpu():   %.2fs\n", t_gpu[["elapsed"]]))
#> rs_cor_gpu():   0.57s
```

All three should agree:

``` r

cat(sprintf("max |base - cpu|: %.2e\n", max(abs(cor_base - cor_cpu))))
cat(sprintf("max |cpu  - gpu|: %.2e\n", max(abs(cor_cpu - cor_gpu))))
```

CPU-vs-GPU deviation is fp32 GEMM rounding, not an algorithmic bug.

### Covariance

Same story:

``` r

t_cpu_cov <- system.time({
  cov_cpu <- rs_covariance(random_data)
})
cat(sprintf("rs_covariance() CPU: %.2fs\n", t_cpu_cov[["elapsed"]]))
#> rs_covariance() CPU: 0.73s

t_gpu_cov <- system.time({
  cov_gpu <- rs_cov_gpu(random_data, verbose = FALSE)
})
cat(sprintf("rs_cov_gpu():        %.2fs\n", t_gpu_cov[["elapsed"]]))
#> rs_cov_gpu():        0.45s

cat(sprintf("max |cpu - gpu|:     %.2e\n", max(abs(cov_cpu - cov_gpu))))
#> max |cpu - gpu|:     4.68e-06
```

### When to use the GPU version

The GPU pulls ahead mainly on **very tall matrices**, lots of samples.
Many rows to sum over during centring and the GEMM, so the GPU’s
arithmetic throughput outpaces what the CPU can sustain over host
memory.

On Apple Silicon, faer’s SIMD path is close enough that this crossover
sits fairly far to the right. Kernel launch and host-device transfer
overhead can wipe out the win on smaller inputs, and wgpu doesn’t expose
proper tensor-core equivalents on that hardware, so you don’t get the
blow-out margins you’d see on a CUDA machine.

Rough shape:

- Small feature count, modest sample size:
  [`rs_cor()`](https://gregorlueg.github.io/bixverse/reference/rs_cor.html)
  on CPU wins.
- ~1k features, tens of thousands of samples: CPU and GPU close on Apple
  Silicon; discrete GPUs pull ahead earlier.
- Very tall or very wide: GPU wins clearly.

Same qualitative curve for
[`rs_cov_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cov_gpu.md).

## Conclusions

None of these are the headline GPU workloads (that’s Harmony, PCA, kNN,
UMAP). They’re the small pieces that make full pipelines feel snappy.
Correlation and covariance in particular sit inside a lot of downstream
methods (co-expression networks, PCA on covariance, contrastive PCA, …),
so having a drop-in GPU version keeps the whole thing on one device.

More GPU utilities will land here over time. Watch this space.
