# GPU-accelerated t-SNE

## Intro

`bixverse.gpu` provides a t-SNE wrapper via
[`tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu.md).
The kNN step runs on the GPU, and with the default
`approx_type = "fft_3k_gpu"` so does the optimiser: the embedding stays
on the device from the first epoch to the last.

If you want the algorithm background (perplexity, exaggeration
schedules, BH vs FFT, when to reach for t-SNE over UMAP), read the
[manifoldsR t-SNE
vignette](https://gregorlueg.github.io/manifoldsR/articles/tsne.html).
This vignette only covers what’s different here.

Current split of work:

- **kNN graph construction**: GPU. Three backends (`"nndescent"`,
  `"ivf"`, `"exhaustive"`).
- **Affinities**: CPU.
- **Gradient descent**: GPU with `"fft_3k_gpu"` (the default). CPU with
  `"bh"`, `"bh_qd"`, `"fft"` or `"fft_3k"`, the same Rust code as
  [`manifoldsR::tsne()`](https://gregorlueg.github.io/manifoldsR/reference/tsne.html).

``` r

library(bixverse.gpu)
library(manifoldsR)
#> Warning: package 'manifoldsR' was built under R version 4.5.3
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

## Generating data

Same clustered synthetic data as in the UMAP vignette.

``` r

set.seed(42L)

cluster_data <- manifold_synthetic_data(
  type = "clusters",
  n_samples = 50000L,
  dim = 32L,
  parameters = params_clusters(n_clusters = 25L)
)
```

## Running t-SNE on GPU

NN-descent on GPU for kNN, PCA init, and the default three-kernel FFT
optimiser on the GPU.

``` r

tsne_default <- tsne_gpu(
  data = cluster_data$data,
  perplexity = 15,
  knn_method = "nndescent",
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  .verbose = TRUE
)

plot_df <- as.data.table(tsne_default) |>
  setnames(c("tSNE1", "tSNE2"))
plot_df[, cluster := as.factor(cluster_data$membership)]

ggplot(plot_df, aes(x = tSNE1, y = tSNE2)) +
  geom_point(aes(colour = cluster), size = 0.5, alpha = 0.5) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("tsne_gpu, NN-descent + fft_3k_gpu")
```

![](gpu_tsne_files/figure-html/tsne%20default-1.png)

Clusters separate cleanly, as expected from t-SNE.

## Choosing a kNN backend

Three GPU backends via `knn_method`:

- **`"exhaustive"`**: exact brute force. Small data or ground-truth
  checks. Quadratic in N.
- **`"ivf"`**: inverted file index over Voronoi cells. Wins on large
  data where an approximate answer is fine. Works very well on strongly
  clustered data.
- **`"nndescent"`**: NN-descent with CAGRA-style graph pruning. Solid
  default.

Tuning knobs live in
[`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md),
defaults are fine for most cases. The knobs are identical to the ones
described in the UMAP vignette.

### IVF backend

``` r

tsne_ivf <- tsne_gpu(
  data = cluster_data$data,
  perplexity = 15,
  knn_method = "ivf",
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  .verbose = TRUE
)

plot_df_ivf <- as.data.table(tsne_ivf) |>
  setnames(c("tSNE1", "tSNE2"))
plot_df_ivf[, cluster := as.factor(cluster_data$membership)]

ggplot(plot_df_ivf, aes(x = tSNE1, y = tSNE2)) +
  geom_point(aes(colour = cluster), size = 0.5, alpha = 0.5) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("tsne_gpu, IVF kNN")
```

![](gpu_tsne_files/figure-html/tsne%20ivf-1.png)

Structurally the embeddings agree. Differences are within the noise of
t-SNE’s non-deterministic optimisation.

### Exhaustive backend

``` r

tsne_exhaustive <- tsne_gpu(
  data = cluster_data$data,
  perplexity = 15,
  knn_method = "exhaustive",
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  .verbose = TRUE
)

plot_df_exhaustive <- as.data.table(tsne_exhaustive) |>
  setnames(c("tSNE1", "tSNE2"))
plot_df_exhaustive[, cluster := as.factor(cluster_data$membership)]

ggplot(plot_df_exhaustive, aes(x = tSNE1, y = tSNE2)) +
  geom_point(aes(colour = cluster), size = 0.5, alpha = 0.5) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("tsne_gpu, exhaustive kNN")
```

![](gpu_tsne_files/figure-html/tsne%20exhaustive-1.png)

Structurally the embeddings agree. Differences are within the noise of
t-SNE’s non-deterministic optimisation.

## Choosing an optimiser

Five options via `approx_type`:

- **`"fft_3k_gpu"`**: the default. FFT-interpolated repulsion with three
  kernels (`q`, `q^2 dx`, `q^2 dy`), fully on the GPU. Always runs in
  fp32 (f64 kernels don’t run on Metal); `use_high_precision = TRUE` is
  ignored with a warning. Works on Windows, no FFTW needed.
- **`"bh"`**: Barnes-Hut on the CPU. `O(N log N)`.
- **`"bh_qd"`**: the quick-and-dirty Barnes-Hut from
  [qdtsne](https://github.com/libscran/qdtsne), on the CPU. Tree depth
  capped at `max_depth` in
  [`params_tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_tsne_gpu.md)
  (default `7L`), repulsion computed once per leaf from its centre of
  mass. Coarser repulsive forces, faster than `"bh"`. Works on every
  platform. qdtsne recommends `max_depth` between 7 and
  10. 
- **`"fft"`**: FFT interpolation on the CPU with the classic four-term
  expansion. `O(N)` with a larger constant.
- **`"fft_3k"`**: the CPU twin of `"fft_3k_gpu"`. One forward and three
  inverse FFTs per epoch instead of four each.

The two CPU FFT variants need FFTW, so they’re Unix-only.

``` r

tsne_fft_3k <- tsne_gpu(
  data = cluster_data$data,
  perplexity = 15,
  approx_type = "fft_3k",
  knn_method = "ivf",
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  .verbose = TRUE
)

plot_df_fft_3k <- as.data.table(tsne_fft_3k) |>
  setnames(c("tSNE1", "tSNE2"))
plot_df_fft_3k[, cluster := as.factor(cluster_data$membership)]

ggplot(plot_df_fft_3k, aes(x = tSNE1, y = tSNE2)) +
  geom_point(aes(colour = cluster), size = 0.5, alpha = 0.5) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("tsne_gpu, fft_3k (CPU optimiser)")
```

![](gpu_tsne_files/figure-html/tsne%20fft%203k%20cpu-1.png)

The GPU optimiser sums inside a grid box in whatever order the atomics
land, so two runs with the same seed agree in structure but not bit for
bit. If you need bitwise reproducibility, use one of the CPU optimisers.

If you see an FFT embedding relax into a gapless disc rather than
distinct cluster islands on very large data, that’s the equilibrium at
exaggeration = 1. Set `late_exag_factor` in
[`params_tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_tsne_gpu.md)
to a value between 2 and 4 to pull the clusters apart. Same caveat as
the manifoldsR vignette.

### Timings

Same data, same kNN backend, only the optimiser changes.

``` r

approx_types <- if (.Platform$OS.type == "unix") {
  c("fft_3k_gpu", "bh", "bh_qd", "fft", "fft_3k")
} else {
  c("fft_3k_gpu", "bh", "bh_qd")
}

timings <- rbindlist(lapply(approx_types, \(approx) {
  secs <- system.time(
    tsne_gpu(
      data = cluster_data$data,
      perplexity = 15,
      approx_type = approx,
      knn_method = "ivf",
      seed = 42L,
      .verbose = FALSE
    )
  )[["elapsed"]]
  data.table(approx_type = approx, seconds = round(secs, 2))
}))

timings
#>    approx_type seconds
#>         <char>   <num>
#> 1:  fft_3k_gpu    2.34
#> 2:          bh   18.06
#> 3:       bh_qd    3.58
#> 4:         fft    5.63
#> 5:      fft_3k    4.05
```

## Using a pre-computed kNN graph

Sweeping t-SNE hyperparameters (perplexity, learning rate, exaggeration)
is usually where you spend the most time. Cache the kNN once and hand it
in.

``` r

knn_precomputed <- generate_knn_graph_gpu(
  data = cluster_data$data,
  k = 90L, # ~ 3 * perplexity for perplexity = 30
  knn_method = "ivf",
  .verbose = TRUE
)

tsne_from_knn <- tsne_gpu(
  data = cluster_data$data,
  knn = knn_precomputed,
  perplexity = 30,
  tsne_params = params_tsne_gpu(),
  seed = 42L,
  .verbose = TRUE
)
#> Using provided kNN graph.

plot_df_knn <- as.data.table(tsne_from_knn) |>
  setnames(c("tSNE1", "tSNE2"))
plot_df_knn[, cluster := as.factor(cluster_data$membership)]

ggplot(plot_df_knn, aes(x = tSNE1, y = tSNE2)) +
  geom_point(aes(colour = cluster), size = 0.5, alpha = 0.5) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("tsne_gpu, pre-computed kNN")
```

![](gpu_tsne_files/figure-html/tsne%20precomputed%20knn-1.png)

Note the kNN needs enough neighbours for the perplexity you’re using;
`k ≈ 3 * perplexity` is the usual rule.

## When does the GPU version pay off?

The CPU t-SNE in `manifoldsR` is already fast (Rust, SIMD kNN, FFT
optimiser on Unix). With a CPU optimiser the GPU only buys you the kNN,
so the payoff is the kNN share of the wall-clock. `"fft_3k_gpu"` moves
the optimiser over as well; check the timings above against your own
data size and hardware before assuming it wins.

`"nndescent"` and `"ivf"` are the kNN workhorses, `"exhaustive"` is for
ground-truth checks or small data. Due to usually higher k needed for
tSNE, the `"ivf"` index is the recommendation.

## Conclusions

Same mental model as CPU t-SNE, same knobs, same caveats (t-SNE is a
visualisation tool, cluster sizes and inter-cluster distances are not
meaningful, see the [manifoldsR
vignette](https://gregorlueg.github.io/manifoldsR/articles/tsne.html)
for the details). With `"fft_3k_gpu"` the optimisation runs on the
device; the affinities are still computed on the CPU.
