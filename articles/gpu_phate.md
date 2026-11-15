# GPU-accelerated PHATE

## Intro

`bixverse.gpu` provides a PHATE wrapper via
[`phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/phate_gpu.md).
PHATE is the one to reach for when the data has continuous structure
(trajectories, branching, differentiation) rather than discrete
clusters.

If you want the algorithm background (alpha-decay kernel, diffusion
time, potential distances, when PHATE is the wrong tool), read the
[manifoldsR PHATE
vignette](https://gregorlueg.github.io/manifoldsR/articles/phate.html).
This vignette only covers what’s different here.

Current split of work:

- **kNN graph construction**: GPU. Three backends (`"ivf"`,
  `"exhaustive"`, `"nndescent"`). The search runs `6 * k + 5` wide, 35
  neighbours at the default `k = 5L`: neighbours past `k` stay in the
  kernel while their affinity clears `thresh`.
- **Landmark selection**: GPU for the default `"spectral"` method
  (randomised SVD plus k-means on the device). `"random"` and
  `"density"` are cheap and stay on the CPU.
- **Diffusion time, powering, MDS, landmark interpolation**: CPU.

So a good part of PHATE stays on the CPU. Don’t expect a GPU blow-out on
every data set; measure on yours.

``` r

library(bixverse.gpu)
library(manifoldsR)
library(data.table)
#> 
#> Attaching package: 'data.table'
#> The following object is masked from 'package:base':
#> 
#>     %notin%
library(ggplot2)
```

> **Note**
>
> Vignettes were built locally on a MacBook Pro M1 Max. The GH runners
> were just too slow and do not have proper GPU support. This gives an
> idea of speed on a decent, but older machine.

## Generating data

A branching trajectory, which is what PHATE was built for.

``` r

set.seed(42L)

trajectory_data <- manifold_synthetic_data(
  type = "trajectory",
  n_samples = 50000L
)

plot_phate <- \(embd, title) {
  plot_df <- as.data.table(embd[, 1:2]) |>
    setnames(c("PHATE1", "PHATE2"))
  plot_df[, branch := as.factor(trajectory_data$membership)]

  ggplot(plot_df, aes(x = PHATE1, y = PHATE2)) +
    geom_point(aes(colour = branch), size = 0.5, alpha = 0.5) +
    theme_bw() +
    theme(legend.position = "none") +
    ggtitle(title)
}
```

## Running PHATE on GPU

Default call: NN-descent on the GPU, 2048 spectral landmarks selected on
the GPU, diffusion time from the Von Neumann entropy knee, SGD-MDS.

``` r

phate_default <- phate_gpu(
  data = trajectory_data$data,
  k = 5L,
  knn_method = "ivf",
  phate_params = params_phate_gpu(),
  seed = 42L,
  .verbose = TRUE
)

plot_phate(phate_default, "phate_gpu, default (IVF + spectral)")
```

![](gpu_phate_files/figure-html/phate%20default-1.png)

## Choosing a kNN backend

Same three GPU backends as for UMAP and t-SNE, and the same knobs in
[`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md):

- **`"ivf"`**: inverted file index over Voronoi cells. The default,
  matching manifolds-rs: PHATE searches far wider than UMAP does, and
  that is where NN-descent falls behind.
- **`"nndescent"`**: the NN-descent graph itself, no index and no
  search. Check the timings below before reaching for it.
- **`"exhaustive"`**: exact brute force. Small data or ground-truth
  checks.

``` r

phate_nndescent <- phate_gpu(
  data = trajectory_data$data,
  knn_method = "nndescent",
  seed = 42L,
  .verbose = TRUE
)

plot_phate(phate_nndescent, "phate_gpu, NN-descent kNN")
```

![](gpu_phate_files/figure-html/phate%20nndescent-1.png)

### Timings

Same data, only the kNN backend changes.

``` r

timings <- rbindlist(lapply(c("nndescent", "ivf", "exhaustive"), \(method) {
  secs <- system.time(
    phate_gpu(
      data = trajectory_data$data,
      knn_method = method,
      seed = 42L,
      .verbose = FALSE
    )
  )[["elapsed"]]
  data.table(knn_method = method, seconds = round(secs, 2))
}))

timings
#>    knn_method seconds
#>        <char>   <num>
#> 1:  nndescent    2.67
#> 2:        ivf    2.55
#> 3: exhaustive    3.17
```

## Landmarks

The full PHATE operator is N x N, which gets painful fast. By default
PHATE compresses it onto `n_landmarks = 2048L` landmarks and
interpolates back to every sample at the end. `n_landmarks = NULL` (or
anything at least the number of samples) runs the full operator;
[`phate_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/phate_gpu.md)
warns when you ask for that on 10k samples or more.

Three ways to pick the landmarks via `landmark_method`:

- **`"spectral"`**: k-means on the leading singular vectors of the
  normalised affinity. The SVD and the k-means run on the GPU. The
  default.
- **`"random"`**: random samples. Cheap, coarser.
- **`"density"`**: weighted by node degree. Cheap.

``` r

phate_random <- phate_gpu(
  data = trajectory_data$data,
  phate_params = params_phate_gpu(landmark_method = "random"),
  seed = 42L,
  .verbose = TRUE
)

plot_phate(phate_random, "phate_gpu, random landmarks")
```

![](gpu_phate_files/figure-html/phate%20random%20landmarks-1.png)

## Using a pre-computed kNN graph

Sweeping the diffusion parameters (`decay`, `t_custom`, `gamma`) does
not need a new kNN each time. Build it once, `6 * k + 5` wide, and hand
it in. A narrower graph works but gives a warning: the neighbours past
its edge are missing from the kernel.

``` r

knn_precomputed <- generate_knn_graph_gpu(
  data = trajectory_data$data,
  k = 35L, # 6 * k + 5 for k = 5
  knn_method = "ivf",
  .verbose = TRUE
)

phate_t <- rbindlist(lapply(c(5L, 20L, 60L), \(t) {
  embd <- phate_gpu(
    data = trajectory_data$data,
    knn = knn_precomputed,
    k = 5L,
    phate_params = params_phate_gpu(t_custom = t),
    seed = 42L,
    .verbose = FALSE
  )
  data.table(
    PHATE1 = embd[, 1],
    PHATE2 = embd[, 2],
    branch = as.factor(trajectory_data$membership),
    t = sprintf("t = %d", t)
  )
}))

ggplot(phate_t, aes(x = PHATE1, y = PHATE2)) +
  geom_point(aes(colour = branch), size = 0.3, alpha = 0.5) +
  facet_wrap(~t, scales = "free") +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("phate_gpu, fixed diffusion times on one kNN graph")
```

![](gpu_phate_files/figure-html/phate%20precomputed%20knn-1.png)

Small `t` keeps the noise, large `t` washes out the finer branches. The
automatic knee usually lands somewhere sensible, but it’s worth a look.

## MLX on Apple Silicon

With a build that has the MLX backend (see
[`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)),
`backend = "mlx"` moves the kNN search and the spectral landmark
selection to MLX. Everything that runs on the CPU stays there.

``` r

phate_mlx <- phate_gpu(
  data = trajectory_data$data,
  backend = "mlx",
  seed = 42L,
  .verbose = TRUE
)

plot_phate(phate_mlx, "phate_gpu, MLX backend")
```

## Conclusions

Same knobs and same caveats as the CPU PHATE in manifoldsR (it is the
wrong tool for discrete clusters, see the [manifoldsR
vignette](https://gregorlueg.github.io/manifoldsR/articles/phate.html)).
The GPU takes the kNN search and the spectral landmark selection; the
diffusion and MDS are still CPU work. For single cell data,
[`phate_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/phate_gpu_sc.md)
wraps all of this for `SingleCells`, see the [single cell
vignette](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_single_cell.md).
