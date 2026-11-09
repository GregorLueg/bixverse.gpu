# Methods on plain matrices

These take a numeric matrix (samples x features) rather than a bixverse
object. They mirror `manifoldsR` (UMAP, t-SNE, PHATE, k-means) and `bixverse`
(`rs_cor()`, `rs_covariance()`), with the same mental model and mostly the same
knobs.

## UMAP

```r
emb <- umap_gpu(
  data = mat,
  k = 15L,
  min_dist = 0.5,
  knn_method = "nndescent",          # or "ivf", "exhaustive"
  nn_params = params_nn_gpu(),
  umap_params = params_umap_gpu(),   # optimiser = "adam_gpu" by default
  seed = 42L
)
```

Returns a samples x `n_dim` matrix. kNN and the optimiser run on the device;
fuzzy simplicial set and initialisation on the CPU.

`params_umap_gpu(optimiser = )`:

- `"adam_gpu"`: default, full optimisation on the device.
- `"adam_parallel"`: CPU parallel Adam, the `manifoldsR::umap()` default. GPU
  kNN, CPU optimiser.
- `"adam"`: sequential CPU Adam.
- `"sgd"`: the original UMAP SGD. Pairs with a lower `min_dist`.
- `"sgd_parallel"`: lock-free (Hogwild!) SGD over all threads. Not
  reproducible run to run with more than one thread.

With `n_epochs = NULL`, the SGD optimisers drop to 200 epochs above 10k
points; the Adam optimisers always get 500.

Sweeping `min_dist` or `spread`: build the kNN once with
`generate_knn_graph_gpu()` and pass it as `knn =`. The kNN is the expensive
part.

Below about 10k points `manifoldsR::umap()` on the CPU is competitive or
faster. The GPU pulls ahead from tens of thousands up.

## t-SNE

```r
emb <- tsne_gpu(
  data = mat,
  perplexity = 30,
  approx_type = "fft_3k_gpu",        # default; or "bh", "bh_qd", "fft", "fft_3k"
  knn_method = "exhaustive",
  tsne_params = params_tsne_gpu()
)
```

`approx_type`:

- `"fft_3k_gpu"`: default, three-kernel FFT optimiser on the device.
- `"bh"`, `"bh_qd"`, `"fft"`, `"fft_3k"`: the `manifoldsR` CPU optimisers
  with a GPU kNN. The win is the kNN share of the run.

- `k` is derived as roughly `3 * perplexity`. A pre-computed `knn =` needs at
  least that many neighbours.
- That puts t-SNE at high `k`, where `"nndescent"` loses. Prefer
  `"exhaustive"` or `"ivf"` at perplexity 30 and above.
- FFT on very large data can relax into a gapless disc. Set
  `late_exag_factor` in `params_tsne_gpu()` to 2 to 4.
- UMAP, t-SNE and PHATE always run in fp32; manifolds-rs does the steps prone
  to catastrophic cancellation in fp64 internally.

## PHATE

```r
emb <- phate_gpu(
  data = mat,
  k = 5L,
  knn_method = "ivf",                # default; or "exhaustive", "nndescent"
  phate_params = params_phate_gpu()  # 2048 spectral landmarks by default
)
```

The kNN search and the spectral landmark selection (randomised SVD plus
k-means) run on the device; diffusion time, powering, MDS and the landmark
interpolation on the CPU. For continuous structure (trajectories, branching),
not discrete clusters.

- The search runs `6 * k + 5` wide (35 at `k = 5`). A pre-computed `knn =`
  should be that wide; a narrower one warns.
- `params_phate_gpu(n_landmarks = NULL)` runs the full N x N operator. Fine
  for a few thousand points, warns from 10k up.
- `decay = NULL` gives a binary kernel; `t_custom` fixes the diffusion time
  instead of the entropy knee.
- `backend = "mlx"` moves the kNN and the spectral landmarks to MLX.

## kNN graphs

```r
knn <- generate_knn_graph_gpu(
  data = mat,
  k = 15L,
  knn_method = "nndescent",          # or "cagra", "ivf", "exhaustive"
  nn_params = params_nn_gpu()
)
```

Returns a `manifoldsR` nearest neighbours object: **1-indexed** indices and
true (not squared) Euclidean distances. Feeds `umap_gpu(knn = )`,
`tsne_gpu(knn = )` and `phate_gpu(knn = )`. `umap_gpu()` cuts a graph wider
than `k` down to `k`. `"nndescent"` reads the kNN straight off the descent
graph; `"cagra"` adds the CAGRA prune and a beam search. `umap_gpu()`,
`tsne_gpu()` and `phate_gpu()` have no `"cagra"`. The `extract_knn` argument is deprecated in favour of
`knn_method`.

## Parametric UMAP

```r
model <- parametric_umap(
  data = train,
  n_dim = 2L,
  knn_method = "kmknn",
  parametric_umap_params = params_parametric_umap(n_epochs = 500L),
  use_gpu = TRUE
)
model$embedding                        # training embedding
new_emb <- predict(model, newdata = test)

save_parametric_umap(model, "model.qs2")
model <- load_parametric_umap("model.qs2")
```

Trains an MLP encoder, so new data embeds in one deterministic forward pass.

- The kNN here is the **CPU** `manifoldsR` one: `knn_method` takes
  `"kmknn"`, `"hnsw"`, `"annoy"`, `"nndescent"`, `"balltree"`, `"exhaustive"`,
  and `nn_params` is `manifoldsR::params_nn()`, not `params_nn_gpu()`.
- `use_gpu = FALSE` trains on the CPU. Fine for small data.
- The model holds an external pointer into Rust. `saveRDS()` does not survive
  a session; use `save_parametric_umap()` / `load_parametric_umap()`.
- Out-of-distribution input (a new batch with a big batch effect) projects
  badly. It is an encoder, not a batch corrector.

## k-means

```r
km <- k_means_cluster_gpu(
  data = mat,
  k = 100L,
  kmeans_params = params_kmeans_gpu(metric = "euclidean")
)
km$assignments
manifoldsR::membership(km)
manifoldsR::get_centroids(km)
```

Lloyd's on the device, returns a `KMeansClusterGPU`.

- `k_means_init = NULL` picks `"random"` above 200 centroids and `"parallel"`
  (k-means||, also accepted as `"plusplus"`) otherwise.
- `quantise = TRUE` holds the data at fp16 on the device, centroids and
  accumulators at fp32. Needs `shader-f16` on the adapter.
- Not bit-identical to `manifoldsR::kmeans_cluster()`. Compare with ARI.
- Pays off at large N (roughly 50k and up) and many centroids.

## Correlation and covariance

```r
cor_mat <- rs_cor_gpu(mat, spearman = FALSE, verbose = FALSE)
cov_mat <- rs_cov_gpu(mat, verbose = FALSE)
```

The two `rs_*` functions meant to be called directly; there is no R wrapper,
so all three arguments are required and nothing is validated. Correlations are
between columns. CPU equivalents are `bixverse::rs_cor()` and
`bixverse::rs_covariance()`; differences are f32 rounding.

The GPU wins on tall matrices (many samples). With modest sample counts the
faer CPU path is as fast, and on Apple Silicon the crossover sits far to the
right.
