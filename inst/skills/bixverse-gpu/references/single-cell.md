# Single cell on the GPU

Everything here dispatches on bixverse objects. Load, QC, HVG selection,
clustering, markers and plotting stay in bixverse; see the `bixverse` skill for
those. The GPU functions slot into the same chain.

| Step | CPU (bixverse) | GPU (bixverse.gpu) | Classes |
|---|---|---|---|
| PCA | `calculate_pca_sc()` | `calculate_pca_gpu_sc()` | `SingleCells` |
| Harmony v2 | `harmony_v2_sc()` | `harmony_v2_gpu_sc()` | `SingleCells` |
| kNN + sNN | `find_neighbours_sc()` | `find_neighbours_gpu_sc()` | `SingleCells` |
| kNN only | `generate_knn_sc()` | `generate_gpu_knn_sc()` | `SingleCells` |
| Fast clustering | `fast_cluster_sc()` | `fast_cluster_gpu_sc()` | `SingleCells`, `SingleCellsSubset` |
| BBKNN | `bbknn_sc()` | `bbknn_gpu_sc()` | `SingleCells`, `SingleCellsSubset` |
| UMAP | `umap_sc()` | `umap_gpu_sc()` | `SingleCells` |
| t-SNE | `tsne_sc()` | `tsne_gpu_sc()` | `SingleCells` |
| Scrublet | `scrublet_sc()` | `scrublet_gpu_sc()` | `SingleCells` |
| SEACells | `generate_seacells_sc()` | `generate_seacells_gpu_sc()` | `SingleCells`, `SingleCellsSubset` |
| SCENIC GRN | `scenic_grn_sc()` | `scenic_grn_sc_gpu()` | `SingleCells`, `MetaCells` |
| NMF | `nmf_sc()`, `stabilised_nmf_sc()`, `consensus_nmf_sc()`, `nmf_k_sweep_sc()` | same names with `_gpu_sc` | `SingleCells`, `MetaCells` |
| NEBULA | `nebula_sc()` | `nebula_gpu_sc()` | `SingleCells`, `SingleCellsSubset` |

Check the CPU names against the `bixverse` skill's API index before relying on
them; they live in the other package.

## PCA

```r
obj <- find_hvg_sc(obj, hvg_no = 2000L)
obj <- calculate_pca_gpu_sc(obj, no_pcs = 30L)
```

Sparse randomised SVD with the big matmuls on the device. Scaling is applied
without densifying. Takes `bixverse::params_sc_pca()`, so the PFlogPF option
comes from there.

- It writes to the **same PCA slot** as `calculate_pca_sc()`, so it overwrites
  a CPU PCA. `get_pca_factors()` / `get_pca_singular_val()` read either.
- `hvg` overrides the stored HVGs and is **1-based**. It also resets the
  object's HVG set to what you passed.
- PCs match CPU up to sign and small f32 noise. Compare with a scatter of PC
  scores, not `identical()`.

## Harmony v2

```r
obj <- harmony_v2_gpu_sc(
  obj,
  batch_column = "sample_id",
  harmony_params = params_sc_harmony_v2_gpu()
)
```

- Writes the embedding as **`"harmony_gpu"`**, not `"harmony_v2"`. Downstream
  calls need `embd_to_use = "harmony_gpu"`. CPU and GPU embeddings can coexist.
- **One batch covariate only.** `batch_column` is a single string.
- R is refined by full-batch Jacobi sweeps rather than the original blockwise
  updates. Batch metrics (kBET, ASW, LISI) land in the same place as CPU; the
  embedding is not identical.
- `params_sc_harmony_v2_gpu()` carries the GPU k-means knobs too
  (`k_means_iter`, `k_means_init`, `quantise`).

## kNN

```r
obj <- find_neighbours_gpu_sc(
  obj,
  embd_to_use = "harmony_gpu",
  knn_method = "nndescent",    # or "exhaustive", "ivf"
  nn_params = params_nn_gpu(),
  k = 15L,
  snn_params = params_sc_neighbours()
)
```

Builds the kNN on the device, then the sNN graph on the CPU from
`bixverse::params_sc_neighbours()`. The object afterwards is identical in shape
to one from `find_neighbours_sc()`, so `find_clusters_sc()`, markers and the
rest run unchanged. `generate_gpu_knn_sc()` returns just the kNN.

Picking `knn_method`:

- `"exhaustive"`: exact, no knobs. Cost barely moves with `k`. Quadratic in
  cells, so it gets slow on very large data.
- `"ivf"`: approximate, `n_list` / `n_probes` in `params_nn_gpu()`. The fallback
  when exhaustive is too slow, at any `k`.
- `"nndescent"`: NN-descent pruned into a CAGRA graph, then beam search
  (`extract_knn = FALSE`, default) or direct extraction from the graph
  (`extract_knn = TRUE`, faster, lower recall). Fastest at `k` up to about 30,
  then loses badly because the build degree tracks `k`.

The distance metric is `dist_metric` in `params_nn_gpu()`, `"euclidean"` or
`"cosine"`.

## Fast clustering

```r
fc <- fast_cluster_gpu_sc(
  obj,
  embd_to_use = "harmony_gpu",
  resolutions = c(2, 1, 0.5),
  grid_search = TRUE,
  no_seeds = 25L,
  return_kmeans = TRUE
)
obj <- add_sc_new_obs(obj, obs_data = get_data(fc))
```

k-means coarsening on the device, then centroid kNN, optional sNN and Louvain
on the CPU. Returns a `SingleCellFastClusters` object, **not** the updated
`SingleCells`, so push the memberships back yourself as above. Columns are
`res_<value>`. With `grid_search = TRUE`, `fc$stats` has `mean_ari` (stability
across seeds), `mean_conductance` (lower is better) and `mean_n_comms`, which is
how you pick a resolution. No mini-batch k-means on the GPU path.

## BBKNN

```r
obj <- bbknn_gpu_sc(
  obj,
  batch_column = "sample_id",
  no_neighbours_to_keep = 15L,
  bbknn_params = params_sc_bbknn_gpu(neighbours_within_batch = 10L)
)
```

Corrects the graph, not the embedding. One index per batch, every cell queried
against each. Work scales with `n_cells * n_batches`, so the GPU pays off with
many batches and barely at all with two.

- **Overwrites the kNN and the graph on the object.** Run it last, or after
  you have taken what you need from a previous kNN.
- Graph weights are BBKNN connectivities, not sNN counts.
- `no_neighbours_to_keep` must be below `neighbours_within_batch * n_batches`
  for the distance filtering to do anything. Above it you get a warning and
  every neighbour.
- **Do not judge it with kBET.** kBET measures the batch proportions BBKNN
  forces by construction, so it looks great regardless. Use
  `calculate_lisi_sc(obj, label_column = "sample_id", type = "batch")`. ASW
  needs an embedding BBKNN never produces.
- GPU and CPU agree exactly with `knn_method = "exhaustive"`; the approximate
  backends break ties differently.

## UMAP and t-SNE on the object

```r
obj <- umap_gpu_sc(obj, embd_to_use = "harmony_gpu", slot_name = "umap_harm")
obj <- tsne_gpu_sc(obj, embd_to_use = "harmony_gpu", perplexity = 30)
```

Both write to `other_embeddings[[slot_name]]` (defaults `"umap"` / `"tsne"`),
readable by `bixverse.plots::embedding_plot_sc()`.

- `umap_gpu_sc(use_knn = TRUE)` is the **default** and reuses the stored kNN.
- `tsne_gpu_sc(use_knn = FALSE)` is the **default** and builds a fresh kNN each
  call, because t-SNE needs `k` around `3 * perplexity`. Only set
  `use_knn = TRUE` if the stored `k` is at least that. At perplexity 30 that
  is `k = 90`, well into the range where `"nndescent"` loses; pick
  `knn_method = "exhaustive"` or `"ivf"`.
- The t-SNE optimiser runs on the CPU; only the kNN is on the device. UMAP runs
  both kNN and the Adam optimiser on the device.
- Knobs are in `params_umap_gpu()` / `params_tsne_gpu()`, see
  `matrix-methods.md`.

## Scrublet

```r
res <- scrublet_gpu_sc(
  obj,
  scrublet_params = params_scrublet_gpu(expected_doublet_rate = 0.1),
  group_by = "sample_id"
)
```

SVD, doublet projection and the kNN on the device; HVG, simulation, scoring and
Otsu on the CPU. Returns bixverse's `ScrubletRes`, so `plot()`, `get_data()`,
`call_doublets_manual()` and `add_sc_new_obs()` work as usual. With `group_by`
the threshold is per group and `call_doublets_manual(..., for_sample = )`
adjusts one.

- Scrublet searches at a **high `k`**. `knn = list(k = 0L)` (the default)
  scales `k` with the combined embedding, so 20k cells lands around `k = 175`.
  Leave `knn_method` at `"exhaustive"` (the default) or use `"ivf"`.
  `"nndescent"` at that `k` was about 40x slower than exhaustive when measured.
- `knn_backend = "cpu"` keeps SVD and projection on the device but uses the
  bixverse CPU indices. For reproducing a CPU run, not for speed. The `knn`
  list is then validated against `bixverse::params_knn_defaults()` instead,
  and keys from the wrong backend error.
- Scores correlate with CPU but do not match: different SVD sketch, different
  tie breaking, and Otsu can land one histogram bin over. Compare calls and
  precision / recall, not scores.

## SEACells

```r
mc <- generate_seacells_gpu_sc(
  obj,
  seacell_params = params_sc_seacells(n_sea_cells = 1000L, pruning = TRUE)
)
```

Both Frank-Wolfe solves on the device; kNN, kernel, initialisation and RSS on
the CPU. Same `params_sc_seacells()` as the CPU version, returns a bixverse
`MetaCells` that feeds straight into `scenic_grn_sc_gpu()` and the NMF
functions.

- The speedup grows with `n_sea_cells`. At 50 meta cells expect little.
- For large data set `n_landmarks` (Nystroem initialisation; otherwise init is
  quadratic in cells) and keep `pruning = TRUE`.
- If no GPU workgroup tier covers the kNN `k`, a solve quietly falls back to
  the CPU for that iteration.
- `regenerate_knn = FALSE` (default) reuses the stored kNN. `cells_to_use`
  forces a rebuild on the subset.
- Assignments differ from CPU for cells sitting between two archetypes. Compare
  meta cell size distributions and final RSS.

## SCENIC

```r
grn <- scenic_grn_sc_gpu(
  obj,
  tf_ids = tf_gene_ids,
  genes_to_take = scenic_gene_filter_sc(obj, scenic_params = params_scenic()),
  scenic_params = params_scenic(learner_type = "extratrees", gene_batch_size = 64L)
)
```

Returns bixverse's `ScenicGrn`, so `identify_tf_to_genes()`,
`tf_to_genes_correlations()` and `tf_to_genes_motif_enrichment()` run
unchanged.

- `learner_type` must be `"extratrees"` (recommended) or `"randomforest"`.
  **`"grnboost2"` errors**; use `bixverse::scenic_grn_sc()` for it.
- `wave_byte_budget` (default 4 GiB) caps VRAM for the per-wave tensors.
  Shrink it on an 8 GB card shared with a display.
- `streaming = NULL` picks streaming vs in-memory like the CPU path. On
  `MetaCells` it is ignored.
- At very large cell counts the packed feature tensor (about 1 byte per cell
  per TF) hits the adapter's binding limit first. The driver checks up front;
  subsample cells with `n_subsample` or trim the TF list.
- Small data (a few thousand cells, a few hundred targets): stay on the CPU.

## NMF

`nmf_gpu_sc()`, `stabilised_nmf_gpu_sc()`, `consensus_nmf_gpu_sc()`,
`nmf_k_sweep_gpu_sc()`. Same arguments and result classes as the bixverse CPU
versions, taking `params_nmf_hals()` / `params_nmf_consensus()`.

- The HALS solve runs on the device; NNDSVD init and the whole consensus step
  stay on the CPU.
- A single fit barely gains. The win is many solves over one matrix, so the k
  sweep flatters it most.
- **Rank above 128 errors.** Use the CPU version.
- On few restarts, `params_nmf_consensus(density_threshold = 2)` switches the
  density filter off; with it on, a run that drops below `k` survivors errors.
- From a sweep, take the last `k` before stability falls away, not the most
  stable row: `k = 2` is always stable.

## NEBULA

```r
res <- nebula_gpu_sc(
  subset_obj,
  subject_col = "donor",
  design = ~condition,
  genes_to_use = hvg_names
)
```

Drop-in for `bixverse::nebula_sc()`, returns the same `ScNebula`. Stage two
(per-gene penalised fits) runs on the device in f32 and is finished in f64 on
the host.

- `params_nebula_gpu()` has **no `reml`**. Need REML, use the CPU version.
- No `MetaCells` method.
- At a few thousand cells and 500 genes it was a wash against CPU. Benchmark on
  your own data before switching.
