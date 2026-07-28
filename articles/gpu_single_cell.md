# GPU-accelerated workflows for bixverse single cell

## Intro

This vignette walks through a GPU-accelerated single cell workflow on
the two-batch PBMC data set (pbmc3k + pbmc4k) used in the [batch
correction
vignette](https://gregorlueg.github.io/bixverse/articles/single_cell_batch_correction.html).
The point is to demonstrate how `bixverse.gpu` can drop into the heavier
steps of a standard pipeline: PCA, Harmony v2 batch correction, and the
kNN search. Two batches give us a natural reason to run all three in
sequence: PCA generates the embedding, Harmony corrects it, and the kNN
methods build the neighbour graph for downstream clustering. If none of
this single cell stuff makes sense in the `bixverse` framework, please
read
[this](https://gregorlueg.github.io/bixverse/articles/thinking_single_cell.html)
first.

The core idea behind `bixverse.gpu` is hardware-agnostic GPU code, hence
the use of [cubecl with the wpgu
backend](https://github.com/tracel-ai/cubecl). For small data sets the
host-to-device transfer and kernel launch overhead can outweigh the
speedup, but the larger the data, the more these methods pay off (if you
have sufficient VRAM/unified memory that is…).

`bixverse.gpu` currently provides:

- **GPU PCA** via
  [`calculate_pca_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/calculate_pca_gpu_sc.md):
  a sparse, randomised SVD where the large matrix multiplications run on
  GPU. Scaling (if desired) is applied without ever materialising the
  dense matrix. Also, option for the PFLogPF normalisation.
- **GPU Harmony v2** via
  [`harmony_v2_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/harmony_v2_gpu_sc.md):
  a GPU implementation of Harmony v2 ([Patikas et al.,
  2026](https://www.biorxiv.org/content/10.64898/2026.03.16.711825v1))
  with the Arrowhead matrix inversion. Single batch covariate only on
  the GPU path. R is refined via full-batch Jacobi sweeps rather than
  the blockwise updates of the original: faster, but results are very
  similar rather than bit-identical. GPU floating point reduction
  ordering adds small further deviations.
- **GPU kNN** via two generics:
  - [`find_neighbours_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_gpu_sc.md)
    covers **exhaustive** (exact brute force) and **IVF** (inverted file
    index, approximate).
  - [`find_neighbours_cagra_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_cagra_sc.md)
    covers **CAGRA**, which prunes an NNDescent graph and supports
    either direct kNN extraction or beam search.

``` r

library(bixverse)
library(bixverse.gpu)
library(bixverse.plots)
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

## Loading the data

The pbmc3k and pbmc4k data sets share tissue but differ in sequencing
depth and cell counts, which produces a clear batch effect.

Code

``` r

dir_data <- download_pbmc_batches()

tempdir_gpu <- file.path(tempdir(), "gpu_workflow")
dir.create(tempdir_gpu, showWarnings = FALSE, recursive = TRUE)

h5ad_files <- list.files(dir_data)
h5ad_files <- h5ad_files[grepl(".h5ad", h5ad_files)]
h5ad_paths <- file.path(dir_data, h5ad_files)
names(h5ad_paths) <- c("pbmc3k", "pbmc4k")

h5_tasks <- prescan_h5ad_files(h5_paths = h5ad_paths)

sc_object <- SingleCells(dir_data = tempdir_gpu)

sc_object <- load_multi_h5ad(
  object = sc_object,
  prescan_result = h5_tasks,
  .verbose = TRUE
)
#>  Using light streaming for the CSR to CSC conversion.
#> Loading observation data from h5ad files into DuckDB.
#> Loading variable data into DuckDB.
```

## Quality control

Standard QC -\> mitochondrial proportions, library size and complexity
outliers.

Code

``` r

var <- get_sc_var(sc_object)

h5_metadata <- read_h5ad_metadata(h5ad_paths[[1]])

var <- merge(
  var,
  h5_metadata$var[, c("ENSEMBL_ID", "Symbol_TENx")],
  by.x = "gene_id",
  by.y = "ENSEMBL_ID"
)

setnames(var, old = "Symbol_TENx", new = "gene_symbol", skip_absent = TRUE)

gs_of_interest <- list(
  MT = var[grepl("^MT-", gene_symbol), gene_id],
  Ribo = var[grepl("^RPS|^RPL", gene_symbol), gene_id]
)

sc_object <- gene_set_proportions_sc(
  sc_object,
  gs_of_interest,
  streaming = FALSE,
  .verbose = TRUE
)

qc_df <- sc_object[[c("cell_id", "lib_size", "nnz", "MT")]]

metrics <- list(
  log10_lib_size = log10(qc_df$lib_size),
  log10_nnz = log10(qc_df$nnz),
  MT = qc_df$MT
)
directions <- c(
  log10_lib_size = "twosided",
  log10_nnz = "twosided",
  MT = "above"
)

qc <- run_cell_qc(
  metrics = metrics,
  cells_to_keep = get_cells_to_keep(sc_object),
  directions = directions,
  threshold = 3
)

sc_object[["outlier"]] <- qc$combined
cells_to_keep <- qc_df[!qc$combined, cell_id]
sc_object <- set_cells_to_keep(sc_object, cells_to_keep)
```

## Highly variable genes

HVG selection stays on the CPU.

``` r

sc_object <- find_hvg_sc(
  object = sc_object,
  hvg_no = 2000L,
  .verbose = TRUE
)
```

## GPU-accelerated PCA

However, with PCA things start changing. Let’s run first the CPU version
which you know

``` r

sc_object <- calculate_pca_sc(
  object = sc_object,
  no_pcs = 32L,
  sparse_svd = TRUE
)
#> Using sparse SVD solving on scaled data on 2000 HVG.
# extract the factors and singular values
cpu_res <- get_pca_factors(sc_object)
cpu_s <- get_pca_singular_val(sc_object)
```

The GPU version looks very similar

``` r

sc_object <- calculate_pca_gpu_sc(object = sc_object, no_pcs = 32L)
#> Using GPU-accelerated, randomised sparse SVD data with 2000 HVG.
gpu_res <- get_pca_factors(sc_object)
gpu_s <- get_pca_singular_val(sc_object)
```

Let’s compare against the CPU version:

``` r

ggplot(
  data = data.table(PC1_cpu = cpu_res[, 1], PC1_gpu = gpu_res[, 1]),
  mapping = aes(x = PC1_cpu, y = PC1_gpu)
) +
  geom_point() +
  theme_bw() +
  xlab("PC1 (CPU)") +
  ylab("PC1 (GPU)") +
  ggtitle("CPU vs GPU")
```

![](gpu_single_cell_files/figure-html/cpu%20vs%20gpu%20-%20pc1-1.png)

``` r

ggplot(
  data = data.table(sv_cpu = cpu_s, sv_gpu = gpu_s),
  mapping = aes(x = sv_cpu, y = sv_gpu)
) +
  geom_point() +
  theme_bw() +
  xlab("Singular values (CPU)") +
  ylab("Singular values (GPU)") +
  ggtitle("CPU vs GPU")
```

![](gpu_single_cell_files/figure-html/cpu%20vs%20gpu%20-%20singular%20values-1.png)

You might see some slight differences here driven by floating operation
differences between CPU and GPU. The overall data structure is however
clearly captured.

## GPU-accelerated Harmony v2

With two batches we need batch correction.
[`harmony_v2_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/harmony_v2_gpu_sc.md)
runs Harmony v2 with the Arrowhead inversion on GPU and writes the
corrected embedding to the object as `"harmony_gpu"`.

``` r

sc_object <- harmony_v2_gpu_sc(
  object = sc_object,
  batch_column = "exp_id",
  harmony_params = params_sc_harmony_v2_gpu()
)
#>  Auto-determined number of Harmony clusters: 100
```

For comparison, the CPU version of v2; stored as `"harmony_v2"` so both
embeddings coexist.

``` r

sc_object <- harmony_v2_sc(
  object = sc_object,
  batch_column = "exp_id",
  harmony_params = params_sc_harmony_v2()
)
#>  Auto-determined number of Harmony clusters: 100
```

We will compare the two below, but only once we have kNN graphs on each
embedding, since most of the batch correction metrics need a
neighbourhood structure to begin with. Which is a convenient segue.

## GPU-accelerated kNN

Three methods, each with different speed/precision trade-offs. Below we
run all three against the GPU Harmony embedding to show the API. In
practice, pick one based on data size and how exact you need the
neighbours to be.

### Exhaustive

Exact brute-force search on GPU. Best for smaller data sets or whenever
you need exact neighbours. Scales quadratically, so it gets painful on
large data.

``` r

sc_object <- find_neighbours_gpu_sc(
  object = sc_object,
  embd_to_use = "harmony_gpu",
  gpu_method = "exhaustive",
  k = 15L,
  dist_metric = "euclidean",
  .verbose = TRUE
)
#> Generating GPU kNN data with exhaustive method.
#> Generating sNN graph (full: TRUE).
#> Transforming sNN data to igraph.
```

### IVF

Inverted file index. Partitions the embedding space into Voronoi cells
and probes only a subset at query time. Worthwhile on larger data sets.
The key knobs (`nlist` and `nprobe`) live in
[`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md);
defaults are fine here.

``` r

sc_object <- find_neighbours_gpu_sc(
  object = sc_object,
  embd_to_use = "harmony_gpu",
  gpu_method = "ivf",
  ivf_params = params_sc_ivf(),
  .verbose = TRUE
)
#> Generating GPU kNN data with ivf method.
#> Generating sNN graph (full: TRUE).
#> Transforming sNN data to igraph.
```

### CAGRA

Builds a pruned NNDescent graph, based on the fantastic work by
[Nvidia](https://arxiv.org/pdf/2308.15136). (Unfortunately, some of the
cuda primitives are not available in wgpu, but it is still a blazingly
fast approximate kNN search even on wgpu). With `extract_knn = TRUE` you
pull the kNN straight out of the NNDescent graph: faster, slightly less
precise, useful for rapid iteration over parameters. With
`extract_knn = FALSE` (the default) the function runs beam search over
the pruned graph for higher recall, which matters more on larger,
higher-dimensional data.

``` r

sc_object <- find_neighbours_cagra_sc(
  object = sc_object,
  embd_to_use = "harmony_gpu",
  cagra_params = params_sc_cagra(),
  extract_knn = FALSE,
  .verbose = TRUE
)
#> Generating GPU kNN data with CAGRA method.
#> Generating sNN graph (full: TRUE).
#> Transforming sNN data to igraph.
```

From here downstream methods, clustering, UMAP/tSNE, marker detection,
work without modification, exactly as after a
[`find_neighbours_sc()`](https://gregorlueg.github.io/bixverse/reference/find_neighbours_sc.html)
call.

## Comparing GPU vs CPU Harmony

With the kNN graph in place on the GPU Harmony embedding, we can compute
batch metrics for it and then repeat the kNN step on the CPU embedding
to compare.

``` r

kbet_gpu <- calculate_kbet_sc(sc_object, batch_column = "exp_id")
asw_gpu <- calculate_batch_asw_sc(
  sc_object,
  embd_to_use = "harmony_gpu",
  batch_column = "exp_id"
)
lisi_gpu <- calculate_batch_lisi_sc(sc_object, batch_column = "exp_id")

kbet_gpu
#> kBET Scores
#>   Cells: 5841 | Batches: 2 | Threshold: 0.050
#>   Rejection rate:      0.2785 (1627 / 5841)
#>   Mean Chi-Square:     3.1452 (expected under H0: 1)
#>   Median Chi-Square:   1.9151
asw_gpu
#> Batch Silhouette Width
#>   Cells: 5000 | Batches: 2
#>   Mean ASW:    0.0239 (-1 = strong intermixing, 0 = mixed, 1 = separated)
#>   Median ASW:  0.0457
lisi_gpu
#> Batch LISI Scores
#>   Cells: 5841 | Batches: 2
#>   Mean LISI:    1.4465 (1 = no mixing, 2 = perfect mixing)
#>   Median LISI:  1.4706
```

Same kNN setup on the CPU Harmony embedding:

``` r

sc_object <- find_neighbours_cagra_sc(
  object = sc_object,
  embd_to_use = "harmony_v2",
  cagra_params = params_sc_cagra(),
  extract_knn = FALSE,
  .verbose = TRUE
)
#> Generating GPU kNN data with CAGRA method.
#> Generating sNN graph (full: TRUE).
#> Transforming sNN data to igraph.

kbet_cpu <- calculate_kbet_sc(sc_object, batch_column = "exp_id")
asw_cpu <- calculate_batch_asw_sc(
  sc_object,
  embd_to_use = "harmony_v2",
  batch_column = "exp_id"
)
lisi_cpu <- calculate_batch_lisi_sc(sc_object, batch_column = "exp_id")

kbet_cpu
#> kBET Scores
#>   Cells: 5841 | Batches: 2 | Threshold: 0.050
#>   Rejection rate:      0.2722 (1590 / 5841)
#>   Mean Chi-Square:     3.1018 (expected under H0: 1)
#>   Median Chi-Square:   1.9151
asw_cpu
#> Batch Silhouette Width
#>   Cells: 5000 | Batches: 2
#>   Mean ASW:    0.0245 (-1 = strong intermixing, 0 = mixed, 1 = separated)
#>   Median ASW:  0.0437
lisi_cpu
#> Batch LISI Scores
#>   Cells: 5841 | Batches: 2
#>   Mean LISI:    1.4509 (1 = no mixing, 2 = perfect mixing)
#>   Median LISI:  1.4706
```

Harmony has stochastic elements, so the two embeddings will not be
identical, but the batch correction quality should be in the same
ballpark across the metrics.

### UMAP on the GPU Harmony embedding

Everyone loves visuals, even if you should not over-interpret them
([Chari et al.,
2023](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288)).
[`umap_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu_sc.md)
runs the full GPU-accelerated UMAP path: GPU kNN plus a GPU Adam
optimiser (`optimiser = "adam_gpu"` in
[`params_umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_umap_gpu.md),
the default). On the CAGRA kNN we just computed above, it plugs straight
in with `use_knn = TRUE`.

``` r

sc_object <- find_neighbours_cagra_sc(
  object = sc_object,
  embd_to_use = "harmony_gpu",
  cagra_params = params_sc_cagra(),
  extract_knn = FALSE,
  .verbose = TRUE
)
#> Generating GPU kNN data with CAGRA method.
#> Generating sNN graph (full: TRUE).
#> Transforming sNN data to igraph.

sc_object <- umap_gpu_sc(
  object = sc_object,
  embd_to_use = "harmony_gpu",
  slot_name = "umap_harm_gpu",
  use_knn = TRUE
)
#> Running GPU UMAP.
#> Using n_epochs = 500 (dataset <10k samples or 'adam_parallel'/'adam_gpu' optimiser)
#> Using provided kNN graph.
```

``` r

embedding_plot_sc(
  sc_object,
  embedding = "umap_harm_gpu",
  colour_by = "exp_id",
  discrete = TRUE
) +
  labs(
    title = "GPU Harmony v2 + GPU CAGRA kNN + GPU UMAP",
    colour = "Batch:"
  )
```

![](gpu_single_cell_files/figure-html/umap%20plot-1.png)

### tSNE on the GPU Harmony embedding

[`tsne_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu_sc.md)
mirrors the shape of
[`umap_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu_sc.md)
but only the kNN step runs on GPU. The optimiser (Barnes-Hut or FIt-SNE
FFT) still runs on CPU; a GPU optimiser is on the roadmap. Because t-SNE
derives `k` from `3 * perplexity` on the Rust side, `use_knn` defaults
to `FALSE` so that every call builds a fresh GPU kNN sized to the
requested perplexity. Handy when sweeping perplexities.

``` r

sc_object <- tsne_gpu_sc(
  object = sc_object,
  embd_to_use = "harmony_gpu",
  slot_name = "tsne_harm_gpu",
  perplexity = 30.0
)
#> Running GPU t-SNE.
```

``` r

embedding_plot_sc(
  sc_object,
  embedding = "tsne_harm_gpu",
  colour_by = "exp_id",
  discrete = TRUE
) +
  labs(
    title = "GPU Harmony v2 + GPU t-SNE (GPU kNN, CPU optimiser)",
    colour = "Batch:"
  )
```

![](gpu_single_cell_files/figure-html/tsne%20plot-1.png)

## Conclusions

The full GPU path (PCA, Harmony v2, kNN, UMAP with GPU Adam optimiser,
t-SNE with GPU kNN) plugs into the existing `SingleCells` workflow
without any glue code. The downstream object behaves identically to
whatever you would get from the CPU equivalents.

Longer term, the GPU kNN methods are the obvious building block for
GPU-accelerated versions of methods that lean heavily on kNN graphs:

- **BBKNN**
  ([paper](https://academic.oup.com/bioinformatics/article/36/3/964/5545955),
  CPU
  [implementation](https://gregorlueg.github.io/bixverse/reference/bbknn_sc.html))
- **fastMNN** ([paper](https://www.nature.com/articles/nbt.4091), CPU
  [implementation](https://gregorlueg.github.io/bixverse/reference/fast_mnn_sc.html))

Doublet detection is another candidate. Another potentially interesting
area would be GPU-accelerated NMF … ? Basically, watch the space.

## Clean up

``` r

unlink(tempdir_gpu, recursive = TRUE, force = TRUE)
```
