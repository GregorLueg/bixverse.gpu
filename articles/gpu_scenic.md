# GPU-accelerated SCENIC

## Intro

This vignette walks through the GPU-accelerated version of the SCENIC
gene regulatory network inference in `bixverse`. If you have not read
the [CPU SCENIC
vignette](https://gregorlueg.github.io/bixverse/articles/bag_of_genes_single_cells.html)
first, do so. That vignette explains the biology and the CPU code path.
This one focuses on what the GPU version buys you and how the API
differs.

Behind
[`scenic_grn_sc_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/scenic_grn_sc_gpu.md)
sits a wave-scheduled, multi-tree GPU driver that dispatches ExtraTrees
and RandomForest fits to the wgpu backend via
[cubecl](https://github.com/tracel-ai/cubecl). GRNBoost2 stays on the
CPU because gradient boosting doesn’t map cleanly onto the same
multi-output kernel design.

The `bixverse.gpu` package provides:

- **[`scenic_grn_sc_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/scenic_grn_sc_gpu.md)**
  for `SingleCells` and `MetaCells` objects. Streaming and non-streaming
  drivers on the `SingleCells` path; a single in-memory driver on the
  `MetaCells` path.
- A single GPU-specific knob, `wave_byte_budget`, that caps VRAM use of
  the per-wave histogram tensors. Default is 4 GiB. Shrink on 8 GB
  adapters shared with other workloads; raise on 16 GB+ adapters to let
  the scheduler pick a wider wave.

Everything else, the params, the target-gene shortlisting, the
downstream motif enrichment, is imported straight from `bixverse`. The
GPU version reuses the same
[`params_scenic()`](https://gregorlueg.github.io/bixverse/reference/params_scenic.html),
the same
[`scenic_gene_filter_sc()`](https://gregorlueg.github.io/bixverse/reference/scenic_gene_filter_sc.html),
the same `ScenicGrn` result class, so downstream code
(`identify_tf_to_genes`, `tf_to_genes_correlations`,
`tf_to_genes_motif_enrichment`) is identical.

``` r

library(bixverse)
library(bixverse.gpu)
library(data.table)
#> Warning: package 'data.table' was built under R version 4.5.2
library(magrittr)
#> Warning: package 'magrittr' was built under R version 4.5.2
```

> **Note**
>
> Vignettes were built locally on a MacBook Pro M1 Max. The GH runners
> were just too slow and do not have proper GPU support. This gives an
> idea of speed on a decent, but older machine.

## Rebuilding a processed PBMC3k object

We reconstruct the same `SingleCells` object used in the CPU SCENIC
vignette. The chunk is folded because we care about SCENIC, not QC.

Rebuild the PBMC3k object (click to expand)

``` r

pbmc3k_path <- download_pbmc3k()
tempdir_pbmc <- file.path(tempdir(), "gpu_scenic")
dir.create(tempdir_pbmc, showWarnings = FALSE, recursive = TRUE)

sc_object <- SingleCells(dir_data = tempdir_pbmc)
mtx_io_params <- get_cell_ranger_params(pbmc3k_path)

sc_object <- load_mtx(
  object = sc_object,
  sc_mtx_io_param = mtx_io_params,
  mtx_streaming = FALSE,
  .verbose = FALSE
)
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.

setnames_sc(sc_object, table = "var", old = "column1", new = "gene_symbol")
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.

var <- get_sc_var(sc_object)
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
ensembl_to_symbol <- setNames(var$gene_symbol, var$gene_id)
symbol_to_ensembl <- setNames(var$gene_id, var$gene_symbol)

gs_of_interest <- list(
  MT = var[grepl("^MT-", gene_symbol), gene_id],
  Ribo = var[grepl("^RPS|^RPL", gene_symbol), gene_id]
)
sc_object <- gene_set_proportions_sc(
  sc_object,
  gs_of_interest,
  streaming = FALSE,
  .verbose = FALSE
)
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.

qc_df <- sc_object[[c("cell_id", "lib_size", "nnz", "MT")]]
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
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
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
cells_to_keep <- qc_df[!qc$combined, cell_id]
sc_object <- set_cells_to_keep(sc_object, cells_to_keep)
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.

sc_object <- find_hvg_sc(sc_object, hvg_no = 2000L, .verbose = FALSE)
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
sc_object <- calculate_pca_sc(sc_object, no_pcs = 30L, .verbose = FALSE)
sc_object <- find_neighbours_sc(
  sc_object,
  neighbours_params = params_sc_neighbours(knn = list(k = 15L)),
  .verbose = FALSE
)
```

## Gene filtering

Same as the CPU workflow. The gene filter is a cheap min-counts /
min-cells scan; there’s no GPU version because there’s no need for one.

``` r

scenic_genes <- scenic_gene_filter_sc(
  object = sc_object,
  scenic_params = params_scenic(min_counts = 50L),
  .verbose = FALSE
)

length(scenic_genes)
#> [1] 5430
```

## Transcription factor list

Aerts lab curated list, mapped to Ensembl.

``` r

tf_dt <- data.table::fread(
  "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
  header = FALSE,
  col.names = "tf"
)
tf_dt[, gene_id := symbol_to_ensembl[tf]]
tf_dt <- tf_dt[!is.na(gene_id)]

nrow(tf_dt)
#> [1] 1100
```

## GRN inference on the GPU

Same signature as
[`scenic_grn_sc()`](https://gregorlueg.github.io/bixverse/reference/scenic_grn_sc.html)
from `bixverse`, with the extra `wave_byte_budget` knob. Everything
else, `scenic_params`, `tf_ids`, `genes_to_take`, works the same way.

### ExtraTrees

`extratrees` is the recommended default on GPU. Random split thresholds
mean each level dispatches a single split-scoring kernel, and the driver
saturates the device with wide waves.

``` r

scenic_res_et <- scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tf_dt$gene_id,
  genes_to_take = scenic_genes,
  scenic_params = params_scenic(
    learner_type = "extratrees",
    gene_batch_size = 64L
  ),
  .verbose = TRUE
)
#> SCENIC GPU: 5430 target genes, 466 TFs, 2163 cells (streaming: FALSE)

scenic_res_et
#> ScenicGrn (GRN results)
#>   No genes:                 5430 
#>   No TFs:                   466 
#>   Applied learner:          extratrees 
#>   TF to gene generated:     FALSE 
#>   CisTarget res generated:  FALSE
```

### RandomForest

`randomforest` uses exhaustive split evaluation. It’s a bit slower per
tree than ExtraTrees but often lands on tighter importance rankings.

``` r

scenic_res_rf <- scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tf_dt$gene_id,
  genes_to_take = scenic_genes,
  scenic_params = params_scenic(
    learner_type = "randomforest",
    gene_batch_size = 64L
  ),
  .verbose = TRUE
)
#> SCENIC GPU: 5430 target genes, 466 TFs, 2163 cells (streaming: FALSE)

scenic_res_rf
#> ScenicGrn (GRN results)
#>   No genes:                 5430 
#>   No TFs:                   466 
#>   Applied learner:          randomforest 
#>   TF to gene generated:     FALSE 
#>   CisTarget res generated:  FALSE
```

### GRNBoost2

Not available on GPU. The wrapper fails fast with a clear message before
touching the device.

``` r

scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tf_dt$gene_id,
  genes_to_take = scenic_genes,
  scenic_params = params_scenic(learner_type = "grnboost2"),
  .verbose = FALSE
)
#> Error in `method(scenic_grn_sc_gpu, bixverse::SingleCells)`:
#> ! GRNBoost2 (gradient boosting) is not supported on GPU. Use bixverse::scenic_grn_sc() for the CPU implementation.
```

Fall back to
[`bixverse::scenic_grn_sc()`](https://gregorlueg.github.io/bixverse/reference/scenic_grn_sc.html)
if you need it.

### Streaming vs non-streaming

On disk-backed `SingleCells` objects both drivers exist. Non-streaming
loads the full sparse target column vector into host RAM up front and
dispatches wave by wave. Streaming reads targets in I/O chunks and
dispatches per chunk, capping peak host memory at roughly one chunk of
sparse columns.

Set `streaming = FALSE` to force the non-streaming path when host RAM
isn’t the limiting factor:

``` r

scenic_res_streaming <- scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tf_dt$gene_id,
  genes_to_take = scenic_genes,
  scenic_params = params_scenic(
    learner_type = "extratrees",
    gene_batch_size = 64L
  ),
  streaming = TRUE,
  .verbose = TRUE
)
```

Leave `streaming = NULL` to let the same `auto_streaming()` heuristic
that the CPU version uses pick for you.

## TF-to-gene refinement

Because the result is a `ScenicGrn` object, the CPU downstream steps
drop straight in.

``` r

scenic_res_et <- identify_tf_to_genes(
  scenic_res_et,
  n_sd = 1,
  .verbose = TRUE
)
#> Extracting TF to gene associations via per-gene threshold (mean + 1.0 * SD).

scenic_res_et <- tf_to_genes_correlations(
  x = scenic_res_et,
  object = sc_object,
  cor_filter = 0.01,
  .verbose = TRUE
)
#> Calculating the pairwise correlations between the TFs and genes
#> Removing TF <> gene pairs with cors <= 0.010
#> Removing self loops (TF controlling its own expression

tf_gene_dt <- get_tf_to_gene(scenic_res_et)
tf_gene_dt[, tf_symbol := ensembl_to_symbol[tf]]
tf_gene_dt[, gene_symbol := ensembl_to_symbol[gene]]

head(tf_gene_dt[order(-importance)], 5L)
#>                 tf            gene importance pairwise_cor tf_symbol
#>             <char>          <char>      <num>        <num>    <char>
#> 1: ENSG00000139187 ENSG00000113088  0.2567620    0.2531497     KLRG1
#> 2: ENSG00000171223 ENSG00000120129  0.2535830    0.3743879      JUNB
#> 3: ENSG00000139187 ENSG00000161570  0.2012684    0.3895632     KLRG1
#> 4: ENSG00000138795 ENSG00000166681  0.2005688    0.1963190      LEF1
#> 5: ENSG00000221869 ENSG00000115828  0.1988880    0.2871167     CEBPD
#>    gene_symbol
#>         <char>
#> 1:        GZMK
#> 2:       DUSP1
#> 3:        CCL5
#> 4:     NGFRAP1
#> 5:        QPCT
```

## MetaCells path

The GPU driver also dispatches on `MetaCells`. Counts already live in
memory as a sparse CSC block, so the streaming flag has no effect (and
is silently ignored). Build the metacells the same way as the CPU
vignette:

``` r

mc_object <- generate_bt_meta_cells_sc(
  sc_object,
  sc_meta_cell_params = params_sc_bt_metacells(
    target_no_metacells = 200L
  ),
  .verbose = FALSE
)
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.

scenic_res_mc <- scenic_grn_sc_gpu(
  object = mc_object,
  tf_ids = tf_dt$gene_id,
  scenic_params = params_scenic(
    learner_type = "extratrees",
    min_counts = 50L,
    min_cells = 0.02,
    gene_batch_size = 64L
  ),
  .verbose = TRUE
)
#> The mean leafs per sample is set quite high for meta cells. Reducing to 10L.
#> No target genes supplied, running CPU gene filter...
#> SCENIC gene filter: 7566 / 11139 genes pass.
#> SCENIC GPU: 7566 target genes, 703 TFs, 200 cells

scenic_res_mc
#> ScenicGrn (GRN results)
#>   No genes:                 7566 
#>   No TFs:                   703 
#>   Applied learner:          extratrees 
#>   TF to gene generated:     FALSE 
#>   CisTarget res generated:  FALSE
```

Same result class, same downstream methods.
[`identify_tf_to_genes()`](https://gregorlueg.github.io/bixverse/reference/identify_tf_to_genes.html),
[`tf_to_genes_correlations()`](https://gregorlueg.github.io/bixverse/reference/tf_to_genes_correlations.html)
and
[`tf_to_genes_motif_enrichment()`](https://gregorlueg.github.io/bixverse/reference/tf_to_genes_motif_enrichment.html)
all work unchanged.

## CisTarget motif enrichment

Nothing changes on the CisTarget side. Same rankings, same annotations,
same result. Not evaluated by default because the reference files are
around 1.5 GB.

``` r

paths <- download_cistarget_hg38()

rankings <- read_motif_ranking(paths$rankings)
annotations <- read_motif_annotation_file(paths$motif_annotations)

scenic_res_et <- tf_to_genes_motif_enrichment(
  x = scenic_res_et,
  motif_rankings = rankings,
  annot_data = annotations,
  gene_id_to_symbol = ensembl_to_symbol,
  cis_target_params = params_cistarget(nes_threshold = 3.0),
  only_high_conf_tf = FALSE,
  .verbose = TRUE
)

cistarget_dt <- get_cistarget_res(scenic_res_et)
head(cistarget_dt[order(-nes)])
```

## Choosing between CPU and GPU

Numbers first. M1 Max (8P+2E cores, 32 GPU cores, unified memory), 10k
cells, 1000 TFs, 4000 target genes in roughly 63 batches, 250 trees,
`max_depth = 10`, `min_samples_leaf = 50`:

| Learner      | CPU     | GPU    | Speed-up |
|--------------|---------|--------|----------|
| ExtraTrees   | 113.09s | 56.66s | 2.00x    |
| RandomForest | 87.02s  | 48.10s | 1.81x    |

That is end to end, gene filtering and batching included. Look at a
single 64-target batch against one CPU core and the GPU side is 21.3x
(ExtraTrees) and 23.6x (RandomForest). The end-to-end number is lower
because the CPU version parallelises across all cores, so you’re really
comparing 32 GPU cores against 10 CPU cores. RandomForest ends up the
fastest absolute option here: `subsample_rate = 0.632` means each tree
only sees 63% of the cells.

Fidelity against the CPU implementation is per-target Pearson 0.987 for
RandomForest and 0.975 with bootstrapping on. Same trees, different
scheduling, so the importances agree to within sampling noise.

Where the GPU still loses:

- Small data (a few thousand cells, a few hundred targets). Fixed setup
  cost eats the win. Stay on CPU.
- GRNBoost2. Its single-target histogram-subtraction algorithm shares no
  structure across targets and gains nothing from a GPU port, so
  [`scenic_grn_sc_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/scenic_grn_sc_gpu.md)
  rejects it outright. Use
  [`bixverse::scenic_grn_sc()`](https://gregorlueg.github.io/bixverse/reference/scenic_grn_sc.html).

RandomForest used to belong on that list. Its per-wave tensors needed
roughly 8.4 GB at a million cells even at wave size 1, and the driver
refused rather than thrashing. The fused-kernel rewrite moved the
histogram into threadgroup memory and collapsed that to about 10 MB, so
RF now scales like ExtraTrees and runs at cell counts where it
previously would not start.

What breaks first at very large cell counts is the packed feature
tensor, one byte per (cell, TF) at four bins per `u32` word, so 1 GB at
1000 TFs and a million cells. The driver checks that against the
adapter’s per-binding limit up front. If you hit it, subsample cells via
`n_subsample` or trim the TF list.

`wave_byte_budget` is the knob for the middle ground. It caps VRAM for
the per-wave histogram and cumulative tensors. Shrink it on an 8 GB
adapter you’re sharing with a display; raise it on 16 GB+ to let the
scheduler pick a wider wave.
