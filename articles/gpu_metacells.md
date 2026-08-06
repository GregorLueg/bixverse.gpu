# GPU-accelerated SEACells

## Intro

SEACells ([Persad et al.,
2023](https://www.nature.com/articles/s41587-023-01716-9)) finds meta
cells by kernel archetypal analysis. The expensive part is the pair of
inner Frank-Wolfe loops: the `B` update scans a gradient over all cells
for every archetype and takes an argmin, and the `A` update solves a
column per cell. Both now run on the GPU in
[`generate_seacells_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_seacells_gpu_sc.md).

The rest stays on the CPU: the kNN graph, the kernel matrix, the
archetype initialisation, the `K²B` bookkeeping, the RSS evaluation and
the aggregation into pseudo-bulk counts. So the speedup is bounded by
how much of the run the two solves own. That share grows with
`n_sea_cells`, because both scale with `n_cells x k` while most of the
CPU-side work does not. Ask for 50 meta cells and the win is small. Ask
for a few thousand and it gets interesting.

Two things worth knowing about the kernels. Each solve picks a workgroup
width from a small set of tiers, and falls back to its CPU sibling for
that iteration if no tier covers your `k`, so a very large `n_sea_cells`
can quietly land back on the host. And nothing on the GPU path holds a
dense `n x k` buffer: `B`, `K²B` and the transposed `A` all stay sparse,
which is what makes million-cell runs fit at all.

If you have not seen the CPU version, read the [bixverse meta cell
vignette](https://gregorlueg.github.io/bixverse/articles/single_cell_meta_cells.html)
first. The params, the result class and everything downstream are
identical here.

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

## Rebuilding a processed CD34

We use the same CD34 cells from the [SEACells
vignette](https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_computation.ipynb)
to set up the parent `SingleCells` object: load (QC not needed - it’s
already filtered data), HVG selection, PCA, kNN graph. We can use the
provided cell type labels to check purity.

Rebuild the CD34 cell object (click to expand)

``` r

cd34_path <- download_cd34_data()

tempdir_cd34 <- tempdir()

sc_object <- SingleCells(dir_data = tempdir_cd34)

sc_object <- load_h5ad(object = sc_object, h5_path = cd34_path)
#>  Using light streaming for the CSR to CSC conversion.
#> Loading observations data from h5ad into the DuckDB.
#> duckdb is storing downloaded extensions and secrets under ~/.duckdb:
#> ℹ /Users/gregorlueg/.duckdb
#> This persists across sessions and is shared with the DuckDB CLI and other clients.
#> ℹ Run duckdb(shared_home = FALSE) to use a temporary directory instead.
#> ℹ See ?duckdb_storage for details and alternatives.
#> Loading variables data from h5ad into the DuckDB.
#> 
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

sc_object <- find_hvg_sc(
  object = sc_object,
  hvg_no = 2000L,
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

sc_object <- calculate_pca_sc(
  object = sc_object,
  no_pcs = 30L,
  sparse_svd = FALSE
)
#> Using dense SVD solving on scaled data on 2000 HVG.

sc_object <- find_neighbours_sc(
  object = sc_object,
  neighbours_params = params_sc_neighbours(
    knn = list(ann_dist = "euclidean", knn_method = "kmknn")
  )
)
#> 
#> Generating sNN graph (full: TRUE).
#> Transforming sNN data to igraph.

# we need the kNN object for the diffusion maps
knn_object <- get_knn_obj(x = sc_object)
```

## Running it

One call, and the params are the same
[`params_sc_seacells()`](https://gregorlueg.github.io/bixverse/reference/params_sc_seacells.html)
the CPU version takes. We will go with the same parameters as for the
other vignette.

``` r

seacell_params <- params_sc_seacells(
  n_sea_cells = 250L,
  convergence_epsilon = 0.001,
  pruning = TRUE
)

gpu_time <- system.time({
  mc_gpu <- generate_seacells_gpu_sc(
    sc_object,
    seacell_params = seacell_params,
    .verbose = TRUE
  )
})
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

mc_gpu
#> Single cell experiment (Meta Cells).
#>   Meta cell method: seacell
#>   No meta cells: 250
#>   No genes: 12464
#>   No cells aggregated: 6881
#>   No obs rows in source: 6881
#>   HVG calculated: FALSE
#>   PCA calculated: FALSE
#>   Other embeddings: none
#>   KNN generated: FALSE
#>   SNN generated: FALSE
```

The result is a `MetaCells` object, byte for byte the same shape as the
CPU one. Assignments live in `@original_assignment`, the RSS trace and
the archetype cell indices in `@other_data`.

``` r

head(mc_gpu[[]], 3L)
#>    meta_cell_idx  meta_cell_id no_originating_cells
#>            <int>        <char>                <num>
#> 1:             1 meta_cell_001                   49
#> 2:             2 meta_cell_002                   46
#> 3:             3 meta_cell_003                   25
#>                        original_cell_idx
#>                                   <list>
#> 1:        68,106,110,156,345,386,...[49]
#> 2:        65,108,205,229,737,751,...[46]
#> 3:   15,  25,  41, 567, 893,1242,...[25]

tail(mc_gpu@other_data$rss, 5L)
#> [1] 132.6602 132.3294 132.0941 131.8425 131.7310
```

### Knobs that matter on large data

Two params change the picture once you leave toy datasets:

- **`n_landmarks`** switches archetype initialisation to a Nystroem
  approximation on a landmark subsample. Without it, initialisation is
  quadratic in cells and starts to dominate the run well before the
  Frank-Wolfe loop does.
- **`pruning`** drops tiny values during the Frank-Wolfe updates. It
  costs a bit of numerical fidelity and buys a lot of memory, which is
  what makes million- cell runs feasible at all. `pruning_threshold`
  controls how aggressive it is.

``` r

seacell_params_large <- params_sc_seacells(
  n_sea_cells = 5000L,
  n_landmarks = 20000L,
  pruning = TRUE,
  pruning_threshold = 1e-7,
  knn = list(k = 25L)
)
```

## CPU versus GPU

Same object, same params, same seed.

``` r

cpu_time <- system.time({
  mc_cpu <- generate_seacells_sc(
    sc_object,
    seacell_params = seacell_params,
    .verbose = TRUE
  )
})
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

data.table(
  version = c("CPU", "GPU"),
  seconds = round(c(cpu_time[["elapsed"]], gpu_time[["elapsed"]]), 2)
)[, speed_up := round(seconds[1] / seconds, 2)][]
#>    version seconds speed_up
#>     <char>   <num>    <num>
#> 1:     CPU    6.89     1.00
#> 2:     GPU    7.14     0.96
```

The two do not produce bit-identical assignments. GPU reduction ordering
breaks ties differently, so a cell sitting almost equidistant between
two archetypes can land either way. What should agree is the structure.
Compare how the meta cells partition the cells:

``` r

mc_sizes <- data.table(
  version = c("CPU", "GPU"),
  n_meta_cells = c(nrow(mc_cpu[[]]), nrow(mc_gpu[[]])),
  median_cells = c(
    median(mc_cpu[[]]$no_originating_cells),
    median(mc_gpu[[]]$no_originating_cells)
  ),
  final_rss = round(
    c(
      tail(mc_cpu@other_data$rss, 1L),
      tail(mc_gpu@other_data$rss, 1L)
    ),
    2
  )
)

mc_sizes
#>    version n_meta_cells median_cells final_rss
#>     <char>        <int>        <num>     <num>
#> 1:     CPU          250         21.5    131.83
#> 2:     GPU          250         21.0    131.73
```

We can already see a difference here in speed, and this is a small data
set asking for only 250 meta cells. The more cells and SEACells you
throw at the problem, the bigger the delta: upstream measures the full
fit at roughly 2.6x on 50k cells and 666 archetypes.

## Feeding into SCENIC

The resulting `MetaCells` goes straight into the GPU SCENIC driver, no
conversion step.

``` r

scenic_res <- scenic_grn_sc_gpu(
  object = mc_gpu,
  tf_ids = tf_dt$gene_id,
  scenic_params = params_scenic(learner_type = "extratrees"),
  .verbose = TRUE
)
```

See the [GPU SCENIC
vignette](https://gregorlueg.github.io/bixverse.gpu/articles/gpu_scenic.md)
for the full workflow.

## What is next

Neither kernel is the bottleneck any more. Each beats the CPU scan it
replaces by several fold in isolation, but the fit as a whole moves by
less, and the gap is Amdahl: the blocking readbacks around the two
solves were about a fifth of wall-clock when last attributed, and the
RSS evaluation plus the `K²B` maintenance still sit on the host.
Overlapping the readbacks and moving RSS are the next two levers. Watch
the space.
