# gpu nmf ----------------------------------------------------------------------

library(magrittr)
library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "nmf_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

min_lib_size <- 300L
min_genes_exp <- 45L
min_cells_exp <- 500L
hvg_to_keep <- 60L
no_pcs <- 10L
target_n_metacells <- 60L

k_default <- 4L
n_runs_default <- 4L
k_range_default <- 2:5

# 100 iterations is plenty for a fixture this size and keeps the suite quick.
hals_params <- params_nmf_hals(max_iter = 100L)

# The density filter is flaky at small `n_runs`: a handful of restarts gives the
# local-density estimate almost nothing to work with, and a run that drops
# below `k` survivors errors out. Switched off here (2 is the cosine ceiling) so
# the tests measure the solver rather than the filter.
consensus_params <- params_nmf_consensus(density_threshold = 2)

## synthetic single-cell data --------------------------------------------------

single_cell_test_data <- generate_single_cell_test_data()

## SingleCells object ----------------------------------------------------------

sc_dir <- file.path(test_temp_dir, "sc")
dir.create(sc_dir, showWarnings = FALSE)

sc_object <- SingleCells(dir_data = sc_dir)

sc_object <- load_r_data(
  object = sc_object,
  counts = single_cell_test_data$counts,
  obs = single_cell_test_data$obs,
  var = single_cell_test_data$var,
  sc_qc_param = params_sc_min_quality(
    min_unique_genes = min_genes_exp,
    min_lib_size = min_lib_size,
    min_cells = min_cells_exp
  ),
  streaming = 0L,
  .verbose = FALSE
)

sc_object <- find_hvg_sc(sc_object, hvg_no = hvg_to_keep, .verbose = FALSE)
sc_object <- calculate_pca_sc(sc_object, no_pcs = no_pcs, .verbose = FALSE)
sc_object <- find_neighbours_sc(
  sc_object,
  neighbours_params = params_sc_neighbours(knn = list(k = 15L)),
  .verbose = FALSE
)

n_cells_kept <- length(get_cells_to_keep(sc_object))

## MetaCells object ------------------------------------------------------------

mc_object <- generate_bt_meta_cells_sc(
  sc_object,
  sc_meta_cell_params = params_sc_bt_metacells(
    target_no_metacells = target_n_metacells
  ),
  .verbose = FALSE
)
mc_object <- find_hvg_sc(mc_object, hvg_no = hvg_to_keep, .verbose = FALSE)

n_meta_cells <- nrow(get_sc_obs(mc_object))

## single run ------------------------------------------------------------------

sc_single <- nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  nmf_hals_params = hals_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(sc_single, "NmfResult"),
  info = "nmf gpu (SingleCells) - returns an NmfResult"
)

expect_equal(
  current = dim(get_w(sc_single)),
  target = c(hvg_to_keep, k_default),
  info = "nmf gpu (SingleCells) - W is genes x k"
)

expect_equal(
  current = dim(get_h(sc_single)),
  target = c(k_default, n_cells_kept),
  info = "nmf gpu (SingleCells) - H is k x cells"
)

expect_true(
  current = min(get_w(sc_single)) >= 0 && min(get_h(sc_single)) >= 0,
  info = "nmf gpu (SingleCells) - factors are non-negative"
)

expect_equal(
  current = rownames(get_w(sc_single)),
  target = sc_single$gene_ids,
  info = "nmf gpu (SingleCells) - W rows carry the gene ids"
)

expect_equal(
  current = colnames(get_h(sc_single)),
  target = sc_single$cell_ids,
  info = "nmf gpu (SingleCells) - H columns carry the cell ids"
)

expect_true(
  current = checkmate::testNumber(sc_single$final_loss, lower = 0),
  info = "nmf gpu (SingleCells) - final loss is a non-negative number"
)

expect_true(
  current = checkmate::testFlag(sc_single$converged),
  info = "nmf gpu (SingleCells) - convergence flag is a boolean"
)

## stabilised ------------------------------------------------------------------

sc_stab <- stabilised_nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(sc_stab, "StabilisedNmfResult"),
  info = "stabilised nmf gpu (SingleCells) - returns a StabilisedNmfResult"
)

expect_equal(
  current = dim(get_w(sc_stab)),
  target = c(hvg_to_keep, k_default * n_runs_default),
  info = "stabilised nmf gpu (SingleCells) - w_all is genes x (k * n_runs)"
)

expect_equal(
  current = length(sc_stab$losses),
  target = n_runs_default,
  info = "stabilised nmf gpu (SingleCells) - one loss per restart"
)

expect_equal(
  current = length(sc_stab$h_per_run),
  target = n_runs_default,
  info = "stabilised nmf gpu (SingleCells) - one H per restart"
)

expect_true(
  current = sc_stab$best_idx >= 1L && sc_stab$best_idx <= n_runs_default,
  info = "stabilised nmf gpu (SingleCells) - best_idx is a 1-based run index"
)

expect_equal(
  current = which.min(sc_stab$losses),
  target = sc_stab$best_idx,
  info = "stabilised nmf gpu (SingleCells) - best_idx points at the lowest loss"
)

expect_true(
  current = inherits(get_best_run(sc_stab), "NmfResult"),
  info = "stabilised nmf gpu (SingleCells) - get_best_run gives an NmfResult"
)

## consensus -------------------------------------------------------------------

sc_cons <- consensus_nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(sc_cons, "ConsensusNmfResult"),
  info = "consensus nmf gpu (SingleCells) - returns a ConsensusNmfResult"
)

expect_equal(
  current = dim(get_w(sc_cons)),
  target = c(hvg_to_keep, k_default),
  info = "consensus nmf gpu (SingleCells) - W is genes x k"
)

expect_equal(
  current = dim(get_h(sc_cons)),
  target = c(k_default, n_cells_kept),
  info = "consensus nmf gpu (SingleCells) - H is k x cells"
)

expect_true(
  current = min(get_w(sc_cons)) >= 0 && min(get_h(sc_cons)) >= 0,
  info = "consensus nmf gpu (SingleCells) - factors are non-negative"
)

expect_true(
  current = sc_cons$stability >= -1 && sc_cons$stability <= 1,
  info = "consensus nmf gpu (SingleCells) - stability is a mean silhouette"
)

expect_equal(
  current = length(sc_cons$rel_run_errors),
  target = n_runs_default,
  info = "consensus nmf gpu (SingleCells) - one relative error per restart"
)

expect_equal(
  current = nrow(sc_cons$clusters),
  target = k_default * n_runs_default,
  info = "consensus nmf gpu (SingleCells) - one cluster row per pooled component"
)

# The filter is off, so every pooled component has to survive into a cluster.
expect_equal(
  current = sum(sc_cons$cluster_sizes$n),
  target = k_default * n_runs_default,
  info = "consensus nmf gpu (SingleCells) - cluster sizes account for all components"
)

expect_true(
  current = checkmate::testNames(
    names(get_stability(sc_cons)),
    must.include = c("stability", "rel_error", "rel_run_errors")
  ),
  info = "consensus nmf gpu (SingleCells) - get_stability exposes the diagnostics"
)

## k sweep ---------------------------------------------------------------------

sc_sweep <- nmf_k_sweep_gpu_sc(
  object = sc_object,
  k_range = k_range_default,
  n_runs = 3L,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(sc_sweep, "NmfKSweepResult"),
  info = "nmf k sweep gpu (SingleCells) - returns an NmfKSweepResult"
)

expect_true(
  current = checkmate::testDataTable(sc_sweep),
  info = "nmf k sweep gpu (SingleCells) - is data.table backed"
)

expect_equal(
  current = sc_sweep$k,
  target = as.integer(k_range_default),
  info = "nmf k sweep gpu (SingleCells) - one row per k, in the order requested"
)

expect_true(
  current = checkmate::testNames(
    names(sc_sweep),
    must.include = c(
      "k",
      "stability",
      "best_error",
      "median_error",
      "consensus_failed",
      "n_dropped",
      "n_empty_clusters",
      "n_converged"
    )
  ),
  info = "nmf k sweep gpu (SingleCells) - all diagnostic columns present"
)

expect_true(
  current = all(sc_sweep$best_error <= sc_sweep$median_error),
  info = "nmf k sweep gpu (SingleCells) - best error never exceeds the median"
)

expect_true(
  current = all(diff(sc_sweep$best_error) <= 0),
  info = "nmf k sweep gpu (SingleCells) - reconstruction error falls as k grows"
)

expect_true(
  current = inherits(plot(sc_sweep), "ggplot"),
  info = "nmf k sweep gpu (SingleCells) - plot() gives a ggplot"
)

## MetaCells dispatch ----------------------------------------------------------

mc_single <- nmf_gpu_sc(
  object = mc_object,
  k = k_default,
  nmf_hals_params = hals_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(mc_single, "NmfResult"),
  info = "nmf gpu (MetaCells) - returns an NmfResult"
)

expect_equal(
  current = dim(get_h(mc_single)),
  target = c(k_default, n_meta_cells),
  info = "nmf gpu (MetaCells) - H is k x meta cells"
)

expect_equal(
  current = mc_single$source_class,
  target = "MetaCells",
  info = "nmf gpu (MetaCells) - result records the source class"
)

mc_stab <- stabilised_nmf_gpu_sc(
  object = mc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(mc_stab, "StabilisedNmfResult"),
  info = "stabilised nmf gpu (MetaCells) - returns a StabilisedNmfResult"
)

mc_cons <- consensus_nmf_gpu_sc(
  object = mc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(mc_cons, "ConsensusNmfResult"),
  info = "consensus nmf gpu (MetaCells) - returns a ConsensusNmfResult"
)

expect_equal(
  current = dim(get_w(mc_cons)),
  target = c(hvg_to_keep, k_default),
  info = "consensus nmf gpu (MetaCells) - W is genes x k"
)

mc_sweep <- nmf_k_sweep_gpu_sc(
  object = mc_object,
  k_range = 2:4,
  n_runs = 3L,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  .verbose = FALSE
)

expect_true(
  current = inherits(mc_sweep, "NmfKSweepResult"),
  info = "nmf k sweep gpu (MetaCells) - returns an NmfKSweepResult"
)

expect_true(
  current = all(diff(mc_sweep$best_error) <= 0),
  info = "nmf k sweep gpu (MetaCells) - reconstruction error falls as k grows"
)

## gene and cell subsets -------------------------------------------------------

subset_cells <- get_cell_names(sc_object)[
  get_cells_to_keep(sc_object)[1:200] + 1L
]
subset_genes <- get_gene_names(sc_object)[get_hvg(sc_object)[1:25] + 1L]

sc_subset <- nmf_gpu_sc(
  object = sc_object,
  k = 3L,
  cell_ids = subset_cells,
  gene_ids = subset_genes,
  nmf_hals_params = hals_params,
  .verbose = FALSE
)

expect_equal(
  current = dim(get_w(sc_subset)),
  target = c(length(subset_genes), 3L),
  info = "nmf gpu (SingleCells) - gene subset shapes W"
)

expect_equal(
  current = dim(get_h(sc_subset)),
  target = c(3L, length(subset_cells)),
  info = "nmf gpu (SingleCells) - cell subset shapes H"
)

expect_equal(
  current = sort(colnames(get_h(sc_subset))),
  target = sort(subset_cells),
  info = "nmf gpu (SingleCells) - H columns are the requested cells"
)

## reproducibility -------------------------------------------------------------

sc_seed_a <- nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  nmf_hals_params = hals_params,
  seed = 11L,
  .verbose = FALSE
)
sc_seed_b <- nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  nmf_hals_params = hals_params,
  seed = 11L,
  .verbose = FALSE
)

expect_equal(
  current = get_w(sc_seed_a),
  target = get_w(sc_seed_b),
  info = "nmf gpu - same seed reproduces the same loadings"
)

cons_seed_a <- consensus_nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  seed = 11L,
  .verbose = FALSE
)
cons_seed_b <- consensus_nmf_gpu_sc(
  object = sc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  seed = 11L,
  .verbose = FALSE
)

expect_equal(
  current = cons_seed_a$stability,
  target = cons_seed_b$stability,
  info = "consensus nmf gpu - same seed reproduces the same stability"
)

## gpu vs cpu ------------------------------------------------------------------

# f32 GEMM on the device reduces in a different order than the CPU path, so the
# factors are not bit-identical. What has to hold is that both solvers land on
# the same reconstruction quality.
cpu_single <- nmf_sc(
  object = sc_object,
  k = k_default,
  nmf_hals_params = hals_params,
  seed = 11L,
  .verbose = FALSE
)

expect_true(
  current = abs(sc_seed_a$final_loss - cpu_single$final_loss) /
    cpu_single$final_loss <
    0.01,
  info = "nmf gpu vs cpu - final loss within 1%"
)

cpu_cons <- consensus_nmf_sc(
  object = sc_object,
  k = k_default,
  n_runs = n_runs_default,
  nmf_hals_params = hals_params,
  nmf_consensus_params = consensus_params,
  seed = 11L,
  .verbose = FALSE
)

expect_true(
  current = abs(cons_seed_a$rel_error - cpu_cons$rel_error) /
    cpu_cons$rel_error <
    0.01,
  info = "consensus nmf gpu vs cpu - relative error within 1%"
)

expect_true(
  current = abs(cons_seed_a$stability - cpu_cons$stability) < 0.05,
  info = "consensus nmf gpu vs cpu - stability agrees"
)

## input validation ------------------------------------------------------------

expect_error(
  current = nmf_gpu_sc(
    object = sc_object,
    k = bixverse.gpu:::NMF_GPU_MAX_RANK + 1L,
    .verbose = FALSE
  ),
  pattern = "caps the rank",
  info = "nmf gpu - rank above the GPU cap is refused"
)

expect_error(
  current = nmf_k_sweep_gpu_sc(
    object = sc_object,
    k_range = c(2L, bixverse.gpu:::NMF_GPU_MAX_RANK + 1L),
    .verbose = FALSE
  ),
  pattern = "caps the rank",
  info = "nmf k sweep gpu - a k_range entry above the cap is refused"
)

expect_error(
  current = consensus_nmf_gpu_sc(
    object = sc_object,
    k = 1L,
    n_runs = n_runs_default,
    .verbose = FALSE
  ),
  info = "consensus nmf gpu - k below 2 is refused"
)

expect_error(
  current = consensus_nmf_gpu_sc(
    object = sc_object,
    k = k_default,
    n_runs = 1L,
    .verbose = FALSE
  ),
  info = "consensus nmf gpu - a single restart is refused"
)

expect_error(
  current = nmf_k_sweep_gpu_sc(
    object = sc_object,
    k_range = 1:3,
    .verbose = FALSE
  ),
  info = "nmf k sweep gpu - a k_range reaching below 2 is refused"
)

expect_error(
  current = nmf_gpu_sc(
    object = sc_object,
    k = k_default,
    nmf_hals_params = list(max_iter = 100L),
    .verbose = FALSE
  ),
  info = "nmf gpu - an incomplete hals param list is refused"
)

expect_error(
  current = consensus_nmf_gpu_sc(
    object = sc_object,
    k = k_default,
    n_runs = n_runs_default,
    nmf_consensus_params = params_nmf_consensus(density_threshold = 1e-9),
    .verbose = FALSE
  ),
  pattern = "density",
  info = "consensus nmf gpu - an unsatisfiable density filter errors actionably"
)

expect_error(
  current = nmf_gpu_sc(
    object = sc_object,
    k = k_default,
    preprocessing = "garbage",
    .verbose = FALSE
  ),
  info = "nmf gpu - an unknown preprocessing mode is refused"
)
