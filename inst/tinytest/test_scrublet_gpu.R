# gpu scrublet -----------------------------------------------------------------

library(magrittr)
library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "scrublet_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

n_doublets <- 200L
no_pcs <- 15L
sim_ratio <- 1.0

## helper functions ------------------------------------------------------------

#' Precision, recall and F1 from a 2x2 confusion matrix
metrics_helper <- function(cm) {
  tp <- cm[2, 2]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- 2 * (precision * recall) / (precision + recall)
  c(precision = precision, recall = recall, f1 = f1)
}

## synthetic data with planted doublets ----------------------------------------

syn_data <- generate_single_cell_test_data(seed = 123L)

ct1_idx <- which(syn_data$obs$cell_grp == "cell_type_1")
ct2_idx <- which(syn_data$obs$cell_grp == "cell_type_2")
ct3_idx <- which(syn_data$obs$cell_grp == "cell_type_3")

n_12 <- ceiling(n_doublets / 3)
n_23 <- ceiling(n_doublets / 3)
n_13 <- n_doublets - n_12 - n_23

doublet_pairs <- rbind(
  cbind(
    sample(ct1_idx, n_12, replace = TRUE),
    sample(ct2_idx, n_12, replace = TRUE)
  ),
  cbind(
    sample(ct2_idx, n_23, replace = TRUE),
    sample(ct3_idx, n_23, replace = TRUE)
  ),
  cbind(
    sample(ct1_idx, n_13, replace = TRUE),
    sample(ct3_idx, n_13, replace = TRUE)
  )
)

doublet_matrix <- do.call(
  rbind,
  lapply(seq_len(nrow(doublet_pairs)), \(i) {
    syn_data$counts[doublet_pairs[i, 1], ] +
      syn_data$counts[doublet_pairs[i, 2], ]
  })
)

all_counts <- rbind(syn_data$counts, doublet_matrix)
n_total <- nrow(all_counts)

doublet_obs <- data.table::data.table(
  cell_id = sprintf("doublet_%04d", seq_len(n_doublets)),
  cell_grp = "doublet",
  batch_index = 1,
  doublet = TRUE
)
new_obs <- data.table::rbindlist(list(
  data.table::copy(syn_data$obs)[, doublet := FALSE],
  doublet_obs
))
new_obs[, sample_id := rep(c("sample_A", "sample_B"), length.out = .N)]

## SingleCells object ----------------------------------------------------------

sc_dir <- file.path(test_temp_dir, "sc")
dir.create(sc_dir, showWarnings = FALSE)

sc_object <- SingleCells(dir_data = sc_dir)

# QC is fully off: every planted doublet has to survive to be measurable
sc_object <- load_r_data(
  object = sc_object,
  counts = all_counts,
  obs = new_obs,
  var = syn_data$var,
  sc_qc_param = params_sc_min_quality(
    min_unique_genes = 0L,
    min_lib_size = 0L,
    min_cells = 0L
  ),
  streaming = 0L,
  .verbose = FALSE
)

truth <- get_sc_obs(sc_object)$doublet

## shared parameters -----------------------------------------------------------

# n_bins_histogram is pinned to 50L on both sides on purpose. The installed
# bixverse links against a bixverse-rs that reads only the `n_bins_hist` key and
# so always falls back to 50; passing 50 explicitly makes CPU and GPU agree
# both before and after that fix ships.
gpu_params <- params_scrublet_gpu(
  normalisation = list(target_size = 1e4),
  hvg = list(min_gene_var_pctl = 0.0),
  no_pcs = no_pcs,
  expected_doublet_rate = 0.2,
  sim_doublet_ratio = sim_ratio,
  n_bins_histogram = 50L
)

cpu_params <- params_scrublet(
  normalisation = list(target_size = 1e4),
  pca = list(no_pcs = no_pcs),
  hvg = list(min_gene_var_pctl = 0.0),
  expected_doublet_rate = 0.2,
  sim_doublet_ratio = sim_ratio,
  n_bins_histogram = 50L
)

# parameter wrappers -----------------------------------------------------------

expect_true(
  current = isTRUE(bixverse.gpu:::checkScrubletGpu(gpu_params)),
  info = "params_scrublet_gpu - default GPU list passes its own checker"
)

expect_equal(
  current = gpu_params$knn_backend,
  target = "gpu",
  info = "params_scrublet_gpu - defaults to the GPU kNN backend"
)

expect_equal(
  current = gpu_params$knn_method,
  target = "exhaustive",
  info = "params_scrublet_gpu - GPU arm defaults to exact exhaustive search"
)

expect_equal(
  current = gpu_params$k,
  target = 0L,
  info = "params_scrublet_gpu - k defaults to 0L (Rust picks it)"
)

expect_true(
  current = "n_bins_hist" %in% names(gpu_params),
  info = "params_scrublet_gpu - emits the n_bins_hist key Rust reads"
)

expect_false(
  current = any(c("random_svd", "sparse") %in% names(gpu_params)),
  info = "params_scrublet_gpu - drops the CPU-only PCA knobs"
)

## cpu backend arm -------------------------------------------------------------

cpu_arm_params <- params_scrublet_gpu(
  knn_backend = "cpu",
  knn = list(knn_method = "hnsw", m = 32L)
)

expect_true(
  current = isTRUE(bixverse.gpu:::checkScrubletGpu(cpu_arm_params)),
  info = "params_scrublet_gpu - CPU arm passes the checker"
)

expect_equal(
  current = cpu_arm_params$m,
  target = 32L,
  info = "params_scrublet_gpu - CPU arm keeps CPU-only kNN keys"
)

## rejections ------------------------------------------------------------------

expect_error(
  current = params_scrublet_gpu(knn = list(m = 32L)),
  pattern = "Unknown kNN parameter",
  info = "params_scrublet_gpu - CPU-only kNN key rejected on the GPU arm"
)

expect_error(
  current = params_scrublet_gpu(knn_backend = "cuda"),
  info = "params_scrublet_gpu - unknown backend rejected"
)

expect_false(
  current = isTRUE(bixverse.gpu:::checkScrubletGpu(
    utils::modifyList(gpu_params, list(knn_method = "kmknn"))
  )),
  info = "checkScrubletGpu - CPU kNN method rejected on the GPU arm"
)

expect_false(
  current = isTRUE(bixverse.gpu:::checkScrubletGpu(
    utils::modifyList(gpu_params, list(knn_backend = "cuda"))
  )),
  info = "checkScrubletGpu - unknown backend rejected"
)

expect_false(
  current = isTRUE(bixverse.gpu:::checkScrubletGpu(
    utils::modifyList(gpu_params, list(no_pcs = 0L))
  )),
  info = "checkScrubletGpu - no_pcs must be >= 1"
)

# rust layer -------------------------------------------------------------------

f_path_gene <- bixverse:::get_rust_count_gene_f_path(sc_object)
f_path_cell <- bixverse:::get_rust_count_cell_f_path(sc_object)
cells_to_keep <- get_cells_to_keep(sc_object)

rust_res <- rs_sc_scrublet_gpu(
  f_path_gene = f_path_gene,
  f_path_cell = f_path_cell,
  cells_to_keep = cells_to_keep,
  scrublet_params = gpu_params,
  seed = 42L,
  verbose = 0L,
  streaming = FALSE,
  return_combined_pca = TRUE,
  return_pairs = TRUE
)

expect_true(
  current = checkmate::qtest(
    rust_res$predicted_doublets,
    sprintf("B%s", n_total)
  ),
  info = "rs_sc_scrublet_gpu - one call per observed cell"
)

expect_true(
  current = checkmate::qtest(
    rust_res$doublet_scores_obs,
    sprintf("N%s", n_total)
  ),
  info = "rs_sc_scrublet_gpu - one observed score per cell"
)

expect_true(
  current = checkmate::qtest(
    rust_res$doublet_scores_sim,
    sprintf("N%s", n_total * sim_ratio)
  ),
  info = "rs_sc_scrublet_gpu - simulated scores follow sim_doublet_ratio"
)

expect_true(
  current = checkmate::qtest(
    rust_res$doublet_errors_obs,
    sprintf("N%s", n_total)
  ) &&
    checkmate::qtest(rust_res$z_scores, sprintf("N%s", n_total)),
  info = "rs_sc_scrublet_gpu - errors and z-scores are per observed cell"
)

expect_true(
  current = all(purrr::map_lgl(
    rust_res[c(
      "threshold",
      "detected_doublet_rate",
      "detectable_doublet_fraction",
      "overall_doublet_rate"
    )],
    \(x) checkmate::qtest(x, "N1")
  )),
  info = "rs_sc_scrublet_gpu - the four summary scalars are single numerics"
)

expect_true(
  current = checkmate::testMatrix(
    rust_res$pca,
    mode = "numeric",
    nrows = n_total + n_total * sim_ratio,
    ncols = no_pcs
  ),
  info = "rs_sc_scrublet_gpu - PCA covers observed plus simulated cells"
)

expect_true(
  current = checkmate::qtest(rust_res$pair_1, sprintf("I%s", n_total)) &&
    checkmate::qtest(rust_res$pair_2, sprintf("I%s", n_total)) &&
    all(c(rust_res$pair_1, rust_res$pair_2) < n_total),
  info = "rs_sc_scrublet_gpu - doublet parents are in-range cell indices"
)

## optional returns withheld ---------------------------------------------------

rust_lean <- rs_sc_scrublet_gpu(
  f_path_gene = f_path_gene,
  f_path_cell = f_path_cell,
  cells_to_keep = cells_to_keep,
  scrublet_params = gpu_params,
  seed = 42L,
  verbose = 0L,
  streaming = FALSE,
  return_combined_pca = FALSE,
  return_pairs = FALSE
)

expect_true(
  current = is.null(rust_lean$pca) &&
    is.null(rust_lean$pair_1) &&
    is.null(rust_lean$pair_2),
  info = "rs_sc_scrublet_gpu - PCA and pairs are NULL when not requested"
)

## reproducibility -------------------------------------------------------------

# GPU reduction ordering can shift ties, so this asserts a near-identical run
# rather than bit equality
expect_true(
  current = cor(
    rust_lean$doublet_scores_obs,
    rust_res$doublet_scores_obs
  ) >
    0.999,
  info = "rs_sc_scrublet_gpu - same seed reproduces the scores"
)

## accuracy on planted doublets ------------------------------------------------

gpu_metrics <- metrics_helper(table(truth, rust_res$predicted_doublets))

expect_true(
  current = gpu_metrics[["recall"]] >= 0.7,
  info = "rs_sc_scrublet_gpu - recovers at least 70% of planted doublets"
)

expect_true(
  current = gpu_metrics[["f1"]] >= 0.7,
  info = "rs_sc_scrublet_gpu - F1 clears 0.7 on planted doublets"
)

# gpu vs cpu -------------------------------------------------------------------

# Both SVDs are randomised but draw different sketches, and the GPU index breaks
# neighbour ties differently, so the scores correlate rather than match. Otsu's
# threshold is a step function of the histogram bins, so a handful of borderline
# calls flip. Assert on structure and agreement, never on equality.

cpu_res <- rs_sc_scrublet(
  f_path_gene = f_path_gene,
  f_path_cell = f_path_cell,
  cells_to_keep = cells_to_keep,
  scrublet_params = cpu_params,
  seed = 42L,
  verbose = 0L,
  streaming = FALSE,
  return_combined_pca = FALSE,
  return_pairs = FALSE
)

expect_true(
  current = cor(
    rust_res$doublet_scores_obs,
    cpu_res$doublet_scores_obs,
    method = "pearson"
  ) >
    0.95,
  info = "scrublet gpu vs cpu - observed scores correlate (Pearson)"
)

# No Spearman assertion here on purpose. Doublet scores are neighbour-count
# fractions, so on this fixture 1200 cells share only ~34 distinct values and
# Spearman is dominated by tie-breaking. Measured: GPU vs CPU 0.918, but CPU vs
# CPU at two seeds is 0.888, so the statistic sits below its own noise floor and
# would gate on nothing.

expect_true(
  current = mean(
    rust_res$predicted_doublets == cpu_res$predicted_doublets
  ) >
    0.9,
  info = "scrublet gpu vs cpu - at least 90% of calls agree"
)

expect_true(
  current = abs(rust_res$threshold - cpu_res$threshold) /
    cpu_res$threshold <
    0.25,
  info = "scrublet gpu vs cpu - thresholds land within 25% of each other"
)

cpu_metrics <- metrics_helper(table(truth, cpu_res$predicted_doublets))

expect_true(
  current = abs(gpu_metrics[["f1"]] - cpu_metrics[["f1"]]) < 0.05,
  info = "scrublet gpu vs cpu - F1 within 0.05"
)

# knn backends -----------------------------------------------------------------

## cpu indices through the gpu driver ------------------------------------------

cpu_knn_params <- params_scrublet_gpu(
  normalisation = list(target_size = 1e4),
  hvg = list(min_gene_var_pctl = 0.0),
  no_pcs = no_pcs,
  expected_doublet_rate = 0.2,
  sim_doublet_ratio = sim_ratio,
  n_bins_histogram = 50L,
  knn_backend = "cpu",
  knn = list(knn_method = "hnsw", k = 0L)
)

cpu_knn_res <- rs_sc_scrublet_gpu(
  f_path_gene = f_path_gene,
  f_path_cell = f_path_cell,
  cells_to_keep = cells_to_keep,
  scrublet_params = cpu_knn_params,
  seed = 42L,
  verbose = 0L,
  streaming = FALSE,
  return_combined_pca = FALSE,
  return_pairs = FALSE
)

expect_true(
  current = cor(
    cpu_knn_res$doublet_scores_obs,
    rust_res$doublet_scores_obs
  ) >
    0.9,
  info = "scrublet gpu - CPU kNN backend agrees with the GPU one"
)

## the backend switch is live --------------------------------------------------

# Annoy is genuinely approximate and is a CPU-only method name. A
# `ScrubletParamsGpu::from_r_list` that ignored `knn_backend` and always built
# the GPU arm would not recognise "annoy", would fall back to GPU exhaustive,
# and would land back on `rust_res` exactly. Deliberately starved of trees and
# search budget so the approximation bites.
annoy_params <- params_scrublet_gpu(
  normalisation = list(target_size = 1e4),
  hvg = list(min_gene_var_pctl = 0.0),
  no_pcs = no_pcs,
  expected_doublet_rate = 0.2,
  sim_doublet_ratio = sim_ratio,
  n_bins_histogram = 50L,
  knn_backend = "cpu",
  knn = list(
    knn_method = "annoy",
    k = 15L,
    n_trees = 5L,
    search_budget = 20L
  )
)

annoy_res <- rs_sc_scrublet_gpu(
  f_path_gene = f_path_gene,
  f_path_cell = f_path_cell,
  cells_to_keep = cells_to_keep,
  scrublet_params = annoy_params,
  seed = 42L,
  verbose = 0L,
  streaming = FALSE,
  return_combined_pca = FALSE,
  return_pairs = FALSE
)

expect_false(
  current = identical(
    annoy_res$doublet_scores_obs,
    rust_res$doublet_scores_obs
  ),
  info = "scrublet gpu - knn_backend really selects the CPU indices"
)

## ivf on the gpu arm ----------------------------------------------------------

ivf_params <- params_scrublet_gpu(
  normalisation = list(target_size = 1e4),
  hvg = list(min_gene_var_pctl = 0.0),
  no_pcs = no_pcs,
  expected_doublet_rate = 0.2,
  sim_doublet_ratio = sim_ratio,
  n_bins_histogram = 50L,
  knn = list(knn_method = "ivf", k = 0L)
)

ivf_res <- rs_sc_scrublet_gpu(
  f_path_gene = f_path_gene,
  f_path_cell = f_path_cell,
  cells_to_keep = cells_to_keep,
  scrublet_params = ivf_params,
  seed = 42L,
  verbose = 0L,
  streaming = FALSE,
  return_combined_pca = FALSE,
  return_pairs = FALSE
)

expect_true(
  current = cor(ivf_res$doublet_scores_obs, rust_res$doublet_scores_obs) > 0.9,
  info = "scrublet gpu - IVF index agrees with exhaustive search"
)

# s7 method --------------------------------------------------------------------

obj_res <- scrublet_gpu_sc(
  object = sc_object,
  scrublet_params = gpu_params,
  seed = 42L,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(obj_res, "ScrubletRes"),
  info = "scrublet_gpu_sc - returns bixverse's ScrubletRes class"
)

expect_true(
  current = checkmate::qtest(attr(obj_res, "cell_indices"), "I+") &&
    identical(attr(obj_res, "cell_indices"), cells_to_keep),
  info = "scrublet_gpu_sc - carries the 0-indexed cell indices"
)

expect_true(
  current = cor(obj_res$doublet_scores_obs, rust_res$doublet_scores_obs) > 0.99,
  info = "scrublet_gpu_sc - matches the direct Rust call"
)

expect_true(
  current = checkmate::testClass(plot(obj_res), "ggplot"),
  info = "scrublet_gpu_sc - bixverse's plot method dispatches"
)

obj_dt <- get_data(obj_res)

expect_true(
  current = checkmate::testDataTable(obj_dt, nrows = n_total) &&
    checkmate::testNames(
      names(obj_dt),
      must.include = c("doublet", "doublet_score", "cell_idx")
    ),
  info = "scrublet_gpu_sc - get_data returns the expected obs table"
)

## manual threshold ------------------------------------------------------------

manual_res <- call_doublets_manual(
  obj_res,
  threshold = 0.175,
  .verbose = FALSE
)

expect_equal(
  current = manual_res$threshold,
  target = 0.175,
  info = "scrublet_gpu_sc - manual threshold is applied"
)

expect_true(
  current = identical(
    manual_res$predicted_doublets,
    manual_res$doublet_scores_obs > 0.175
  ),
  info = "scrublet_gpu_sc - manual threshold recomputes the calls"
)

## cells to use ----------------------------------------------------------------

subset_ids <- get_cell_names(sc_object)[seq(1L, n_total, by = 2L)]

subset_res <- scrublet_gpu_sc(
  object = sc_object,
  scrublet_params = gpu_params,
  cells_to_use = subset_ids,
  .verbose = FALSE
)

expect_equal(
  current = length(subset_res$predicted_doublets),
  target = length(subset_ids),
  info = "scrublet_gpu_sc - cells_to_use narrows the run"
)

expect_equal(
  current = length(attr(subset_res, "cell_indices")),
  target = length(subset_ids),
  info = "scrublet_gpu_sc - cell_indices track the requested subset"
)

# grouped runs -----------------------------------------------------------------

grouped_res <- scrublet_gpu_sc(
  object = sc_object,
  scrublet_params = gpu_params,
  group_by = "sample_id",
  .verbose = FALSE
)

expect_true(
  current = isTRUE(attr(grouped_res, "grouped")) &&
    identical(attr(grouped_res, "group_by_col"), "sample_id"),
  info = "scrublet_gpu_sc - grouped runs carry the grouping attributes"
)

expect_true(
  current = checkmate::testNumeric(
    grouped_res$threshold,
    names = "named",
    len = 2L
  ) &&
    setequal(names(grouped_res$threshold), c("sample_A", "sample_B")),
  info = "scrublet_gpu_sc - one threshold per group, named"
)

expect_true(
  current = checkmate::testCharacter(
    grouped_res$cell_groups,
    len = length(grouped_res$predicted_doublets)
  ),
  info = "scrublet_gpu_sc - group labels align with the predictions"
)

expect_false(
  current = is.unsorted(attr(grouped_res, "cell_indices")),
  info = "scrublet_gpu_sc - grouped results are reordered by cell index"
)

grouped_manual <- call_doublets_manual(
  grouped_res,
  threshold = 0.2,
  for_sample = "sample_A",
  .verbose = FALSE
)

expect_equal(
  current = grouped_manual$threshold[["sample_A"]],
  target = 0.2,
  info = "scrublet_gpu_sc - per-sample manual threshold hits the right group"
)

expect_equal(
  current = grouped_manual$threshold[["sample_B"]],
  target = grouped_res$threshold[["sample_B"]],
  info = "scrublet_gpu_sc - per-sample manual threshold leaves others alone"
)

# input validation -------------------------------------------------------------

expect_error(
  current = scrublet_gpu_sc(
    object = sc_object,
    scrublet_params = list(no_pcs = 15L),
    .verbose = FALSE
  ),
  info = "scrublet_gpu_sc - malformed params are rejected"
)

expect_error(
  current = scrublet_gpu_sc(
    object = sc_object,
    scrublet_params = gpu_params,
    cells_to_use = "not_a_cell_id",
    .verbose = FALSE
  ),
  info = "scrublet_gpu_sc - unknown cell identifiers are rejected"
)

expect_error(
  current = scrublet_gpu_sc(
    object = sc_object,
    scrublet_params = gpu_params,
    group_by = "does_not_exist",
    .verbose = FALSE
  ),
  info = "scrublet_gpu_sc - unknown group_by column is rejected"
)

# clean up ---------------------------------------------------------------------

on.exit(unlink(test_temp_dir, recursive = TRUE, force = TRUE), add = TRUE)
