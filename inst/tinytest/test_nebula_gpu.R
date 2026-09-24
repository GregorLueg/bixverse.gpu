# gpu nebula -------------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

# the NEBULA kernel reduces within a plane; the paravirtualised GPU of the
# macos-15-intel runners advertises plane operations but never runs them
if (!bixverse.gpu:::rs_gpu_plane_ops()) {
  exit_file("GPU adapter does not run plane operations")
}

library(magrittr)
library(bixverse)

test_temp_dir <- file.path(tempdir(), "nebula_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

min_lib_size <- 300L
min_genes_exp <- 45L
min_cells_exp <- 500L
n_samples <- 6L
n_genes_test <- 25L

# Stage two runs in f32 on the device. On this fixture the largest relative
# CPU vs GPU gap measured 1.8e-4 (subject overdispersion); this leaves headroom.
gpu_rel_tol <- 1e-3

## synthetic single-cell data --------------------------------------------------

single_cell_test_data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_samples = n_samples,
    sample_bias = "even"
  )
)

# sample_id is the subject the cells came from, condition a per-subject label
single_cell_test_data$obs[,
  condition := ifelse(
    sample_id %in% c("sample_1", "sample_2", "sample_3"),
    "ctr",
    "trt"
  )
]

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
    min_cells = min_cells_exp,
    target_size = 1000
  ),
  streaming = 0L,
  .verbose = FALSE
)

genes_to_test <- head(get_gene_names(sc_object), n_genes_test)

## helpers ---------------------------------------------------------------------

#' Maximum relative difference between two numeric vectors
max_rel_diff <- function(a, b) {
  max(abs(a - b) / pmax(abs(a), 1e-12))
}

cols_to_compare <- c(
  "log_fc",
  "effect_se",
  "z",
  "p_value",
  "fdr",
  "subject_overdispersion",
  "cell_overdispersion",
  "cell_overdispersion_shrunk"
)

# tests ------------------------------------------------------------------------

## parameters ------------------------------------------------------------------

nebula_params <- params_nebula_gpu()

expect_true(
  current = bixverse.gpu:::checkNebulaGpuParams(nebula_params),
  info = "the default GPU NEBULA params pass their own check"
)

expect_false(
  current = "reml" %in% names(nebula_params),
  info = "GPU NEBULA params do not carry reml"
)

expect_equal(
  current = nebula_params[names(nebula_params)],
  target = bixverse::params_nebula()[names(nebula_params)],
  info = "GPU NEBULA params share the CPU defaults"
)

expect_error(
  current = params_nebula_gpu(nebula_method = "nonsense"),
  info = "params_nebula_gpu rejects an unknown method"
)

expect_error(
  current = params_nebula_gpu(min_sigma = 20, max_sigma = 10),
  pattern = "below",
  info = "params_nebula_gpu rejects min_sigma above max_sigma"
)

expect_error(
  current = params_nebula_gpu(min_phi = 20, max_phi = 10),
  pattern = "below",
  info = "params_nebula_gpu rejects min_phi above max_phi"
)

expect_error(
  current = params_nebula_gpu(gene_batch_size = 0L),
  info = "params_nebula_gpu rejects a zero gene batch size"
)

expect_true(
  current = is.character(bixverse.gpu:::checkNebulaGpuParams(
    utils::modifyList(nebula_params, list(min_sigma = 100))
  )),
  info = "checkNebulaGpuParams catches bounds that cross"
)

## input validation ------------------------------------------------------------

expect_error(
  current = nebula_gpu_sc(
    object = sc_object,
    subject_col = "not_a_column",
    design = ~condition,
    genes_to_use = genes_to_test,
    .verbose = FALSE
  ),
  info = "nebula_gpu_sc rejects a subject column that is not in the obs table"
)

expect_error(
  current = nebula_gpu_sc(
    object = sc_object,
    subject_col = "sample_id",
    design = ~not_a_column,
    genes_to_use = genes_to_test,
    .verbose = FALSE
  ),
  info = "nebula_gpu_sc rejects a design term that is not in the obs table"
)

expect_error(
  current = nebula_gpu_sc(
    object = sc_object,
    subject_col = "sample_id",
    design = ~condition,
    genes_to_use = genes_to_test,
    nebula_params = c(params_nebula_gpu(), list(reml = TRUE)),
    .verbose = FALSE
  ),
  pattern = "reml",
  info = "nebula_gpu_sc rejects reml, which the device fit does not implement"
)

## cpu vs gpu: null contrast ---------------------------------------------------

cpu_res <- bixverse::nebula_sc(
  object = sc_object,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

gpu_res <- nebula_gpu_sc(
  object = sc_object,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

expect_equal(
  current = class(gpu_res),
  target = class(cpu_res),
  info = "GPU NEBULA returns the CPU result class"
)

expect_equal(
  current = names(gpu_res$results),
  target = names(cpu_res$results),
  info = "GPU NEBULA results carry the CPU columns"
)

expect_equal(
  current = gpu_res$results$gene_id,
  target = cpu_res$results$gene_id,
  info = "GPU and CPU NEBULA keep the same genes"
)

for (col in cols_to_compare) {
  expect_true(
    current = max_rel_diff(cpu_res$results[[col]], gpu_res$results[[col]]) <
      gpu_rel_tol,
    info = sprintf("GPU NEBULA `%s` within tolerance of CPU", col)
  )
}

expect_equal(
  current = gpu_res$results$sigma_at_bound,
  target = cpu_res$results$sigma_at_bound,
  info = "GPU and CPU NEBULA agree on sigma_at_bound"
)

expect_equal(
  current = gpu_res$results$convergence,
  target = cpu_res$results$convergence,
  info = "GPU and CPU NEBULA agree on the convergence codes"
)

expect_equal(
  current = dimnames(gpu_res$coefficients),
  target = dimnames(cpu_res$coefficients),
  info = "GPU NEBULA coefficient matrix is named like the CPU one"
)

## cpu vs gpu: real contrast ---------------------------------------------------

# the synthetic cell groups carry marker genes, so this contrast has signal
cpu_grp <- bixverse::nebula_sc(
  object = sc_object,
  subject_col = "sample_id",
  design = ~cell_grp,
  coef = "cell_grpcell_type_2",
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

gpu_grp <- nebula_gpu_sc(
  object = sc_object,
  subject_col = "sample_id",
  design = ~cell_grp,
  coef = "cell_grpcell_type_2",
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

expect_true(
  current = sum(cpu_grp$results$fdr <= 0.05) > 0L,
  info = "the cell group contrast has significant genes on the CPU"
)

expect_equal(
  current = gpu_grp$results$fdr <= 0.05,
  target = cpu_grp$results$fdr <= 0.05,
  info = "GPU and CPU NEBULA call the same genes on a real contrast"
)

expect_true(
  current = max_rel_diff(cpu_grp$results$log_fc, gpu_grp$results$log_fc) <
    gpu_rel_tol,
  info = "GPU NEBULA log_fc within tolerance of CPU on a real contrast"
)

expect_true(
  current = max_rel_diff(cpu_grp$results$z, gpu_grp$results$z) < gpu_rel_tol,
  info = "GPU NEBULA z within tolerance of CPU on a real contrast"
)

## determinism and batching ----------------------------------------------------

gpu_rerun <- nebula_gpu_sc(
  object = sc_object,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

expect_equal(
  current = gpu_rerun$results,
  target = gpu_res$results,
  info = "GPU NEBULA is deterministic across reruns"
)

gpu_small_batch <- nebula_gpu_sc(
  object = sc_object,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = genes_to_test,
  nebula_params = params_nebula_gpu(gene_batch_size = 5L),
  .verbose = FALSE
)

expect_equal(
  current = gpu_small_batch$results,
  target = gpu_res$results,
  info = "GPU NEBULA does not depend on the gene batch size"
)

## SingleCellsSubset -----------------------------------------------------------

subset_object <- SingleCellsSubset(
  sc_object = sc_object,
  grouping_column = "cell_grp",
  group = "cell_type_1"
)

cpu_sub <- bixverse::nebula_sc(
  object = subset_object,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

gpu_sub <- nebula_gpu_sc(
  object = subset_object,
  subject_col = "sample_id",
  design = ~condition,
  genes_to_use = genes_to_test,
  .verbose = FALSE
)

expect_equal(
  current = gpu_sub$params$n_cells,
  target = cpu_sub$params$n_cells,
  info = "GPU NEBULA on a subset uses the subset's cells"
)

expect_equal(
  current = gpu_sub$results$gene_id,
  target = cpu_sub$results$gene_id,
  info = "GPU and CPU NEBULA keep the same genes on a subset"
)

expect_true(
  current = max_rel_diff(cpu_sub$results$z, gpu_sub$results$z) < gpu_rel_tol,
  info = "GPU NEBULA z within tolerance of CPU on a subset"
)

## clean up --------------------------------------------------------------------

on.exit(unlink(test_temp_dir, recursive = TRUE, force = TRUE), add = TRUE)
