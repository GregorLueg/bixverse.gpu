# gpu scenic -------------------------------------------------------------------

library(magrittr)
library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "scenic_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

min_lib_size <- 300L
min_genes_exp <- 45L
min_cells_exp <- 500L
hvg_to_keep <- 30L
no_pcs <- 10L
target_n_metacells <- 200L

## synthetic single-cell data --------------------------------------------------

# same generator the CPU SCENIC tests use: plants marker structure in
# genes 1-30 across four cell types
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

## MetaCells object -----------------------------------------------------------

mc_object <- generate_bt_meta_cells_sc(
  sc_object,
  sc_meta_cell_params = params_sc_bt_metacells(
    target_no_metacells = target_n_metacells
  ),
  .verbose = FALSE
)

## scenic fixtures ------------------------------------------------------------

# planted TFs: some really do drive genes 1-30, others are decoy
tfs <- sprintf(
  "gene_%03i",
  c(1, 2, 11, 12, 21, 22, 50, 60, 70, 80, 90, 100)
)

# expected TF -> target set (marker gene blocks)
tf_to_gene_map <- list(
  sprintf("gene_%03i", 1:10),
  sprintf("gene_%03i", 1:10),
  sprintf("gene_%03i", 11:20),
  sprintf("gene_%03i", 11:20),
  sprintf("gene_%03i", 21:30),
  sprintf("gene_%03i", 21:30)
) %>%
  `names<-`(sprintf("gene_%03i", c(1, 2, 11, 12, 21, 22)))

# assertion helper: for each expected TF, does the mean importance of its
# planted targets exceed the mean importance of non-targets?
check_tf_importances <- function(x, tf_to_gene_map) {
  checkmate::assertMatrix(
    x,
    mode = "numeric",
    row.names = "named",
    col.names = "named"
  )
  checkmate::assertList(tf_to_gene_map)

  ok <- rep(FALSE, times = length(tf_to_gene_map))
  represented_genes <- rownames(x)

  for (i in seq_along(tf_to_gene_map)) {
    tf_i <- names(tf_to_gene_map)[[i]]
    if (!(tf_i %in% colnames(x))) next
    genes_i <- intersect(tf_to_gene_map[[i]], represented_genes)
    not_genes_i <- setdiff(represented_genes, genes_i)
    ok[[i]] <- mean(x[genes_i, tf_i]) > mean(x[not_genes_i, tf_i])
  }

  names(ok) <- names(tf_to_gene_map)
  ok
}

## shared target gene list ----------------------------------------------------

genes_to_keep_scenic <- scenic_gene_filter_sc(
  object = sc_object,
  scenic_params = params_scenic(min_counts = 500L),
  .verbose = FALSE
)

# tests ------------------------------------------------------------------------

## SingleCells - non-streaming (extra_trees) -----------------------------------

et_scenic_sc <- scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tfs,
  scenic_params = params_scenic(
    learner_type = "extratrees",
    gene_batch_size = 32L
  ),
  genes_to_take = genes_to_keep_scenic,
  streaming = FALSE,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(et_scenic_sc, "ScenicGrn"),
  info = "scenic grn gpu (SingleCells, extra_trees): correct class returned"
)

expect_true(
  current = checkmate::testMatrix(
    as.matrix(et_scenic_sc),
    mode = "numeric",
    row.names = "named",
    col.names = "named"
  ),
  info = "scenic grn gpu (SingleCells, extra_trees): as.matrix() well formed"
)

importances_et_sc <- check_tf_importances(
  x = as.matrix(et_scenic_sc),
  tf_to_gene_map = tf_to_gene_map
)

expect_true(
  current = all(importances_et_sc),
  info = "scenic grn gpu (SingleCells, extra_trees): planted TFs recovered"
)

## SingleCells - streaming (structural only) -----------------------------------

et_scenic_sc_stream <- scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tfs,
  scenic_params = params_scenic(
    learner_type = "extratrees",
    gene_batch_size = 16L
  ),
  genes_to_take = genes_to_keep_scenic,
  streaming = TRUE,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(et_scenic_sc_stream, "ScenicGrn"),
  info = "scenic grn gpu (SingleCells, streaming): correct class returned"
)

expect_equal(
  current = dim(as.matrix(et_scenic_sc_stream)),
  target = dim(as.matrix(et_scenic_sc)),
  info = "scenic grn gpu (SingleCells, streaming): shape matches non-streaming"
)

## SingleCells - random_forest -------------------------------------------------

rf_scenic_sc <- scenic_grn_sc_gpu(
  object = sc_object,
  tf_ids = tfs,
  scenic_params = params_scenic(
    learner_type = "randomforest",
    gene_batch_size = 32L
  ),
  genes_to_take = genes_to_keep_scenic,
  streaming = FALSE,
  .verbose = FALSE
)

importances_rf_sc <- check_tf_importances(
  x = as.matrix(rf_scenic_sc),
  tf_to_gene_map = tf_to_gene_map
)

expect_true(
  current = all(importances_rf_sc),
  info = "scenic grn gpu (SingleCells, random_forest): planted TFs recovered"
)

## gradient boosting rejection (SingleCells) -----------------------------------

expect_error(
  current = scenic_grn_sc_gpu(
    object = sc_object,
    tf_ids = tfs,
    scenic_params = params_scenic(learner_type = "grnboost2"),
    genes_to_take = genes_to_keep_scenic,
    .verbose = FALSE
  ),
  pattern = "GRNBoost2",
  info = "scenic grn gpu: GBM rejected on SingleCells with clear message"
)

## MetaCells - extra_trees -----------------------------------------------------

et_scenic_mc <- scenic_grn_sc_gpu(
  object = mc_object,
  tf_ids = tfs,
  scenic_params = params_scenic(
    learner_type = "extratrees",
    min_counts = 50L,
    min_cells = 0.02,
    gene_batch_size = 32L
  ),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(et_scenic_mc, "ScenicGrn"),
  info = "scenic grn gpu (MetaCells, extra_trees): correct class returned"
)

expect_true(
  current = checkmate::testMatrix(
    as.matrix(et_scenic_mc),
    mode = "numeric",
    row.names = "named",
    col.names = "named"
  ),
  info = "scenic grn gpu (MetaCells, extra_trees): as.matrix() well formed"
)

## MetaCells - random_forest ---------------------------------------------------

rf_scenic_mc <- scenic_grn_sc_gpu(
  object = mc_object,
  tf_ids = tfs,
  scenic_params = params_scenic(
    learner_type = "randomforest",
    min_counts = 50L,
    min_cells = 0.02,
    gene_batch_size = 32L
  ),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testClass(rf_scenic_mc, "ScenicGrn"),
  info = "scenic grn gpu (MetaCells, random_forest): correct class returned"
)

## gradient boosting rejection (MetaCells) -------------------------------------

expect_error(
  current = scenic_grn_sc_gpu(
    object = mc_object,
    tf_ids = tfs,
    scenic_params = params_scenic(
      learner_type = "grnboost2",
      min_counts = 50L,
      min_cells = 0.02
    ),
    .verbose = FALSE
  ),
  pattern = "GRNBoost2",
  info = "scenic grn gpu: GBM rejected on MetaCells with clear message"
)

## wave_byte_budget validation -------------------------------------------------

expect_error(
  current = scenic_grn_sc_gpu(
    object = sc_object,
    tf_ids = tfs,
    scenic_params = params_scenic(learner_type = "extra_trees"),
    wave_byte_budget = -1,
    genes_to_take = genes_to_keep_scenic,
    .verbose = FALSE
  ),
  info = "scenic grn gpu: negative wave_byte_budget rejected"
)

# clean up ---------------------------------------------------------------------

on.exit(unlink(test_temp_dir, recursive = TRUE, force = TRUE), add = TRUE)
