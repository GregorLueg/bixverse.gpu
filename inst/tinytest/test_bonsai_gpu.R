# gpu bonsai -------------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

# Sanity reduces within a plane; the paravirtualised GPU of the
# macos-15-intel runners advertises plane operations but never runs them
if (!bixverse.gpu:::rs_gpu_plane_ops()) {
  exit_file("GPU adapter does not run plane operations")
}

library(bixverse)

test_temp_dir <- file.path(tempdir(), "bonsai_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## SingleCells object ----------------------------------------------------------

syn_data <- generate_single_cell_test_data(seed = 42L)

sc_object <- load_r_data(
  object = SingleCells(dir_data = test_temp_dir),
  counts = syn_data$counts,
  obs = syn_data$obs,
  var = syn_data$var,
  sc_qc_param = params_sc_min_quality(
    min_unique_genes = 0L,
    min_lib_size = 0L,
    min_cells = 0L
  ),
  .verbose = FALSE
)

cell_names <- get_cell_names(sc_object, filtered = TRUE)

tree_gpu <- bonsai_gpu_sc(sc_object, .verbose = FALSE)
tree_cpu <- bonsai_sc(sc_object, .verbose = FALSE)

## tests -----------------------------------------------------------------------

expect_true(
  inherits(tree_gpu, "BonsaiTree"),
  info = "bonsai gpu: returns the bixverse BonsaiTree"
)

expect_equal(
  tree_gpu$nodes[(is_leaf)]$cell_id,
  cell_names,
  info = "bonsai gpu: one leaf per cell, in the object's cell order"
)

expect_equal(
  sum(is.na(tree_gpu$nodes$parent)),
  1L,
  info = "bonsai gpu: exactly one root"
)

expect_equal(
  tree_gpu$timings$stage,
  c("sanity", "ingest", "bonsai", "layout", "total"),
  info = "bonsai gpu: every stage timed"
)

# the device works in f32, so a gene on the threshold may flip; the bulk of the
# selection has to agree with the CPU run. On this fixture every gene passes on
# both paths (measured 1.0), so this guards gross breakage, not the borderline
gene_jaccard <- length(intersect(tree_gpu$genes_used, tree_cpu$genes_used)) /
  length(union(tree_gpu$genes_used, tree_cpu$genes_used))

expect_true(
  gene_jaccard >= 0.95,
  info = "bonsai gpu: gene selection agrees with the CPU run"
)

expect_true(
  inherits(plot(tree_gpu), "ggplot"),
  info = "bonsai gpu: plots like the CPU tree"
)

## metacells -------------------------------------------------------------------

sc_object <- find_hvg_sc(sc_object, hvg_no = 30L, .verbose = FALSE)
sc_object <- calculate_pca_sc(sc_object, no_pcs = 10L, .verbose = FALSE)
sc_object <- find_neighbours_sc(sc_object, .verbose = FALSE)
mc_object <- generate_bt_meta_cells_sc(
  sc_object,
  sc_meta_cell_params = params_sc_bt_metacells(target_no_metacells = 100L),
  .verbose = FALSE
)

tree_mc_gpu <- bonsai_gpu_sc(mc_object, .verbose = FALSE)
tree_mc_cpu <- bonsai_sc(mc_object, .verbose = FALSE)

expect_equal(
  tree_mc_gpu$nodes[(is_leaf)]$cell_id,
  mc_object[[]]$meta_cell_id,
  info = "bonsai gpu mc: one leaf per metacell"
)

expect_equal(
  sum(is.na(tree_mc_gpu$nodes$parent)),
  1L,
  info = "bonsai gpu mc: exactly one root"
)

expect_equal(
  sort(tree_mc_gpu$genes_used),
  sort(tree_mc_cpu$genes_used),
  info = "bonsai gpu mc: same genes as the CPU run on this fixture"
)

## clean up --------------------------------------------------------------------

unlink(test_temp_dir, recursive = TRUE, force = TRUE)
