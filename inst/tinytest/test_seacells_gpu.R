# gpu seacells -----------------------------------------------------------------

library(magrittr)
library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "seacells_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

min_lib_size <- 300L
min_genes_exp <- 45L
min_cells_exp <- 500L
hvg_to_keep <- 30L
no_pcs <- 10L
n_sea_cells <- 50L

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

sc_cell_type <- as.character(get_sc_obs(sc_object)$cell_grp)

## default run -----------------------------------------------------------------

seacell_params <- params_sc_seacells(
  n_sea_cells = n_sea_cells,
  min_iter = 5L,
  knn = list(k = 10L)
)

gpu_mc <- generate_seacells_gpu_sc(
  sc_object,
  seacell_params = seacell_params,
  .verbose = FALSE
)

### structure ------------------------------------------------------------------

expect_true(
  current = S7::S7_inherits(gpu_mc, MetaCells),
  info = "seacells gpu - returns a MetaCells object"
)

expect_true(
  current = checkmate::testDataTable(gpu_mc[[]], nrows = n_sea_cells),
  info = "seacells gpu - obs table has one row per requested meta cell"
)

expect_true(
  current = checkmate::testNames(
    names(gpu_mc[[]]),
    must.include = c(
      "meta_cell_idx",
      "meta_cell_id",
      "no_originating_cells",
      "original_cell_idx"
    )
  ),
  info = "seacells gpu - obs table has the expected columns"
)

expect_true(
  current = checkmate::testDataTable(get_sc_var(gpu_mc)),
  info = "seacells gpu - var table survives"
)

expect_true(
  current = checkmate::testClass(gpu_mc[], "dgRMatrix"),
  info = "seacells gpu - raw counts come back as a dgRMatrix"
)

expect_equal(
  current = nrow(gpu_mc[]),
  target = n_sea_cells,
  info = "seacells gpu - one count row per meta cell"
)

expect_true(
  current = all(
    unlist(gpu_mc[[]]$original_cell_idx) %in%
      (get_cells_to_keep(sc_object) + 1L)
  ),
  info = "seacells gpu - members stay inside the QC-passing cells"
)

### archetypes -----------------------------------------------------------------

# `archetypes` is 0-indexed, unlike `original_cell_idx`. That is what the CPU
# path returns, so parity beats consistency here.
archetypes <- gpu_mc@other_data$archetypes

expect_true(
  current = checkmate::testIntegerish(archetypes, len = n_sea_cells),
  info = "seacells gpu - one archetype per meta cell"
)

expect_true(
  current = all(archetypes %in% get_cells_to_keep(sc_object)),
  info = "seacells gpu - archetype indices live in the original cell space"
)

expect_true(
  current = checkmate::testNumeric(gpu_mc@other_data$rss, min.len = 2L),
  info = "seacells gpu - RSS history is returned"
)

expect_true(
  current = gpu_mc@other_data$rss[length(gpu_mc@other_data$rss)] <
    gpu_mc@other_data$rss[1],
  info = "seacells gpu - RSS decreases over the fit"
)

### purity ---------------------------------------------------------------------

gpu_mc <- calc_meta_cell_purity(
  gpu_mc,
  original_cell_type = sc_cell_type
)

expect_true(
  current = mean(gpu_mc[[]]$mc_purity) > 0.9,
  info = "seacells gpu - similar cell types are being pulled together"
)

## gpu vs cpu ------------------------------------------------------------------

# GPU reduction ordering shifts ties, so the assignments are not bit-identical
# to the CPU ones. What has to hold is that both land on the same structure.

cpu_mc <- generate_seacells_sc(
  sc_object,
  seacell_params = seacell_params,
  .verbose = FALSE
)

cpu_mc <- calc_meta_cell_purity(
  cpu_mc,
  original_cell_type = sc_cell_type
)

expect_equal(
  current = nrow(gpu_mc[[]]),
  target = nrow(cpu_mc[[]]),
  info = "seacells gpu vs cpu - same number of meta cells"
)

expect_equal(
  current = dim(gpu_mc[]),
  target = dim(cpu_mc[]),
  info = "seacells gpu vs cpu - same count matrix dimensions"
)

expect_true(
  current = abs(mean(gpu_mc[[]]$mc_purity) - mean(cpu_mc[[]]$mc_purity)) < 0.05,
  info = "seacells gpu vs cpu - comparable mean meta cell purity"
)

gpu_rss <- gpu_mc@other_data$rss
cpu_rss <- cpu_mc@other_data$rss

expect_true(
  current = abs(gpu_rss[length(gpu_rss)] - cpu_rss[length(cpu_rss)]) /
    cpu_rss[length(cpu_rss)] <
    0.05,
  info = "seacells gpu vs cpu - final RSS within 5%"
)

expect_equal(
  current = gpu_mc@original_assignment$n_cells,
  target = cpu_mc@original_assignment$n_cells,
  info = "seacells gpu vs cpu - assignments span the same cell space"
)

expect_equal(
  current = length(gpu_mc@other_data$archetypes),
  target = length(cpu_mc@other_data$archetypes),
  info = "seacells gpu vs cpu - same number of archetypes"
)

expect_true(
  current = all(cpu_mc@other_data$archetypes %in% get_cells_to_keep(sc_object)),
  info = "seacells gpu vs cpu - archetypes share the CPU index space"
)

## regenerated kNN -------------------------------------------------------------

gpu_mc_regen <- generate_seacells_gpu_sc(
  sc_object,
  seacell_params = seacell_params,
  regenerate_knn = TRUE,
  .verbose = FALSE
) %>%
  calc_meta_cell_purity(original_cell_type = sc_cell_type)

expect_true(
  current = mean(gpu_mc_regen[[]]$mc_purity) > 0.9,
  info = "seacells gpu - regenerated kNN gives comparable purity"
)

## cells_to_use ----------------------------------------------------------------

cells_to_use <- sc_object[[]][cell_grp != "cell_type_3", cell_id]

gpu_mc_small <- generate_seacells_gpu_sc(
  sc_object,
  seacell_params = params_sc_seacells(
    n_sea_cells = n_sea_cells,
    min_iter = 5L,
    convergence_epsilon = 0.001,
    knn = list(k = 10L)
  ),
  regenerate_knn = TRUE,
  cells_to_use = cells_to_use,
  .verbose = FALSE
) %>%
  calc_meta_cell_purity(original_cell_type = sc_cell_type)

right_cell_types <- purrr::map_lgl(
  gpu_mc_small[[]]$original_cell_idx,
  \(idx) all(sc_cell_type[idx] != "cell_type_3")
)

expect_true(
  current = all(right_cell_types),
  info = "seacells gpu - no unexpected cell types in the cells_to_use version"
)

expect_true(
  current = mean(gpu_mc_small[[]]$mc_purity) > 0.9,
  info = "seacells gpu - cells_to_use version keeps meta cells pure"
)

## non-contiguous cells to keep ------------------------------------------------

# The runs above load with QC parameters, which drops failing cells at write
# time and leaves cells_to_keep as 0..n. That hides any confusion between the
# embedding row space and the count file row space. Here the data is loaded
# unfiltered and narrowed afterwards, so the two spaces genuinely differ.

gap_temp_dir <- file.path(test_temp_dir, "gaps")
dir.create(gap_temp_dir, recursive = TRUE, showWarnings = FALSE)

gap_object <- SingleCells(dir_data = gap_temp_dir)

gap_object <- load_r_data(
  object = gap_object,
  counts = single_cell_test_data$counts,
  obs = single_cell_test_data$obs,
  var = single_cell_test_data$var,
  sc_qc_param = params_sc_min_quality(
    min_unique_genes = 1L,
    min_lib_size = 1L,
    min_cells = 1L
  ),
  streaming = 0L,
  .verbose = FALSE
)

gap_all_ids <- get_sc_obs(gap_object)$cell_id
gap_n_total <- length(gap_all_ids)

# every other cell, so no working row lines up with its count file row
gap_object <- set_cells_to_keep(
  gap_object,
  gap_all_ids[seq(1L, gap_n_total, by = 2L)]
)

gap_object <- find_hvg_sc(gap_object, hvg_no = hvg_to_keep, .verbose = FALSE)
gap_object <- calculate_pca_sc(gap_object, no_pcs = no_pcs, .verbose = FALSE)
gap_object <- find_neighbours_sc(
  gap_object,
  neighbours_params = params_sc_neighbours(knn = list(k = 15L)),
  .verbose = FALSE
)

gap_keep_1idx <- get_cells_to_keep(gap_object) + 1L

expect_false(
  current = identical(gap_keep_1idx, seq_along(gap_keep_1idx)),
  info = "seacells gpu gap fixture - cells_to_keep is genuinely non-contiguous"
)

# maps a count file row onto the row of the source matrix it was written from
gap_file_to_src <- match(
  get_sc_obs(gap_object)$cell_id,
  rownames(single_cell_test_data$counts)
)

gap_mc <- generate_seacells_gpu_sc(
  gap_object,
  seacell_params = params_sc_seacells(n_sea_cells = 10L, min_iter = 5L),
  .verbose = FALSE
)

gap_members <- gap_mc[[]]$original_cell_idx
gap_raw <- get_sc_counts(gap_mc, assay = "raw")
gap_gene_match <- match(
  colnames(gap_raw),
  colnames(single_cell_test_data$counts)
)

# Aggregated counts must equal the sum of the member cells' raw counts. This is
# the check that actually catches a wrong index space, since misindexed members
# still look plausible on their own.
gap_agree <- purrr::map_lgl(
  seq_len(min(5L, nrow(gap_raw))),
  \(i) {
    manual <- Matrix::colSums(
      single_cell_test_data$counts[
        gap_file_to_src[gap_members[[i]]],
        ,
        drop = FALSE
      ]
    )
    isTRUE(all.equal(
      as.numeric(manual[gap_gene_match]),
      as.numeric(gap_raw[i, ])
    ))
  }
)

expect_true(
  current = all(unlist(gap_members) %in% gap_keep_1idx),
  info = "seacells gpu non-contiguous - members stay inside cells_to_keep"
)

expect_true(
  current = all(gap_agree),
  info = "seacells gpu non-contiguous - aggregated counts match the members"
)

expect_equal(
  current = gap_mc@original_assignment$n_cells,
  target = gap_n_total,
  info = "seacells gpu non-contiguous - assignments span the full count file"
)

expect_true(
  current = all(
    gap_mc@other_data$archetypes %in% get_cells_to_keep(gap_object)
  ),
  info = "seacells gpu non-contiguous - archetypes are remapped to file space"
)

## SingleCellsSubset dispatch --------------------------------------------------

subset_object <- SingleCellsSubset(
  sc_object = sc_object,
  grouping_column = "cell_grp",
  group = "cell_type_1"
)

subset_object <- find_hvg_sc(
  subset_object,
  hvg_no = hvg_to_keep,
  .verbose = FALSE
)
subset_object <- calculate_pca_sc(
  subset_object,
  no_pcs = no_pcs,
  .verbose = FALSE
)
subset_object <- find_neighbours_sc(
  subset_object,
  neighbours_params = params_sc_neighbours(knn = list(k = 10L)),
  .verbose = FALSE
)

subset_parent_rows <- get_cells_to_keep(subset_object) + 1L

subset_mc <- generate_seacells_gpu_sc(
  subset_object,
  seacell_params = params_sc_seacells(n_sea_cells = 8L, min_iter = 5L),
  .verbose = FALSE
)

subset_members <- unlist(subset_mc[[]]$original_cell_idx)

expect_true(
  current = all(subset_members %in% subset_parent_rows),
  info = "seacells gpu subset - members stay inside the subset"
)

expect_true(
  current = all(sc_cell_type[subset_members] == "cell_type_1"),
  info = "seacells gpu subset - members carry the subset group label"
)

expect_true(
  current = checkmate::testDataTable(get_sc_var(subset_mc)),
  info = "seacells gpu subset - var table survives the subset"
)

## input validation ------------------------------------------------------------

expect_error(
  current = generate_seacells_gpu_sc(
    sc_object,
    seacell_params = list(n_sea_cells = 10L),
    .verbose = FALSE
  ),
  info = "seacells gpu - malformed params are rejected"
)

expect_error(
  current = generate_seacells_gpu_sc(
    sc_object,
    seacell_params = seacell_params,
    cells_to_use = "not_a_cell_id",
    regenerate_knn = TRUE,
    .verbose = FALSE
  ),
  info = "seacells gpu - unknown cell identifiers are rejected"
)
