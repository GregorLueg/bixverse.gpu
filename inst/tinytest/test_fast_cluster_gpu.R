# gpu fast clustering ----------------------------------------------------------

library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "fast_cluster_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

min_lib_size <- 300L
min_genes_exp <- 45L
min_cells_exp <- 500L
hvg_to_keep <- 30L
no_pcs <- 10L
n_centroids <- 30L
resolutions <- c(1.0, 0.5)

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

sc_cell_type <- as.character(get_sc_obs(sc_object)$cell_grp)
n_cells_kept <- length(get_cells_to_keep(sc_object))

## default run -----------------------------------------------------------------

fc_gpu <- fast_cluster_gpu_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  .verbose = FALSE
)

### structure ------------------------------------------------------------------

expect_true(
  current = inherits(fc_gpu, "SingleCellFastClusters"),
  info = "fast cluster gpu - returns a SingleCellFastClusters object"
)

obs_fc <- get_data(fc_gpu)

expect_true(
  current = checkmate::testDataTable(obs_fc, nrows = n_cells_kept),
  info = "fast cluster gpu - one membership row per QC-passing cell"
)

expect_true(
  current = checkmate::testNames(
    names(obs_fc),
    must.include = c("cell_idx", paste0("res_", resolutions))
  ),
  info = "fast cluster gpu - one membership column per resolution"
)

expect_true(
  current = all(obs_fc$cell_idx == (get_cells_to_keep(sc_object) + 1L)),
  info = "fast cluster gpu - cell_idx is 1-indexed original positions"
)

expect_null(
  current = fc_gpu$stats,
  info = "fast cluster gpu - no grid stats without grid_search"
)

### k-means getters warn without the data --------------------------------------

expect_warning(
  current = get_centroids_sc(fc_gpu),
  info = "fast cluster gpu - warns when no centroids were kept"
)

expect_warning(
  current = get_kmeans_clusters(fc_gpu),
  info = "fast cluster gpu - warns when no k-means clusters were kept"
)

### cluster quality ------------------------------------------------------------

# The synthetic data carries three well-separated cell types. Louvain at the
# lower resolution should recover them rather than shredding them.
ari_truth <- manifoldsR::calc_ari(
  as.integer(factor(sc_cell_type)),
  as.integer(obs_fc[["res_0.5"]])
)

expect_true(
  current = ari_truth > 0.5,
  info = "fast cluster gpu - recovers the synthetic cell types"
)

## with k-means results --------------------------------------------------------

fc_gpu_km <- fast_cluster_gpu_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  return_kmeans = TRUE,
  .verbose = FALSE
)

centroids <- get_centroids_sc(fc_gpu_km)
kmeans_clusters <- get_kmeans_clusters(fc_gpu_km)

expect_true(
  current = checkmate::testMatrix(
    centroids,
    mode = "numeric",
    nrows = n_centroids,
    ncols = no_pcs
  ),
  info = "fast cluster gpu - centroids have the requested shape"
)

expect_true(
  current = checkmate::qtest(kmeans_clusters, sprintf("I%d", n_cells_kept)),
  info = "fast cluster gpu - one k-means assignment per cell"
)

## grid search -----------------------------------------------------------------

fc_gpu_grid <- fast_cluster_gpu_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  grid_search = TRUE,
  no_seeds = 4L,
  .verbose = FALSE
)

grid_stats <- fc_gpu_grid$stats

expect_true(
  current = checkmate::testDataTable(
    grid_stats,
    nrows = length(resolutions)
  ),
  info = "fast cluster gpu grid - one stats row per resolution"
)

expect_true(
  current = checkmate::testNames(
    names(grid_stats),
    permutation.of = c(
      "resolution",
      "mean_ari",
      "median_ari",
      "mean_conductance",
      "median_conductance",
      "mean_n_comms"
    )
  ),
  info = "fast cluster gpu grid - stats carry the expected metrics"
)

expect_equal(
  current = grid_stats$resolution,
  target = resolutions,
  info = "fast cluster gpu grid - stats rows line up with the resolutions"
)

expect_true(
  current = all(grid_stats$mean_ari >= -1 & grid_stats$mean_ari <= 1),
  info = "fast cluster gpu grid - mean ARI is in range"
)

expect_true(
  current = all(grid_stats$mean_conductance >= 0),
  info = "fast cluster gpu grid - conductance is non-negative"
)

expect_true(
  current = all(grid_stats$mean_n_comms >= 1),
  info = "fast cluster gpu grid - at least one community per resolution"
)

expect_true(
  current = checkmate::testDataTable(
    get_data(fc_gpu_grid),
    nrows = n_cells_kept
  ),
  info = "fast cluster gpu grid - memberships still cover every cell"
)

## sNN path --------------------------------------------------------------------

fc_gpu_snn <- fast_cluster_gpu_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  fc_params = params_sc_fast_cluster_gpu(knn = list(k = 10L)),
  snn = TRUE,
  .verbose = FALSE
)

ari_snn <- manifoldsR::calc_ari(
  as.integer(factor(sc_cell_type)),
  as.integer(get_data(fc_gpu_snn)[["res_0.5"]])
)

expect_true(
  current = ari_snn > 0.5,
  info = "fast cluster gpu - sNN path also recovers the cell types"
)

## reproducibility -------------------------------------------------------------

fc_gpu_a <- fast_cluster_gpu_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  seed = 99L,
  .verbose = FALSE
)

fc_gpu_b <- fast_cluster_gpu_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  seed = 99L,
  .verbose = FALSE
)

expect_equal(
  current = get_data(fc_gpu_a),
  target = get_data(fc_gpu_b),
  info = "fast cluster gpu - deterministic with the same seed"
)

## gpu vs cpu ------------------------------------------------------------------

# GPU reduction ordering breaks k-means ties differently, so the centroids and
# hence the memberships are not identical to the CPU ones. What has to hold is
# that both partition the cells the same way.

fc_cpu <- fast_cluster_sc(
  object = sc_object,
  resolutions = resolutions,
  n_centroids = n_centroids,
  .verbose = FALSE
)

obs_cpu <- get_data(fc_cpu)

expect_equal(
  current = nrow(obs_cpu),
  target = nrow(obs_fc),
  info = "fast cluster gpu vs cpu - same number of cells"
)

ari_gpu_cpu <- manifoldsR::calc_ari(
  as.integer(obs_cpu[["res_0.5"]]),
  as.integer(obs_fc[["res_0.5"]])
)

expect_true(
  current = ari_gpu_cpu > 0.5,
  info = "fast cluster gpu vs cpu - comparable partitions"
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

subset_parent_rows <- get_cells_to_keep(subset_object) + 1L

fc_subset <- fast_cluster_gpu_sc(
  object = subset_object,
  resolutions = resolutions,
  n_centroids = 15L,
  .verbose = FALSE
)

expect_true(
  current = inherits(fc_subset, "SingleCellFastClusters"),
  info = "fast cluster gpu subset - returns a SingleCellFastClusters object"
)

expect_true(
  current = all(get_data(fc_subset)$cell_idx %in% subset_parent_rows),
  info = "fast cluster gpu subset - cell_idx stays in parent index space"
)

expect_equal(
  current = nrow(get_data(fc_subset)),
  target = length(subset_parent_rows),
  info = "fast cluster gpu subset - one row per subset cell"
)

## input validation ------------------------------------------------------------

expect_error(
  current = fast_cluster_gpu_sc(
    object = sc_object,
    fc_params = list(k_means_iter = 50L),
    .verbose = FALSE
  ),
  info = "fast cluster gpu - malformed params are rejected"
)

expect_error(
  current = fast_cluster_gpu_sc(
    object = sc_object,
    fc_params = params_sc_fast_cluster_gpu(
      knn = list(ann_dist = "manhattan")
    ),
    .verbose = FALSE
  ),
  info = "fast cluster gpu - manhattan is rejected before it reaches the GPU"
)

expect_error(
  current = fast_cluster_gpu_sc(
    object = sc_object,
    embd_to_use = "not_an_embedding",
    .verbose = FALSE
  ),
  info = "fast cluster gpu - a missing embedding is rejected"
)

expect_error(
  current = fast_cluster_gpu_sc(
    object = sc_object,
    grid_search = TRUE,
    no_seeds = 1L,
    .verbose = FALSE
  ),
  info = "fast cluster gpu grid - fewer than two seeds is rejected"
)
