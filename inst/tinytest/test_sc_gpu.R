# sc gpu workflows -------------------------------------------------------------

library(magrittr)
library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(
  tempdir(),
  "sc_gpu"
)

dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## testing parameters ----------------------------------------------------------

min_lib_size <- 300L
min_genes_exp <- 45L
min_cells_exp <- 450L
hvg_to_keep <- 30L
no_pcs <- 15L
knn_k <- 10L

cell_markers <- list(
  cell_type_1 = list(marker_genes = 0:8L),
  cell_type_2 = list(marker_genes = 9:19L),
  cell_type_3 = list(marker_genes = 20:29L),
  cell_type_4 = list(marker_genes = 30:44L)
)

sc_qc_param <- params_sc_min_quality(
  min_unique_genes = min_genes_exp,
  min_lib_size = min_lib_size,
  min_cells = min_cells_exp
)

## synthetic data --------------------------------------------------------------

# plain (no batches) for PCA / kNN structural tests
sc_data_plain <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1000L,
    marker_genes = cell_markers
  )
)

sc_data_weak <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1200L,
    marker_genes = cell_markers,
    n_batches = 3L,
    batch_effect_strength = "weak"
  )
)

sc_data_med <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1200L,
    marker_genes = cell_markers,
    n_batches = 3L,
    batch_effect_strength = "medium"
  )
)

sc_data_strong <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = 1200L,
    marker_genes = cell_markers,
    n_batches = 3L,
    batch_effect_strength = "strong"
  )
)

## pre-processing --------------------------------------------------------------

dir_data <- list(
  plain = file.path(test_temp_dir, "plain"),
  weak = file.path(test_temp_dir, "weak"),
  medium = file.path(test_temp_dir, "medium"),
  strong = file.path(test_temp_dir, "strong"),
  no_hvg = file.path(test_temp_dir, "no_hvg"),
  no_pca = file.path(test_temp_dir, "no_pca")
)
purrr::walk(dir_data, dir.create, showWarnings = FALSE, recursive = TRUE)

build_sc_object <- function(dir, data) {
  obj <- SingleCells(dir_data = dir)
  obj <- load_r_data(
    object = obj,
    counts = data$counts,
    obs = data$obs,
    var = data$var,
    sc_qc_param = sc_qc_param,
    streaming = 0L,
    .verbose = FALSE
  )
  obj <- find_hvg_sc(obj, hvg_no = hvg_to_keep, .verbose = FALSE)
  obj <- calculate_pca_sc(obj, no_pcs = no_pcs, .verbose = FALSE)
  obj
}

sc_obj_plain <- build_sc_object(dir_data$plain, sc_data_plain)
sc_obj_weak <- build_sc_object(dir_data$weak, sc_data_weak)
sc_obj_med <- build_sc_object(dir_data$medium, sc_data_med)
sc_obj_strong <- build_sc_object(dir_data$strong, sc_data_strong)

# tests ------------------------------------------------------------------------

## calculate_pca_gpu_sc --------------------------------------------------------

### warnings -------------------------------------------------------------------

sc_no_hvg <- SingleCells(dir_data = dir_data$no_hvg)
sc_no_hvg <- load_r_data(
  object = sc_no_hvg,
  counts = sc_data_plain$counts,
  obs = sc_data_plain$obs,
  var = sc_data_plain$var,
  sc_qc_param = sc_qc_param,
  streaming = 0L,
  .verbose = FALSE
)

expect_warning(
  current = calculate_pca_gpu_sc(
    object = sc_no_hvg,
    no_pcs = no_pcs,
    .verbose = FALSE
  ),
  info = "gpu pca - warning when no HVG present"
)

### standard run + comparison vs cpu -------------------------------------------

cpu_pca_factors <- get_pca_factors(sc_obj_plain)

sc_obj_plain <- calculate_pca_gpu_sc(
  object = sc_obj_plain,
  no_pcs = no_pcs,
  .verbose = FALSE
)

gpu_pca_factors <- get_pca_factors(sc_obj_plain)

expect_true(
  current = checkmate::testMatrix(
    gpu_pca_factors,
    mode = "numeric",
    ncols = no_pcs,
    row.names = "named",
    col.names = "named"
  ),
  info = "gpu pca - factor matrix of correct dim and named"
)

expect_true(
  current = checkmate::qtest(
    get_pca_singular_val(sc_obj_plain),
    sprintf("N%i", no_pcs)
  ),
  info = "gpu pca - singular values populated"
)

# fp / reduction-ordering deviations -> use absolute correlations
expect_true(
  current = all(
    abs(diag(cor(cpu_pca_factors[, 1:5], gpu_pca_factors[, 1:5]))) > 0.95
  ),
  info = "gpu pca - top 5 PCs strongly correlated with CPU PCA"
)

### external HVG --------------------------------------------------------------

sc_obj_plain <- calculate_pca_gpu_sc(
  object = sc_obj_plain,
  no_pcs = no_pcs,
  hvg = 1L:hvg_to_keep,
  .verbose = FALSE
)

expect_equal(
  current = get_hvg(sc_obj_plain),
  target = 0L:(hvg_to_keep - 1L),
  info = "gpu pca - external HVG overwrites stored HVG"
)

### CLR / PFlogPF normalisation -----------------------------------------------

sc_obj_plain <- find_hvg_sc(
  sc_obj_plain,
  hvg_no = hvg_to_keep,
  .verbose = FALSE
)

sc_obj_plain <- calculate_pca_gpu_sc(
  object = sc_obj_plain,
  no_pcs = no_pcs,
  pca_params = bixverse::params_sc_pca(
    mean_center = FALSE,
    normalise_variance = FALSE,
    clr = TRUE,
    randomised = TRUE,
    size_factor = 1e3
  ),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_pca_factors(sc_obj_plain),
    mode = "numeric",
    ncols = no_pcs
  ),
  info = "gpu pca - PFlogPF run produces factors of correct dim"
)

### downstream clustering on gpu pca ------------------------------------------

sc_obj_plain <- calculate_pca_gpu_sc(
  object = sc_obj_plain,
  no_pcs = no_pcs,
  .verbose = FALSE
)
sc_obj_plain <- find_neighbours_sc(sc_obj_plain, .verbose = FALSE)
sc_obj_plain <- find_clusters_sc(sc_obj_plain)

expect_true(
  current = all(
    f1_score_confusion_mat(
      unlist(sc_obj_plain[["cell_grp"]]),
      unlist(sc_obj_plain[["leiden_clustering"]])
    ) >
      0.9
  ),
  info = "gpu pca - leiden recovers cell groups"
)

## harmony_v2_gpu_sc -----------------------------------------------------------

### warnings -------------------------------------------------------------------

sc_no_pca <- SingleCells(dir_data = dir_data$no_pca)
sc_no_pca <- load_r_data(
  object = sc_no_pca,
  counts = sc_data_med$counts,
  obs = sc_data_med$obs,
  var = sc_data_med$var,
  sc_qc_param = sc_qc_param,
  streaming = 0L,
  .verbose = FALSE
)

expect_warning(
  current = harmony_v2_gpu_sc(
    object = sc_no_pca,
    batch_column = "batch_index",
    .verbose = FALSE
  ),
  info = "gpu harmony v2 - warning when no PCA in object"
)

### batch correction quality ---------------------------------------------------

assess_harmony_gpu_impact <- function(object) {
  object <- find_neighbours_sc(
    object,
    neighbours_params = params_sc_neighbours(knn = list(k = 10L)),
    .verbose = FALSE
  )

  knn_original <- get_knn_mat(object)
  cell_types <- unlist(object[["cell_grp"]])
  batch_effects <- unlist(object[["batch_index"]])

  neighbours_uncor <- vapply(
    seq_along(cell_types),
    function(i) sum(cell_types[knn_original[i, ] + 1] == cell_types[i]),
    numeric(1)
  )

  kbet_original <- rs_kbet(
    knn_mat = knn_original,
    batch_vector = as.integer(batch_effects),
    verbose = FALSE
  )
  kbet_score_original <- mean(kbet_original$pval <= 0.05)

  object <- harmony_v2_gpu_sc(
    object = object,
    batch_column = "batch_index",
    .verbose = FALSE
  )

  object <- find_neighbours_sc(
    object,
    embd_to_use = "harmony_gpu",
    neighbours_params = params_sc_neighbours(knn = list(k = 10L)),
    .verbose = FALSE
  )

  knn_corrected <- get_knn_mat(object)

  kbet_corrected <- rs_kbet(
    knn_mat = knn_corrected,
    batch_vector = as.integer(batch_effects),
    verbose = FALSE
  )
  kbet_score_corrected <- mean(kbet_corrected$pval <= 0.05)

  neighbours_cor <- vapply(
    seq_along(cell_types),
    function(i) sum(cell_types[knn_corrected[i, ] + 1] == cell_types[i]),
    numeric(1)
  )

  list(
    kbet_original = kbet_score_original,
    kbet_correct = kbet_score_corrected,
    neighbours_original = neighbours_uncor,
    neighbours_corrected = neighbours_cor,
    object = object
  )
}

#### weak ----------------------------------------------------------------------

weak_res <- assess_harmony_gpu_impact(sc_obj_weak)

expect_true(
  current = "harmony_gpu" %in% get_available_embeddings(weak_res$object),
  info = "gpu harmony v2 - harmony_gpu embedding added"
)

expect_true(
  current = weak_res$kbet_original > weak_res$kbet_correct,
  info = "gpu harmony v2 - weak batch effects get regressed out"
)

expect_true(
  current = mean(weak_res$neighbours_corrected) >
    (mean(weak_res$neighbours_original) - 1),
  info = "gpu harmony v2 - biological signal kept (weak)"
)

#### medium --------------------------------------------------------------------

med_res <- assess_harmony_gpu_impact(sc_obj_med)

expect_true(
  current = med_res$kbet_original > med_res$kbet_correct,
  info = "gpu harmony v2 - medium batch effects get regressed out"
)

expect_true(
  current = mean(med_res$neighbours_corrected) >
    (mean(med_res$neighbours_original) - 1),
  info = "gpu harmony v2 - biological signal kept (medium)"
)

#### strong --------------------------------------------------------------------

strong_res <- assess_harmony_gpu_impact(sc_obj_strong)

expect_true(
  current = strong_res$kbet_original > strong_res$kbet_correct,
  info = "gpu harmony v2 - strong batch effects get regressed out"
)

expect_true(
  current = mean(strong_res$neighbours_corrected) >
    (mean(strong_res$neighbours_original) - 1),
  info = "gpu harmony v2 - biological signal kept (strong)"
)

## generate_gpu_knn_sc ---------------------------------------------------------

### warning when embedding missing ---------------------------------------------

expect_warning(
  current = generate_gpu_knn_sc(
    object = sc_obj_plain,
    embd_to_use = "fake_embd"
  ),
  info = "generate_gpu_knn_sc - warning when embedding not found"
)

### exhaustive -----------------------------------------------------------------

knn_ex <- generate_gpu_knn_sc(
  object = sc_obj_plain,
  embd_to_use = "pca",
  gpu_method = "exhaustive",
  k = knn_k,
  dist_metric = "euclidean",
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(knn_ex),
    mode = "integer",
    ncols = knn_k
  ),
  info = "generate_gpu_knn_sc (exhaustive) - kNN matrix of correct dim"
)

### ivf ------------------------------------------------------------------------

knn_ivf <- generate_gpu_knn_sc(
  object = sc_obj_plain,
  embd_to_use = "pca",
  gpu_method = "ivf",
  ivf_params = params_sc_ivf(k = knn_k, nlist = 3L, nprobe = 3L),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(knn_ivf),
    mode = "integer",
    ncols = knn_k
  ),
  info = "generate_gpu_knn_sc (ivf) - kNN matrix of correct dim"
)

### cells_to_use ---------------------------------------------------------------

cells_subset <- get_cell_names(sc_obj_plain, filtered = TRUE)[1:300]

knn_subset <- generate_gpu_knn_sc(
  object = sc_obj_plain,
  embd_to_use = "pca",
  cells_to_use = cells_subset,
  gpu_method = "exhaustive",
  k = knn_k,
  dist_metric = "euclidean",
  .verbose = FALSE
)

expect_equal(
  current = nrow(get_knn_mat(knn_subset)),
  target = length(cells_subset),
  info = "generate_gpu_knn_sc - cells_to_use restricts the search"
)

### no_embd_to_use -------------------------------------------------------------

knn_few_pcs <- generate_gpu_knn_sc(
  object = sc_obj_plain,
  embd_to_use = "pca",
  no_embd_to_use = 5L,
  gpu_method = "exhaustive",
  k = knn_k,
  dist_metric = "euclidean",
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(knn_few_pcs),
    mode = "integer",
    ncols = knn_k
  ),
  info = "generate_gpu_knn_sc - no_embd_to_use truncates dimensions"
)

## generate_cagra_knn_sc -------------------------------------------------------

### warning when embedding missing ---------------------------------------------

expect_warning(
  current = generate_cagra_knn_sc(
    object = sc_obj_plain,
    embd_to_use = "fake_embd"
  ),
  info = "generate_cagra_knn_sc - warning when embedding not found"
)

### extract_knn = TRUE ---------------------------------------------------------

knn_cagra_ext <- generate_cagra_knn_sc(
  object = sc_obj_plain,
  embd_to_use = "pca",
  cagra_params = params_sc_cagra(k = knn_k),
  extract_knn = TRUE,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(knn_cagra_ext),
    mode = "integer",
    ncols = knn_k
  ),
  info = "generate_cagra_knn_sc (extract_knn = TRUE) - kNN matrix of correct dim"
)

### extract_knn = FALSE --------------------------------------------------------

knn_cagra_beam <- generate_cagra_knn_sc(
  object = sc_obj_plain,
  embd_to_use = "pca",
  cagra_params = params_sc_cagra(k = knn_k),
  extract_knn = FALSE,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(knn_cagra_beam),
    mode = "integer",
    ncols = knn_k
  ),
  info = "generate_cagra_knn_sc (extract_knn = FALSE) - kNN matrix of correct dim"
)

## find_neighbours_gpu_sc ------------------------------------------------------

### warning when embedding missing ---------------------------------------------

expect_warning(
  current = find_neighbours_gpu_sc(
    object = sc_obj_plain,
    embd_to_use = "fake_embd",
    .verbose = FALSE
  ),
  info = "find_neighbours_gpu_sc - warning when embedding not found"
)

### exhaustive -----------------------------------------------------------------

obj_ex <- find_neighbours_gpu_sc(
  object = sc_obj_plain,
  gpu_method = "exhaustive",
  k = knn_k,
  dist_metric = "euclidean",
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(obj_ex),
    mode = "integer",
    ncols = knn_k
  ),
  info = "find_neighbours_gpu_sc (exhaustive) - kNN matrix set on object"
)

expect_true(
  current = checkmate::testClass(get_snn_graph(obj_ex), "igraph"),
  info = "find_neighbours_gpu_sc (exhaustive) - sNN graph generated"
)

### ivf ------------------------------------------------------------------------

obj_ivf <- find_neighbours_gpu_sc(
  object = sc_obj_plain,
  gpu_method = "ivf",
  ivf_params = params_sc_ivf(k = knn_k, nlist = 3L, nprobe = 3L),
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(obj_ivf),
    mode = "integer",
    ncols = knn_k
  ),
  info = "find_neighbours_gpu_sc (ivf) - kNN matrix set on object"
)

expect_true(
  current = checkmate::testClass(get_snn_graph(obj_ivf), "igraph"),
  info = "find_neighbours_gpu_sc (ivf) - sNN graph generated"
)

## find_neighbours_cagra_sc ----------------------------------------------------

### warning when embedding missing ---------------------------------------------

expect_warning(
  current = find_neighbours_cagra_sc(
    object = sc_obj_plain,
    embd_to_use = "fake_embd",
    .verbose = FALSE
  ),
  info = "find_neighbours_cagra_sc - warning when embedding not found"
)

### standard run + downstream clustering ---------------------------------------

obj_cagra <- find_neighbours_cagra_sc(
  object = sc_obj_plain,
  cagra_params = params_sc_cagra(k = knn_k),
  extract_knn = FALSE,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(obj_cagra),
    mode = "integer",
    ncols = knn_k
  ),
  info = "find_neighbours_cagra_sc - kNN matrix set on object"
)

expect_true(
  current = checkmate::testClass(get_snn_graph(obj_cagra), "igraph"),
  info = "find_neighbours_cagra_sc - sNN graph generated"
)

obj_cagra <- find_clusters_sc(obj_cagra)

expect_true(
  current = all(
    f1_score_confusion_mat(
      unlist(obj_cagra[["cell_grp"]]),
      unlist(obj_cagra[["leiden_clustering"]])
    ) >
      0.9
  ),
  info = "find_neighbours_cagra_sc - leiden recovers cell groups"
)

# clean up ---------------------------------------------------------------------

on.exit(unlink(test_temp_dir, recursive = TRUE, force = TRUE), add = TRUE)
