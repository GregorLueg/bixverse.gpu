# gpu bbknn --------------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "bbknn_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

n_cells <- 900L
n_genes <- 100L
n_batches <- 3L
hvg_to_keep <- 50L
# not a multiple of 4, so the GPU indices have to pad internally. Regression
# guard: the cross-query dimension check used to compare the raw query dim
# against the padded index dim and rejected everything in between.
no_pcs <- 30L
neighbours_within_batch <- 4L
no_neighbours_to_keep <- 9L

## synthetic single-cell data --------------------------------------------------

sc_data <- generate_single_cell_test_data(
  syn_data_params = params_sc_synthetic_data(
    n_cells = n_cells,
    n_genes = n_genes,
    n_batches = n_batches,
    batch_effect_strength = "medium"
  ),
  seed = 123L
)

# constant column, so the single-batch early return has something to hit
sc_data$obs$batch_single <- "one_batch_only"

## SingleCells object ----------------------------------------------------------

sc_dir <- file.path(test_temp_dir, "sc")
dir.create(sc_dir, showWarnings = FALSE)

sc_object <- SingleCells(dir_data = sc_dir)

sc_object <- load_r_data(
  object = sc_object,
  counts = sc_data$counts,
  obs = sc_data$obs,
  var = sc_data$var,
  sc_qc_param = params_sc_min_quality(
    min_unique_genes = 0L,
    min_lib_size = 0L,
    min_cells = 0L
  ),
  streaming = 0L,
  .verbose = FALSE
)

sc_object <- find_hvg_sc(sc_object, hvg_no = hvg_to_keep, .verbose = FALSE)
sc_object <- calculate_pca_sc(sc_object, no_pcs = no_pcs, .verbose = FALSE)

n_cells_kept <- length(get_cells_to_keep(sc_object))
cell_types <- as.character(get_sc_obs(sc_object)$cell_grp)

# parameter wrappers -----------------------------------------------------------

bbknn_params <- params_sc_bbknn_gpu(
  neighbours_within_batch = neighbours_within_batch
)

expect_true(
  current = checkmate::testList(bbknn_params),
  info = "bbknn gpu - the parameter wrapper returns a flat list"
)

expect_equal(
  current = bbknn_params[["knn_method"]],
  target = "exhaustive",
  info = "bbknn gpu - exhaustive is the default kNN method"
)

# `k` and `extract_knn` do nothing on this path, so they are rejected rather
# than silently ignored
expect_error(
  current = params_sc_bbknn_gpu(knn = list(k = 20L)),
  info = "bbknn gpu - k is rejected, it comes from neighbours_within_batch"
)

expect_error(
  current = params_sc_bbknn_gpu(knn = list(extract_knn = TRUE)),
  info = "bbknn gpu - extract_knn is rejected, the searches are cross-queries"
)

expect_error(
  current = params_sc_bbknn_gpu(knn = list(not_a_knob = 1L)),
  info = "bbknn gpu - unknown kNN keys are an error"
)

expect_equal(
  current = params_sc_bbknn_gpu(knn = list(knn_method = "nndescent"))[[
    "knn_method"
  ]],
  target = "nndescent_gpu",
  info = "bbknn gpu - nndescent is translated for the Rust parser"
)

expect_error(
  current = assertScBbknnGpu(list(neighbours_within_batch = 0L)),
  info = "bbknn gpu - the assertion rejects a malformed parameter list"
)

# rust layer -------------------------------------------------------------------

embd <- get_pca_factors(sc_object)
batch_labels <- as.integer(factor(unlist(sc_object[["batch_index"]]))) - 1L

rust_res <- rs_bbknn_gpu(
  embd = embd,
  batch_labels = batch_labels,
  bbknn_params = bbknn_params,
  seed = 42L,
  verbose = 0L
)

expect_true(
  current = checkmate::testNames(
    names(rust_res),
    permutation.of = c("distances", "connectivities")
  ),
  info = "bbknn gpu - rust returns distances and connectivities"
)

for (slot in c("distances", "connectivities")) {
  csr <- rust_res[[slot]]

  expect_true(
    current = csr$nrow == n_cells_kept && csr$ncol == n_cells_kept,
    info = sprintf("bbknn gpu - %s is a square cell x cell matrix", slot)
  )

  expect_equal(
    current = length(csr$indptr),
    target = n_cells_kept + 1L,
    info = sprintf("bbknn gpu - %s indptr has one entry per cell plus one", slot)
  )
}

# The two are not the same matrix. `distances` is the raw batch-balanced kNN,
# exactly `n_batches * neighbours_within_batch` per cell and stored CSR.
# `connectivities` comes out of the UMAP set operations, which symmetrise, so
# it has more entries and arrives CSC.
expect_equal(
  current = rust_res$distances$cs_type,
  target = "csr",
  info = "bbknn gpu - distances come back as CSR"
)

expect_equal(
  current = rust_res$connectivities$cs_type,
  target = "csc",
  info = "bbknn gpu - connectivities come back as CSC"
)

expect_equal(
  current = length(rust_res$distances$indices),
  target = n_cells_kept * n_batches * neighbours_within_batch,
  info = "bbknn gpu - one distance per cell per batch neighbour"
)

expect_true(
  current = length(rust_res$connectivities$indices) >
    length(rust_res$distances$indices),
  info = "bbknn gpu - symmetrisation adds edges to the connectivities"
)

# The graph is built by reading the connectivities as if they were CSR, which
# is only safe because the set operations leave the matrix symmetric. If that
# ever stops holding, the igraph below silently transposes.
connectivity_mat <- Matrix::sparseMatrix(
  i = rep(
    seq_along(rust_res$connectivities$indptr[-1]),
    diff(rust_res$connectivities$indptr)
  ),
  j = rust_res$connectivities$indices + 1L,
  x = rust_res$connectivities$data,
  dims = c(
    rust_res$connectivities$nrow,
    rust_res$connectivities$ncol
  ),
  index1 = TRUE
)

expect_true(
  current = Matrix::isSymmetric(connectivity_mat),
  info = "bbknn gpu - the connectivity matrix is symmetric"
)

row_of_entry <- rep(
  seq_len(n_cells_kept),
  diff(rust_res$distances$indptr)
)

expect_false(
  current = any(row_of_entry == (rust_res$distances$indices + 1L)),
  info = "bbknn gpu - no cell is its own neighbour"
)

expect_true(
  current = all(rust_res$distances$data >= 0),
  info = "bbknn gpu - distances are non-negative"
)

## cpu parity ------------------------------------------------------------------

# Exhaustive is exact on both sides and both recompute the distances against
# the same global embedding, so the graphs have to agree structurally rather
# than merely correlate.
cpu_res <- rs_bbknn(
  embd = embd,
  batch_labels = batch_labels,
  bbknn_params = params_sc_bbknn(
    neighbours_within_batch = neighbours_within_batch,
    knn = list(knn_method = "exhaustive", ann_dist = "euclidean")
  ),
  seed = 42L,
  verbose = 0L
)

expect_equal(
  current = rust_res$distances$indptr,
  target = cpu_res$distances$indptr,
  info = "bbknn gpu - matches the CPU indptr with an exhaustive search"
)

expect_equal(
  current = rust_res$distances$indices,
  target = cpu_res$distances$indices,
  info = "bbknn gpu - matches the CPU indices with an exhaustive search"
)

expect_equal(
  current = rust_res$distances$data,
  target = cpu_res$distances$data,
  tolerance = 1e-5, # fp32 on both paths, different reduction order
  info = "bbknn gpu - matches the CPU distances with an exhaustive search"
)

expect_equal(
  current = rust_res$connectivities$indptr,
  target = cpu_res$connectivities$indptr,
  info = "bbknn gpu - matches the CPU connectivity structure"
)

expect_equal(
  current = rust_res$connectivities$data,
  target = cpu_res$connectivities$data,
  tolerance = 1e-5,
  info = "bbknn gpu - matches the CPU connectivity weights"
)

## approximate backends --------------------------------------------------------

for (method in c("ivf", "nndescent")) {
  approx_res <- rs_bbknn_gpu(
    embd = embd,
    batch_labels = batch_labels,
    bbknn_params = params_sc_bbknn_gpu(
      neighbours_within_batch = neighbours_within_batch,
      knn = list(knn_method = method, n_list = 8L, n_probe = 8L)
    ),
    seed = 42L,
    verbose = 0L
  )

  recall <- mean(
    approx_res$distances$indices %in% rust_res$distances$indices
  )

  expect_true(
    current = recall > 0.5,
    info = sprintf(
      "bbknn gpu - the %s backend recovers most exhaustive neighbours",
      method
    )
  )
}

# s7 method --------------------------------------------------------------------

kbet_before <- calculate_kbet_sc(
  object = find_neighbours_sc(sc_object, .verbose = FALSE),
  batch_column = "batch_index",
  .verbose = FALSE
)

bbknn_object <- bbknn_gpu_sc(
  object = sc_object,
  batch_column = "batch_index",
  no_neighbours_to_keep = no_neighbours_to_keep,
  bbknn_params = bbknn_params,
  .verbose = FALSE
)

expect_true(
  current = checkmate::testMatrix(
    get_knn_mat(bbknn_object),
    mode = "integer",
    nrows = n_cells_kept,
    ncols = no_neighbours_to_keep
  ),
  info = "bbknn gpu - the kNN matrix is trimmed to no_neighbours_to_keep"
)

expect_true(
  current = checkmate::testClass(get_snn_graph(bbknn_object), "igraph"),
  info = "bbknn gpu - the connectivities become an igraph"
)

expect_true(
  current = igraph::is_weighted(get_snn_graph(bbknn_object)),
  info = "bbknn gpu - the graph carries the connectivities as weights"
)

kbet_after <- calculate_kbet_sc(
  object = bbknn_object,
  batch_column = "batch_index",
  .verbose = FALSE
)

expect_true(
  current = kbet_after$kbet_score < kbet_before$kbet_score,
  info = "bbknn gpu - batch effects get regressed out"
)

# the point of BBKNN is mixing batches without shredding biology, so check the
# neighbourhoods still hold the same cell type
knn_mat <- get_knn_mat(bbknn_object)
same_type <- vapply(
  seq_len(nrow(knn_mat)),
  \(i) mean(cell_types[knn_mat[i, ] + 1L] == cell_types[i]),
  numeric(1)
)

expect_true(
  current = mean(same_type) > 0.5,
  info = "bbknn gpu - biological signal is not regressed out"
)

## warnings and early returns --------------------------------------------------

expect_warning(
  current = bbknn_gpu_sc(
    object = bbknn_object,
    batch_column = "batch_index",
    no_neighbours_to_keep = no_neighbours_to_keep,
    bbknn_params = bbknn_params,
    .verbose = FALSE
  ),
  info = "bbknn gpu - warns when an existing kNN gets overwritten"
)

expect_warning(
  current = bbknn_gpu_sc(
    object = sc_object,
    batch_column = "batch_index",
    # more than n_batches * neighbours_within_batch
    no_neighbours_to_keep = (n_batches * neighbours_within_batch) + 5L,
    bbknn_params = bbknn_params,
    .verbose = FALSE
  ),
  info = "bbknn gpu - warns when too few neighbours can be generated"
)

# Rust errors on a single batch, the R method returns the object untouched so
# that it behaves like the CPU one
single_batch <- suppressWarnings(
  bbknn_gpu_sc(
    object = sc_object,
    batch_column = "batch_single",
    bbknn_params = bbknn_params,
    .verbose = FALSE
  )
)

expect_true(
  current = S7::S7_inherits(single_batch, SingleCells),
  info = "bbknn gpu - a single batch returns the object as is"
)

# clean up ---------------------------------------------------------------------

on.exit(unlink(test_temp_dir, recursive = TRUE, force = TRUE), add = TRUE)
