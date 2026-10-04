# gpu fastmnn ------------------------------------------------------------------

if (!gpu_available()) {
  exit_file("no GPU adapter available")
}

library(bixverse)

set.seed(42L)

test_temp_dir <- file.path(tempdir(), "fast_mnn_gpu")
dir.create(test_temp_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot("Test directory does not exist" = dir.exists(test_temp_dir))

## fixture params --------------------------------------------------------------

n_cells <- 900L
n_genes <- 100L
n_batches <- 3L
hvg_to_keep <- 50L
# not a multiple of 4, so the GPU indices have to pad internally
no_pcs <- 30L
k <- 10L

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

# parameter wrappers -----------------------------------------------------------

fastmnn_params <- params_sc_fastmnn_gpu(knn = list(k = k))

expect_true(
  current = checkmate::testList(fastmnn_params),
  info = "fastmnn gpu - the parameter wrapper returns a flat list"
)

expect_equal(
  current = fastmnn_params[["knn_method"]],
  target = "exhaustive",
  info = "fastmnn gpu - exhaustive is the default kNN method"
)

expect_equal(
  current = params_sc_fastmnn_gpu()[c("k", "ann_dist")],
  target = list(k = 20L, ann_dist = "cosine"),
  info = "fastmnn gpu - k and ann_dist default to the CPU values"
)

expect_error(
  current = params_sc_fastmnn_gpu(knn = list(extract_knn = TRUE)),
  info = "fastmnn gpu - extract_knn is rejected, the searches are cross-queries"
)

expect_error(
  current = params_sc_fastmnn_gpu(knn = list(not_a_knob = 1L)),
  info = "fastmnn gpu - unknown kNN keys are an error"
)

expect_equal(
  current = params_sc_fastmnn_gpu(knn = list(knn_method = "nndescent"))[[
    "knn_method"
  ]],
  target = "nndescent_gpu",
  info = "fastmnn gpu - nndescent is translated for the Rust parser"
)

expect_error(
  current = assertScFastmnnGpu(params_sc_fastmnn_gpu(knn = list(k = 0L))),
  info = "fastmnn gpu - k = 0 is rejected, fastMNN has no fallback for it"
)

# rust layer -------------------------------------------------------------------

embd <- get_pca_factors(sc_object)
batch_labels <- as.integer(factor(unlist(sc_object[["batch_index"]]))) - 1L

rust_res <- rs_fast_mnn_gpu(
  embd = embd,
  batch_labels = batch_labels,
  fastmnn_params = fastmnn_params,
  seed = 42L,
  verbose = 0L
)

expect_equal(
  current = dim(rust_res),
  target = c(n_cells_kept, no_pcs),
  info = "fastmnn gpu - rust returns a cells x PCs matrix"
)

expect_true(
  current = all(is.finite(rust_res)),
  info = "fastmnn gpu - the corrected embedding is finite"
)

# object layer -----------------------------------------------------------------

sc_object <- fast_mnn_gpu_sc(
  object = sc_object,
  batch_column = "batch_index",
  fastmnn_params = fastmnn_params,
  .verbose = FALSE
)

gpu_embd <- get_embedding(sc_object, "mnn_gpu")

expect_equal(
  current = unname(gpu_embd),
  target = unname(rust_res),
  info = "fastmnn gpu - the object method stores the rust result"
)

# CPU parity: both searches exhaustive, so the corrections must agree
sc_object <- fast_mnn_sc(
  object = sc_object,
  batch_column = "batch_index",
  batch_hvg_genes = get_hvg(sc_object),
  fastmnn_params = params_sc_fastmnn(
    no_pcs = no_pcs,
    knn = list(k = k, knn_method = "exhaustive")
  ),
  use_precomputed_pca = TRUE,
  .verbose = FALSE
)

cpu_embd <- get_embedding(sc_object, "mnn")

expect_equal(
  current = unname(gpu_embd),
  target = unname(cpu_embd),
  tolerance = 1e-3,
  info = "fastmnn gpu - exhaustive GPU matches exhaustive CPU"
)

single_batch <- suppressWarnings(
  fast_mnn_gpu_sc(
    object = sc_object,
    batch_column = "batch_single",
    fastmnn_params = fastmnn_params,
    .verbose = FALSE
  )
)

expect_true(
  current = S7::S7_inherits(single_batch, SingleCells),
  info = "fastmnn gpu - a single batch returns the object as is"
)

# clean up ---------------------------------------------------------------------

on.exit(unlink(test_temp_dir, recursive = TRUE, force = TRUE), add = TRUE)
