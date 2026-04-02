# tests of parametric umap -----------------------------------------------------

## synthetic data --------------------------------------------------------------

set.seed(42L)

n_per_cluster <- 100L
n_dim <- 16L
n_clusters <- 3L

# Generate 3 well-separated clusters
centres <- matrix(
  c(rep(5, n_dim), rep(-5, n_dim), rep(0, n_dim)),
  nrow = n_clusters,
  byrow = TRUE
)

cluster_labels <- rep(seq_len(n_clusters), each = n_per_cluster)
data <- centres[cluster_labels, ] +
  matrix(
    rnorm(n_per_cluster * n_clusters * n_dim, sd = 0.5),
    ncol = n_dim
  )

# train/test split
train_idx <- sort(sample(seq_len(nrow(data)), size = floor(0.7 * nrow(data))))
train_data <- data[train_idx, ]
train_labels <- cluster_labels[train_idx]
test_data <- data[-train_idx, ]
test_labels <- cluster_labels[-train_idx]

## helpers ---------------------------------------------------------------------

# Compute purity: for each cluster label, find the most common embedding
# cluster assignment and return the weighted average match rate.
calc_purity <- function(true_labels, embedding, k = 10L) {
  n <- nrow(embedding)
  # brute force kNN in embedding space
  dist_mat <- as.matrix(dist(embedding))
  diag(dist_mat) <- Inf
  pred_labels <- integer(n)
  for (i in seq_len(n)) {
    nn_idx <- order(dist_mat[i, ])[seq_len(k)]
    nn_labels <- true_labels[nn_idx]
    pred_labels[i] <- as.integer(names(which.max(table(nn_labels))))
  }
  mean(pred_labels == true_labels)
}

## fit model -------------------------------------------------------------------

pumap <- parametric_umap(
  data = train_data,
  n_dim = 2L,
  k = 15L,
  min_dist = 0.5,
  spread = 1.0,
  knn_method = "hnsw",
  parametric_umap_params = params_parametric_umap(
    batch_size = 64L,
    n_epochs = 50L
  ),
  use_gpu = FALSE,
  seed = 42L,
  .verbose = FALSE
)

## tests -----------------------------------------------------------------------

### training embedding shape ---------------------------------------------------

expect_equal(
  current = nrow(pumap$embedding),
  target = nrow(train_data),
  info = "parametric umap - embedding nrow matches training data"
)

expect_equal(
  current = ncol(pumap$embedding),
  target = 2L,
  info = "parametric umap - embedding has correct number of dimensions"
)

### training embedding has no NA/Inf -------------------------------------------

expect_true(
  current = all(is.finite(pumap$embedding)),
  info = "parametric umap - training embedding is finite"
)

### training embedding separates clusters --------------------------------------

train_purity <- calc_purity(train_labels, pumap$embedding, k = 10L)

expect_true(
  current = train_purity >= 0.90,
  info = sprintf(
    "parametric umap - training cluster purity %.2f >= 0.90",
    train_purity
  )
)

### predict on held-out data ---------------------------------------------------

test_embedding <- predict(pumap, newdata = test_data)

expect_equal(
  current = nrow(test_embedding),
  target = nrow(test_data),
  info = "parametric umap predict - nrow matches test data"
)

expect_equal(
  current = ncol(test_embedding),
  target = 2L,
  info = "parametric umap predict - correct number of dimensions"
)

expect_true(
  current = all(is.finite(test_embedding)),
  info = "parametric umap predict - test embedding is finite"
)

### held-out embedding separates clusters --------------------------------------

test_purity <- calc_purity(test_labels, test_embedding, k = 10L)

expect_true(
  current = test_purity >= 0.85,
  info = sprintf(
    "parametric umap predict - test cluster purity %.2f >= 0.85",
    test_purity
  )
)

### test points land near corresponding training clusters ----------------------

# For each test point, find its nearest training point in embedding space and
# check the label matches.

combined_embedding <- rbind(pumap$embedding, test_embedding)
n_train <- nrow(pumap$embedding)
n_test <- nrow(test_embedding)

cross_dist <- as.matrix(dist(combined_embedding))
cross_dist <- cross_dist[(n_train + 1):(n_train + n_test), seq_len(n_train)]

nn_train_idx <- apply(cross_dist, 1, which.min)
nn_train_labels <- train_labels[nn_train_idx]
cross_match_rate <- mean(nn_train_labels == test_labels)

expect_true(
  current = cross_match_rate >= 0.85,
  info = sprintf(
    "parametric umap predict - test-train cross match %.2f >= 0.85",
    cross_match_rate
  )
)

### predict is deterministic ---------------------------------------------------

test_embedding_2 <- predict(pumap, newdata = test_data)

expect_equal(
  current = test_embedding_2,
  target = test_embedding,
  info = "parametric umap predict - deterministic forward pass"
)
