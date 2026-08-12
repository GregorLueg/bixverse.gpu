# Package index

## Single cell GPU

Different GPU-accelerated methods specifically for the single cell
applications.

- [`calculate_pca_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/calculate_pca_gpu_sc.md)
  : GPU-accelerated PCA for single cell
- [`generate_cagra_knn_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_cagra_knn_sc.md)
  : Generate CAGRA GPU kNN data for single cells
- [`generate_gpu_knn_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_gpu_knn_sc.md)
  : Generate GPU kNN data for single cells (exhaustive / IVF)
- [`harmony_v2_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/harmony_v2_gpu_sc.md)
  : Run Harmony v2 (GPU)
- [`find_neighbours_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_gpu_sc.md)
  : Find GPU-accelerated neighbours for single cells (exhaustive / IVF)
- [`find_neighbours_cagra_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_cagra_sc.md)
  : Find neighbours via CAGRA GPU-acceleration for single cells
- [`generate_seacells_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_seacells_gpu_sc.md)
  : Generate meta cells based on SEACells on the GPU
- [`scenic_grn_sc_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/scenic_grn_sc_gpu.md)
  : GPU-accelerated SCENIC GRN inference
- [`fast_cluster_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/fast_cluster_gpu_sc.md)
  : Run fast Louvain clustering on a SingleCells object (GPU)
- [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md)
  : Default parameters for IVF-GPU kNN search
- [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md)
  : Default parameters for CAGRA-style kNN search
- [`params_sc_harmony_v2_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_harmony_v2_gpu.md)
  : Default parameters for Harmony v2 GPU batch correction
- [`params_sc_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fast_cluster_gpu.md)
  : Default parameters for GPU fast Louvain clustering

## GPU-accelerated UMAP

GPU-accelerated UMAP with pluggable kNN backends (IVF, NN-descent/CAGRA,
exhaustive) and CPU/GPU optimisers. Also a version for the single cells
analysis suite in bixverse.

- [`umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu.md)
  : Rust-based UMAP (GPU)
- [`umap_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/umap_gpu_sc.md)
  : Run UMAP on a SingleCells object (GPU)
- [`params_umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_umap_gpu.md)
  : Wrapper function to generate UMAP parameters (GPU version)
- [`params_nn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_nn_gpu.md)
  : Wrapper function to generate GPU nearest neighbour parameters

## GPU-accelerated t-SNE

t-SNE with GPU-accelerated kNN backends (IVF, NN-descent, exhaustive).
Optimiser (BH or FFT) runs on CPU via the Rust implementation in
manifoldsR. Also a version for the single cells analysis suite in
bixverse.

- [`tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu.md)
  : Rust-based t-SNE (GPU)
- [`tsne_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/tsne_gpu_sc.md)
  : Run t-SNE on a SingleCells object (GPU)
- [`params_tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_tsne_gpu.md)
  : Wrapper function to generate t-SNE parameters (GPU version)

## Parametric UMAP

Want to train a neural net to do your UMAP embeddings? Want to train
flexibly on CPU or GPU? Here’s what you need.

- [`parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/parametric_umap.md)
  : Parametric UMAP
- [`params_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_parametric_umap.md)
  : Wrapper function to generate parametric UMAP parameters
- [`predict(`*`<ParametricUmapModel>`*`)`](https://gregorlueg.github.io/bixverse.gpu/reference/predict.ParametricUmapModel.md)
  : Predict embeddings for new data using a trained parametric UMAP
  model
- [`load_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/load_parametric_umap.md)
  : Load a parametric UMAP as a qs2 file
- [`save_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/save_parametric_umap.md)
  : Save a parametric UMAP as a qs2 file

## Other GPU-accelerated methods

Other GPU-accelerated methods that you might find useful… k-means
clustering and a direct interface into the GPU-accelerated kNN searches
can be found here.

- [`k_means_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/k_means_cluster_gpu.md)
  : GPU-accelerated k-means clustering
- [`params_kmeans_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_kmeans_gpu.md)
  : Default parameters for GPU k-means
- [`generate_knn_graph_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/generate_knn_graph_gpu.md)
  : Generate a k-nearest neighbour graph (GPU-accelerated)

## Rust wrappers

Everything rusty - only use this if you know what you are doing… Maybe
useful for your own package? Use with care and read the documentation!

- [`rs_cagra_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cagra_gpu_knn.md)
  **\[experimental\]** : Generate a CAGRA-style GPU-accelerated kNN
  graph
- [`rs_cor_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cor_gpu.md)
  **\[experimental\]** : GPU-accelerated correlation calculations
- [`rs_cov_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cov_gpu.md)
  **\[experimental\]** : GPU-accelerated covariance calculations
- [`rs_exhaustive_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_exhaustive_gpu_knn.md)
  **\[experimental\]** : Generate an GPU-accelerated kNN graph from an
  exhaustive search
- [`rs_fast_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_fast_cluster_gpu.md)
  **\[experimental\]** : GPU: fast Louvain clustering on the data
- [`rs_fast_cluster_grid_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_fast_cluster_grid_gpu.md)
  **\[experimental\]** : GPU: fast Louvain clustering on the data (with
  multiple seeds)
- [`rs_harmony_v2_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_harmony_v2_gpu.md)
  **\[experimental\]** : Harmony batch correction in Rust (version 2,
  GPU-accelerated)
- [`rs_ivf_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_ivf_gpu_knn.md)
  **\[experimental\]** : Generate an IVF-GPU-accelerated kNN graph
- [`rs_kmeans_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_kmeans_gpu.md)
  **\[experimental\]** : GPU-accelerated k-means
- [`rs_mc_scenic_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_mc_scenic_gpu.md)
  **\[experimental\]** : GPU: SCENIC GRN inference on MetaCells
  (in-memory sparse)
- [`rs_parametric_umap_predict()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_parametric_umap_predict.md)
  **\[experimental\]** : Predict new data using a trained parametric
  UMAP model
- [`rs_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_parametric_umap.md)
  **\[experimental\]** : Parametric UMAP implementation
- [`rs_sc_pca_sparse_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_sc_pca_sparse_gpu.md)
  **\[experimental\]** : Calculates sparse PCA for single cell
- [`rs_scenic_grn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_scenic_grn_gpu.md)
  **\[experimental\]** : GPU: SCENIC gene-regulatory network inference
  (disk-backed)
- [`rs_scenic_grn_streaming_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_scenic_grn_streaming_gpu.md)
  **\[experimental\]** : GPU: SCENIC GRN inference (streaming; bounded
  host memory)
- [`rs_seacells_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_seacells_gpu.md)
  **\[experimental\]** : GPU: SEACells meta cell generation
- [`rs_umap_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_umap_gpu.md)
  **\[experimental\]** : UMAP implementation
- [`rs_umap_from_knn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_umap_from_knn_gpu.md)
  **\[experimental\]** : UMAP implementation from a pre-computed kNN
  graph
- [`rs_tsne_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_tsne_gpu.md)
  **\[experimental\]** : tSNE implementation
- [`rs_tsne_from_knn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_tsne_from_knn_gpu.md)
  **\[experimental\]** : tSNE implementation from a pre-computed kNN
  graph
