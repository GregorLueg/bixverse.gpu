# Package index

## Single cell GPU

Different GPU-accelerated methods specifically for the GPU.

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
  : Find CAGRA GPU-accelerated neighbours for single cells
- [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md)
  : Default parameters for IVF-GPU kNN search
- [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md)
  : Default parameters for CAGRA-style kNN search
- [`params_sc_harmony_v2_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_harmony_v2_gpu.md)
  : Default parameters for Harmony v2 GPU batch correction

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
clustering has been implemented and likely more to come.

- [`k_means_cluster_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/k_means_cluster_gpu.md)
  : GPU-accelerated k-means clustering
- [`get_centroids()`](https://gregorlueg.github.io/bixverse.gpu/reference/get_centroids.md)
  : Get cluster centroids
- [`membership()`](https://gregorlueg.github.io/bixverse.gpu/reference/membership.md)
  : Get cluster assignments
- [`params_kmeans_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_kmeans_gpu.md)
  : Default parameters for GPU k-means

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
- [`rs_harmony_v2_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_harmony_v2_gpu.md)
  **\[experimental\]** : Harmony batch correction in Rust (version 2,
  GPU-accelerated)
- [`rs_ivf_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_ivf_gpu_knn.md)
  **\[experimental\]** : Generate an IVF-GPU-accelerated kNN graph
- [`rs_kmeans_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_kmeans_gpu.md)
  **\[experimental\]** : GPU-accelerated k-means
- [`rs_parametric_umap_predict()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_parametric_umap_predict.md)
  **\[experimental\]** : Predict new data using a trained parametric
  UMAP model
- [`rs_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_parametric_umap.md)
  **\[experimental\]** : Parametric UMAP implementation
