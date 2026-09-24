# bixverse.gpu API index

Every documented entry point, grouped the way the package website groups them. Generated from `_pkgdown.yml` and `man/*.Rd` by `data-raw/generate_api_index.R`. Do not edit by hand.

Grep this file to check whether a function exists before calling it.

## Single cell GPU

Different GPU-accelerated methods specifically for the single cell applications.

- `calculate_pca_gpu_sc`: GPU-accelerated PCA for single cell
- `generate_gpu_knn_sc`: Generate GPU kNN data for single cells
- `harmony_v2_gpu_sc`: Run Harmony v2 (GPU)
- `find_neighbours_gpu_sc`: Find GPU-accelerated neighbours for single cells
- `generate_seacells_gpu_sc`: Generate meta cells based on SEACells on the GPU
- `scenic_grn_sc_gpu`: GPU-accelerated SCENIC GRN inference
- `scrublet_gpu_sc`: Doublet detection with Scrublet on the GPU
- `fast_cluster_gpu_sc`: Run fast Louvain clustering on a SingleCells object (GPU)
- `bbknn_gpu_sc`: Run BBKNN on the GPU
- `nebula_gpu_sc`: Run NEBULA on single cells on the GPU
- `nmf_gpu_sc`: Run single-run NMF on the GPU over single cell or meta cell data
- `stabilised_nmf_gpu_sc`: Run stabilised (multi-run) NMF on the GPU over single cell or meta cell data
- `consensus_nmf_gpu_sc`: Run consensus NMF on the GPU over single cell or meta cell data
- `nmf_k_sweep_gpu_sc`: Sweep k for consensus NMF on the GPU over single cell or meta cell data
- `params_sc_harmony_v2_gpu`: Default parameters for Harmony v2 GPU batch correction
- `params_sc_fast_cluster_gpu`: Default parameters for GPU fast Louvain clustering
- `params_scrublet_gpu`: Wrapper function for GPU Scrublet doublet detection parameters
- `params_sc_bbknn_gpu`: Wrapper function for the GPU BBKNN parameters
- `params_nebula_gpu`: Wrapper function for parameters for GPU NEBULA
- `params_knn_gpu_defaults`: Default parameters for the GPU nearest neighbour backends

## GPU-accelerated UMAP

GPU-accelerated UMAP with pluggable kNN backends (IVF, NN-descent/CAGRA, exhaustive) and CPU/GPU optimisers. Also a version for the single cells analysis suite in bixverse.

- `umap_gpu`: Rust-based UMAP (GPU)
- `umap_gpu_sc`: Run UMAP on a SingleCells object (GPU)
- `params_umap_gpu`: Wrapper function to generate UMAP parameters (GPU version)
- `params_nn_gpu`: Wrapper function to generate GPU nearest neighbour parameters

## GPU-accelerated t-SNE

t-SNE with GPU-accelerated kNN backends (IVF, NN-descent, exhaustive). Optimiser (BH or FFT) runs on CPU via the Rust implementation in manifoldsR. Also a version for the single cells analysis suite in bixverse.

- `tsne_gpu`: Rust-based t-SNE (GPU)
- `tsne_gpu_sc`: Run t-SNE on a SingleCells object (GPU)
- `params_tsne_gpu`: Wrapper function to generate t-SNE parameters (GPU version)

## Parametric UMAP

Want to train a neural net to do your UMAP embeddings? Want to train flexibly on CPU or GPU? Here's what you need.

- `parametric_umap`: Parametric UMAP
- `params_parametric_umap`: Wrapper function to generate parametric UMAP parameters
- `predict.ParametricUmapModel`: Predict embeddings for new data using a trained parametric UMAP model
- `load_parametric_umap`: Load a parametric UMAP as a qs2 file
- `save_parametric_umap`: Save a parametric UMAP as a qs2 file

## Other GPU-accelerated methods

Other GPU-accelerated methods that you might find useful... k-means clustering and a direct interface into the GPU-accelerated kNN searches can be found here.

- `k_means_cluster_gpu`: GPU-accelerated k-means clustering
- `params_kmeans_gpu`: Default parameters for GPU k-means
- `generate_knn_graph_gpu`: Generate a k-nearest neighbour graph (GPU-accelerated)
- `gpu_available`: Is a GPU available

## Agent skill

Teach your coding agent how to use the package.

- `install_agent_skill_gpu`: Install the bixverse.gpu agent skill

## Rust wrappers

Everything rusty - only use this if you know what you are doing... Maybe useful for your own package? Use with care and read the documentation!

30 `rs_*` functions are exposed here. They are the raw extendr bindings with no input validation. Use the R wrapper instead; only reach for these if you are building on top of bixverse.gpu and know exactly what you are doing.

Except these, which have no R wrapper and are meant to be called:

- `rs_cor_gpu`: GPU-accelerated correlation calculations
- `rs_cov_gpu`: GPU-accelerated covariance calculations

## Deprecated

Kept so existing scripts keep running. Each one warns and forwards to its replacement.

- `generate_cagra_knn_sc`: Generate CAGRA GPU kNN data for single cells (deprecated)
- `find_neighbours_cagra_sc`: Find neighbours via CAGRA GPU-acceleration for single cells (deprecated)
- `params_sc_cagra`: Default parameters for CAGRA-style kNN search (deprecated)
- `params_sc_ivf`: Default parameters for IVF-GPU kNN search (deprecated)
- `rs_cagra_gpu_knn`: CAGRA-style GPU-accelerated kNN graph (deprecated)
- `rs_exhaustive_gpu_knn`: Exhaustive GPU-accelerated kNN graph (deprecated)
- `rs_ivf_gpu_knn`: IVF-GPU-accelerated kNN graph (deprecated)

## Not on the package website

Exported but absent from `_pkgdown.yml`. Mostly internal constructors and `rs_*` bindings that take on-disk streaming input. Usable, but undocumented on the website, so read the roxygen with `?fn` first.

- `.prepare_tsne_params_gpu`: Internal helper to prepare the t-SNE parameters (GPU version)
- `.prepare_umap_params_gpu`: Internal helper to prepare the UMAP parameters (GPU version)

