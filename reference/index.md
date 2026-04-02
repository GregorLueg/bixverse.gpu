# Package index

## Single cell GPU kNN

You need kNN graphs on large single cell data sets and you have some
VRAM to spare? Look no further.

- [`find_neighbours_gpu_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_gpu_sc.md)
  : Find GPU-accelerated neighbours for single cells (exhaustive / IVF)
- [`params_sc_ivf()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_ivf.md)
  : Default parameters for IVF-GPU kNN search
- [`find_neighbours_cagra_sc()`](https://gregorlueg.github.io/bixverse.gpu/reference/find_neighbours_cagra_sc.md)
  : Find CAGRA GPU-accelerated neighbours for single cells
- [`params_sc_cagra()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_cagra.md)
  : Default parameters for CAGRA-style kNN search

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

## Rust wrappers

Everything rusty - only use this if you know what you are doing… Maybe
useful for your own package? Use with care and read the documentation!

- [`rs_cagra_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cagra_gpu_knn.md)
  : Generate a CAGRA-style GPU-accelerated kNN graph
- [`rs_exhaustive_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_exhaustive_gpu_knn.md)
  : Generate an GPU-accelerated kNN graph from an exhaustive search
- [`rs_ivf_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_ivf_gpu_knn.md)
  : Generate an IVF-GPU-accelerated kNN graph
- [`rs_parametric_umap_predict()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_parametric_umap_predict.md)
  : Predict new data using a trained parametric UMAP model
- [`rs_parametric_umap()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_parametric_umap.md)
  : Parametric UMAP implementation
