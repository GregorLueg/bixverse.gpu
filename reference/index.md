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

## Rust wrappers

Everything rusty - only use this if you know what you are doing… Maybe
useful for your own package? Use with care and read the documentation!

- [`rs_exhaustive_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_exhaustive_gpu_knn.md)
  : Generate an GPU-accelerated kNN graph from an exhaustive search
- [`rs_ivf_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_ivf_gpu_knn.md)
  : Generate an IVF-GPU-accelerated kNN graph
- [`rs_cagra_gpu_knn()`](https://gregorlueg.github.io/bixverse.gpu/reference/rs_cagra_gpu_knn.md)
  : Generate a CAGRA-style GPU-accelerated kNN graph
