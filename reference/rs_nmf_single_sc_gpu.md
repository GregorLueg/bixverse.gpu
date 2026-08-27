# Run NMF (HALS) on the GPU over a set of single cells and genes

**\[experimental\]** GPU counterpart of
[`bixverse::rs_nmf_single_sc()`](https://gregorlueg.github.io/bixverse/reference/rs_nmf_single_sc.html).
The counts are loaded from the binary store once, uploaded to the
device, and the whole HALS loop runs there. The NNDSVD initialisation
stays on the host.

## Usage

``` r
rs_nmf_single_sc_gpu(
  f_path_gene,
  gene_indices,
  cell_indices,
  k,
  preprocessing,
  use_second_layer,
  nmf_hals_params,
  seed,
  verbose
)
```

## Arguments

- f_path_gene:

  Path to the `counts_genes.bin` file.

- gene_indices:

  Integer vector. 0-indexed(!) positions of the genes to include.

- cell_indices:

  Integer vector. 0-indexed(!) positions of cells to include in the
  analysis.

- k:

  Integer. Number of latent factors to return. At most 128, the GPU
  solver's rank cap.

- preprocessing:

  String. One of `c("none", "sd", "sqrt_sd")`. Takes the data as is, or
  scales by standard deviation or squared standard deviation per
  feature.

- use_second_layer:

  Boolean. If `TRUE`, runs NMF on the normalised counts; if `FALSE`, on
  the raw counts.

- nmf_hals_params:

  Named list. Contains the NMF parameters.

- seed:

  Integer. Random seed for initialisation.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list with the following items

- w - The left factor matrix (n_cells x k)

- h - The right factor matrix (k x n_genes)

- final_loss - Loss at the final iteration

- n_iter - Number of iterations the algorithm run for

- converged - Did the NMF algorithm converge
