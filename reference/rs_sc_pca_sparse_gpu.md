# Calculates sparse PCA for single cell

**\[experimental\]** Helper function that will calculate sparse PCA
without scaling the data. This has the advantage that you avoid creating
a large dense matrix due to scaling; however, it has the disadvantage
that the first PC will be heavily influenced by average expression.
`svd_solver = "randomised"` runs the randomised SVD: the sparse matrix
is multiplied with a much smaller random dense matrix. Every other
solver runs the exact block Lanczos, as there is no GPU covariance path.

## Usage

``` r
rs_sc_pca_sparse_gpu(
  f_path_gene,
  f_path_cell,
  no_pcs,
  pca_params,
  cell_indices,
  gene_indices,
  seed,
  backend = "wgpu",
  verbose
)
```

## Arguments

- f_path_gene:

  String. Path to the `counts_genes.bin` file.

- f_path_cell:

  String. Path to the `counts_cells.bin` file. Used if you wish to use
  the PFlogPF transformation.

- no_pcs:

  Integer. Number of PCs to calculate.

- pca_params:

  Named list. Contains the parameters to use for this PCA run.
  `"randomised"` runs the randomised SVD, any other `svd_solver` the
  exact block Lanczos.

- cell_indices:

  Integer. The cell indices to use. (0-indexed!)

- gene_indices:

  Integer. The gene indices to use. (0-indexed!)

- seed:

  Integer. Random seed for the sketch or the Lanczos starting block.

- backend:

  String. `"wgpu"` or `"mlx"` (Apple Silicon, needs a build with the MLX
  backend, see
  [`mlx_available()`](https://gregorlueg.github.io/bixverse.gpu/reference/mlx_available.md)).

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

A list with with the following items

- scores - The samples projected on the PCA space (solved via sparse
  SVD).

- loadings - The loadings of the features for the PCA (solved via sparse
  SVD).

- singular_values - The singular values for the PCA (solved via sparse
  SVD).
