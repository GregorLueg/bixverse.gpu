# GPU: fastMNN batch correction

**\[experimental\]** GPU equivalent of
[`bixverse::rs_mnn`](https://gregorlueg.github.io/bixverse/reference/rs_mnn.html),
implementing the fast mutual nearest neighbour correction from
Haghverdi, et al. The MNN searches and the tricube neighbour search run
on the WGPU backend; everything else is shared with the CPU
implementation.

## Usage

``` r
rs_fast_mnn_gpu(embd, batch_labels, fastmnn_params, seed, verbose)
```

## Arguments

- embd:

  Numerical matrix. The embedding to correct, usually PCA. Rows
  represent cells.

- batch_labels:

  Integer vector. These represent to which batch a given cell belongs.
  Needs to be 0-indexed!

- fastmnn_params:

  List. Parameter list, see
  [`params_sc_fastmnn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fastmnn_gpu.md).

- seed:

  Integer. Seed for reproducibility purposes.

- verbose:

  Integer. `0L` - quiet; `1L` - normal verbosity; `2L` - detailed
  verbosity.

## Value

The batch-corrected embedding, cells x dimensions, in the input cell
order.
