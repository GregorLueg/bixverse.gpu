# Run fastMNN on the GPU

GPU counterpart of
[`bixverse::fast_mnn_sc()`](https://gregorlueg.github.io/bixverse/reference/fast_mnn_sc.html),
implementing the fast mutual nearest neighbour correction from
Haghverdi, et al. Batches are merged one after the other: MNN pairs
between the merged block and the next batch give correction vectors,
which are smoothed with a tricube kernel and applied to every cell of
that batch.

## Usage

``` r
fast_mnn_gpu_sc(
  object,
  batch_column,
  no_embd_to_use = NULL,
  fastmnn_params = params_sc_fastmnn_gpu(),
  seed = 42L,
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `SingleCellsSubset` class from `bixverse`.

- batch_column:

  String. The column with the batch information in the obs data of the
  class.

- no_embd_to_use:

  Optional integer. Number of PCs to use. If `NULL` all will be used.

- fastmnn_params:

  List. Output of
  [`params_sc_fastmnn_gpu()`](https://gregorlueg.github.io/bixverse.gpu/reference/params_sc_fastmnn_gpu.md).

- seed:

  Integer. Random seed.

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

The object with a `"mnn_gpu"` embedding added. If the batch column only
has one batch, the object is returned as is with a warning.

## Details

Only the neighbour searches move to the device: both MNN directions and
the tricube search, three index builds per merge. Those dominate the run
time on the CPU path. Centring, MNN pairing and the tricube correction
are shared with the CPU implementation.

The PCA stored in the object is corrected as is. If you want the PCA on
batch-aware HVGs, as
[`bixverse::fast_mnn_sc()`](https://gregorlueg.github.io/bixverse/reference/fast_mnn_sc.html)
can recompute it, run
[`bixverse::find_hvg_batch_aware_sc()`](https://gregorlueg.github.io/bixverse/reference/find_hvg_batch_aware_sc.html)
and the PCA first.

Results match the CPU path with `knn_method = "exhaustive"` on both
sides, since both searches are exact. The approximate backends will not
match.

## References

Haghverdi, et al., Nat Biotechnol, 2018
