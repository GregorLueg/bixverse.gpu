# Build a Bonsai tree over the cells, Sanity on the GPU

GPU counterpart of
[`bixverse::bonsai_sc()`](https://gregorlueg.github.io/bixverse/reference/bonsai_sc.html).
Sanity, which turns the raw counts into posterior log fold changes with
error bars and is most of the runtime on the CPU, runs on the WGPU
backend. Genes are streamed through it in chunks and only the ones
passing Bonsai's signal-to-noise filter are kept. The tree search and
the layout run on the CPU, as they do in `bixverse`.

The device computes in `f32`, so the posteriors match the CPU run to
within the resolution of Sanity's variance grid rather than bit for bit,
and a gene sitting right at the signal-to-noise threshold can end up on
the other side.

## Usage

``` r
bonsai_gpu_sc(
  object,
  hvg = NULL,
  bonsai_params = bixverse::params_sc_bonsai(),
  .verbose = TRUE
)
```

## Arguments

- object:

  `SingleCells` or `MetaCells` class.

- hvg:

  Optional integer. Restrict the candidate genes to these. Please
  provide 1-indexed genes here! If `NULL`, every gene in the object is a
  candidate.

- bonsai_params:

  List. See
  [`bixverse::params_sc_bonsai()`](https://gregorlueg.github.io/bixverse/reference/params_sc_bonsai.html).

- .verbose:

  Boolean or integer. Controls verbosity and returns run times. `FALSE`
  -\> quiet, `TRUE` or `1L` -\> normal verbosity, `2L` -\> detailed
  verbosity.

## Value

A `BonsaiTree` S3 object, see
[`bixverse::bonsai_sc()`](https://gregorlueg.github.io/bixverse/reference/bonsai_sc.html).

## References

de Groot, et al., Nat. Biotechnol., 2026; Breda, et al., Nat.
Biotechnol., 2021.
