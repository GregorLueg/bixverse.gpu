# Pull the meta cell counts for an NMF run

All four `MetaCells` methods slice the counts the same way, so the assay
choice lives here rather than four times over.

## Usage

``` r
.mc_nmf_counts(object, sel, use_second_layer)
```

## Arguments

- object:

  `MetaCells` class from `bixverse`.

- sel:

  List. Output of `bixverse:::.resolve_mc_nmf_selection()`.

- use_second_layer:

  Boolean. If `TRUE`, takes the normalised counts.

## Value

A named list with `data`, `indptr`, `indices`, `cs_type`, `nrow` and
`ncol`, ready for the Rust bindings.
