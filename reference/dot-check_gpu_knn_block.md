# Check a GPU kNN parameter block

Shared validation for the flat GPU kNN block, used by both the GPU
Scrublet and the GPU BBKNN parameters. They take the same keys bar `k`
and `extract_knn`, which BBKNN ignores, hence `required` rather than a
fixed name set.

## Usage

``` r
.check_gpu_knn_block(x, required)
```

## Arguments

- x:

  The kNN block to check.

- required:

  Character vector of key names that must be present.

## Value

`TRUE` if the check was successful, otherwise an error message.
