# Check a GPU kNN parameter block

Validation for the flat GPU kNN block of the GPU Scrublet parameters,
whose block depends on `knn_backend` and so cannot come from devforge.

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
