# Check GPU Scrublet parameters

Checkmate extension for checking the GPU Scrublet parameters. The kNN
block is validated against whichever backend `knn_backend` names.

## Usage

``` r
checkScrubletGpu(x)
```

## Arguments

- x:

  The list to check/assert.

## Value

`TRUE` if the check was successful, otherwise an error message.
