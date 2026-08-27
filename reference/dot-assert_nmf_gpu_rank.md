# Check the rank against the GPU solver cap

Above
[NMF_GPU_MAX_RANK](https://gregorlueg.github.io/bixverse.gpu/reference/NMF_GPU_MAX_RANK.md)
the Rust side refuses the solve, so catch it here with something that
says what to do instead.

## Usage

``` r
.assert_nmf_gpu_rank(k)
```

## Arguments

- k:

  Integer vector. The rank(s) to check.

## Value

`NULL`, invisibly. Called for the error.
