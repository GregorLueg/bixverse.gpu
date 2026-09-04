# Assert that a GPU is available

Hard errors when no WGPU adapter can be initialised. Sits at the top of
the user-facing functions so the failure names the function the user
called rather than the `rs_` wrapper underneath. The Rust side carries
the same guard for direct `rs_` calls.

## Usage

``` r
assert_gpu()
```

## Value

Invisibly `TRUE`. Called for the error.
