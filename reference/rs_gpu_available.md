# Check whether a usable GPU adapter is present

Probes for a WGPU adapter by initialising the same client every GPU
function in this package goes through, so a `TRUE` here means those
functions will actually run rather than that a device merely exists. The
result is cached for the session and the client stays warm, so the first
real call after a successful probe skips the setup cost.

## Usage

``` r
rs_gpu_available()
```

## Value

Boolean. `TRUE` when a WGPU adapter could be initialised.
