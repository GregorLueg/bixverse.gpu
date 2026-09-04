# Is a GPU available

Reports whether a usable WGPU adapter is present. Every function in this
package needs one, and without it the Rust side aborts rather than
returning an error, so this is the way to branch before calling into it.

The probe initialises the same client the GPU functions use, so a `TRUE`
means they will genuinely run rather than that a device merely exists.
It is cached for the session, and the client is left warm, so the first
real call afterwards skips the setup cost.

## Usage

``` r
gpu_available()
```

## Value

Boolean. `TRUE` when a WGPU adapter could be initialised.

## Examples

``` r
if (gpu_available()) {
  # ... GPU path
}
#> NULL
```
