# Check whether the GPU adapter runs plane (subgroup) operations

Advertising plane operations is not enough: the paravirtualised GPU of a
macOS VM, as on the macos-15-intel GitHub runners, reports planes of 4
to 64 lanes but silently drops every dispatch that uses one. This
launches a single plane reduction and checks the answer. The result is
cached for the session.

## Usage

``` r
rs_gpu_plane_ops()
```

## Value

Boolean. `TRUE` when a GPU adapter is present and a plane reduction on
it returns the plane width.
