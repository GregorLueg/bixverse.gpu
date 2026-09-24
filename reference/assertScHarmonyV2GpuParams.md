# Assert Harmony v2 GPU params

Assert Harmony v2 GPU params

## Usage

``` r
assertScHarmonyV2GpuParams(x, .var.name = checkmate::vname(x), add = NULL)
```

## Arguments

- x:

  The object to check.

- .var.name:

  Name of the checked object to print in assertions.

- add:

  Collection to store assertion messages. See
  [`checkmate::makeAssertCollection()`](https://mllg.github.io/checkmate/reference/AssertCollection.html).

## Value

Invisibly returns the checked object if the assertion is successful.
