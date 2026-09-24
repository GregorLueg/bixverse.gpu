# Normalise a GPU kNN method name

The package says `"nndescent"` everywhere, but the parser on the Rust
side of the single cell GPU kNN only matches `"cagra"`, `"cagra_gpu"`
and `"nndescent_gpu"`. A bare `"nndescent"` is not an error there: it
prints a warning and silently runs exhaustive instead. That is a
performance regression nobody would notice, so translate before crossing
over.

Idempotent, so it is safe to apply both in the parameter wrapper and
again at the call site.

## Usage

``` r
.normalise_gpu_knn_method(x)
```

## Arguments

- x:

  String. The kNN method name.

## Value

The string Rust understands.
