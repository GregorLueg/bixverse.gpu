# Wrapper function to generate PHATE parameters (GPU version)

Wrapper function to generate PHATE parameters (GPU version)

## Usage

``` r
params_phate_gpu(
  decay = 40,
  bandwidth_scale = 1,
  thresh = 1e-04,
  graph_symmetry = c("additive", "multiplicative", "mnn", "none"),
  n_landmarks = 2048L,
  landmark_method = c("spectral", "random", "density"),
  n_svd = NULL,
  t_max = 100L,
  t_custom = NULL,
  gamma = 1,
  mds_method = c("sgd_dense", "classic"),
  mds_iter = NULL,
  randomised = TRUE
)
```

## Arguments

- decay:

  Numeric or `NULL`. Alpha decay exponent of the kernel. Higher values
  give sharper transitions. `NULL` gives a binary connectivity kernel.
  Defaults to `40.0`.

- bandwidth_scale:

  Numeric. Multiplicative factor on the kernel bandwidth. Defaults to
  `1.0`.

- thresh:

  Numeric. Affinities below this are set to zero. Neighbours past `k`
  join the kernel while their affinity clears it. Defaults to `1e-04`.

- graph_symmetry:

  String. How to symmetrise the affinity graph. One of
  `c("additive", "multiplicative", "mnn", "none")`. Defaults to
  `"additive"`.

- n_landmarks:

  Integer or `NULL`. Number of landmarks for the compressed diffusion
  operator. `NULL`, or a value of at least the number of samples, runs
  the full N x N operator, which gets painful fast. Defaults to `2048L`.

- landmark_method:

  String. How to select the landmarks. `"spectral"` runs its randomised
  SVD and k-means on the GPU (or on MLX with `backend = "mlx"`). One of
  `c("spectral", "random", "density")`. Defaults to `"spectral"`.

- n_svd:

  Integer or `NULL`. Singular vectors for spectral landmark selection.
  If `NULL`, defaults to `100` on the Rust side. Defaults to `NULL`.

- t_max:

  Integer. Largest diffusion time the Von Neumann entropy knee search
  considers. Defaults to `100L`.

- t_custom:

  Integer or `NULL`. Fixed diffusion time. If `NULL`, picks the knee
  automatically. Defaults to `NULL`.

- gamma:

  Numeric. Informational distance constant. `1` is the log potential.
  Defaults to `1.0`.

- mds_method:

  String. MDS algorithm for the final embedding. One of
  `c("sgd_dense", "classic")`. Defaults to `"sgd_dense"`.

- mds_iter:

  Integer or `NULL`. Number of SGD-MDS iterations. If `NULL`, the Rust
  side runs 300 below 1000 points, 500 below 5000 and 800 otherwise
  (points being the landmarks when landmarks are used). Defaults to
  `NULL`.

- randomised:

  Boolean. Use randomised SVD for classic MDS
  (`mds_method = "classic"`). SGD-MDS initialises from a randomised
  classic MDS either way. Defaults to `TRUE`.

## Value

A named list with the following elements:

- decay - Numeric or `NULL`. Alpha decay exponent of the kernel. Higher
  values give sharper transitions. `NULL` gives a binary connectivity
  kernel. Defaults to `40.0`.

- bandwidth_scale - Numeric. Multiplicative factor on the kernel
  bandwidth. Defaults to `1.0`.

- thresh - Numeric. Affinities below this are set to zero. Neighbours
  past `k` join the kernel while their affinity clears it. Defaults to
  `1e-04`.

- graph_symmetry - String. How to symmetrise the affinity graph. One of
  `c("additive", "multiplicative", "mnn", "none")`. Defaults to
  `"additive"`.

- n_landmarks - Integer or `NULL`. Number of landmarks for the
  compressed diffusion operator. `NULL`, or a value of at least the
  number of samples, runs the full N x N operator, which gets painful
  fast. Defaults to `2048L`.

- landmark_method - String. How to select the landmarks. `"spectral"`
  runs its randomised SVD and k-means on the GPU (or on MLX with
  `backend = "mlx"`). One of `c("spectral", "random", "density")`.
  Defaults to `"spectral"`.

- n_svd - Integer or `NULL`. Singular vectors for spectral landmark
  selection. If `NULL`, defaults to `100` on the Rust side. Defaults to
  `NULL`.

- t_max - Integer. Largest diffusion time the Von Neumann entropy knee
  search considers. Defaults to `100L`.

- t_custom - Integer or `NULL`. Fixed diffusion time. If `NULL`, picks
  the knee automatically. Defaults to `NULL`.

- gamma - Numeric. Informational distance constant. `1` is the log
  potential. Defaults to `1.0`.

- mds_method - String. MDS algorithm for the final embedding. One of
  `c("sgd_dense", "classic")`. Defaults to `"sgd_dense"`.

- mds_iter - Integer or `NULL`. Number of SGD-MDS iterations. If `NULL`,
  the Rust side runs 300 below 1000 points, 500 below 5000 and 800
  otherwise (points being the landmarks when landmarks are used).
  Defaults to `NULL`.

- randomised - Boolean. Use randomised SVD for classic MDS
  (`mds_method = "classic"`). SGD-MDS initialises from a randomised
  classic MDS either way. Defaults to `TRUE`.

## References

Moon, et al., Nat. Biotechnol., 2019
