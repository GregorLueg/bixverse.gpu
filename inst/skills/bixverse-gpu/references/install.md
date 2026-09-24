# Installing bixverse.gpu

## The package itself

r-universe gives a pre-built binary, no Rust toolchain and no compile:

```r
install.packages(
  "bixverse.gpu",
  repos = c("https://gregorlueg.r-universe.dev", "https://cloud.r-project.org")
)
```

`bixverse` and `manifoldsR` are hard dependencies and resolve from the same
r-universe repo. Keep it in the repos list, otherwise they will not be found.

### From source

Needs Rust (`rustc >= 1.95`), cargo and `xz`, plus rextendr in R:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

```r
install.packages("rextendr")
options(repos = c("https://gregorlueg.r-universe.dev", getOption("repos")))
devtools::install_github("https://github.com/GregorLueg/bixverse.gpu")
```

The first source build compiles the whole Rust dependency tree (burn, cubecl,
wgpu, hdf5) and takes a while. That is not a hang.

## The GPU

Nothing to install beyond working drivers. WGPU picks up Metal on macOS,
Vulkan on Linux, DX12 or Vulkan on Windows. No CUDA, no vendor SDK.

```r
bixverse.gpu::gpu_available()
```

`FALSE` means WGPU could not initialise an adapter. That is a driver problem,
not a package problem. The cubecl book covers per-platform set-up:
https://burn.dev/books/cubecl/getting-started/installation.html. On a headless
Linux box that usually means installing the Vulkan loader and a driver
(`libvulkan1`, `mesa-vulkan-drivers` on Debian/Ubuntu) and, if several
backends are present, forcing one with `WGPU_BACKEND=vulkan`.

## Windows

Supported. The HDF5 build used to hit `MAX_PATH`; the cargo target directory
now lives in `~/.bixverse-gpu-cargo` to stay clear of it. Older answers saying
Windows does not work predate that.

One gap: FFT t-SNE. FFTW does not build there, so
`tsne_gpu(approx_type = "fft")` and `tsne_gpu_sc(approx_type = "fft")` error.
Barnes-Hut (`"bh"`, the default) works everywhere.

## CPU training for parametric UMAP

`parametric_umap(use_gpu = FALSE)` trains on the CPU via burn's `flex`
backend. No OpenBLAS or Accelerate set-up is needed any more; older
instructions mentioning them are out of date.
