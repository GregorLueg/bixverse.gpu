//! GPU PHATE: R list parsing and the wrapper into `manifolds_rs::phate_gpu`.
//!
//! Every default comes from `PhateParams::default()`, so a key missing from the
//! R list falls back to whatever manifolds-rs ships. An explicit `NULL` is not
//! the same as a missing key for `decay` (binary kernel) and `n_landmarks` (full
//! operator).

use bixverse_r::prelude::IntoExtendrErr;
use cubecl::wgpu::WgpuDevice;
use extendr_api::*;
use faer::{Mat, MatRef};
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rand_distr::{Distribution, StandardNormal};
use rustc_hash::FxHashMap;

use crate::embeddings::utils::{get_params_nn_ann_gpu, knn_method_for_backend};
use crate::Backend;

/////////////
// Helpers //
/////////////

/// Read an optional real from the parameter map.
///
/// ### Params
///
/// * `p` - The R parameter list as a map
/// * `key` - Key to read
/// * `default` - Value when the key is absent
///
/// ### Returns
///
/// `default` if the key is absent, `None` if it is `NULL`, the value otherwise.
fn nullable_real<T>(p: &FxHashMap<&str, Robj>, key: &str, default: Option<T>) -> Option<T>
where
    T: ManifoldsFloat,
{
    match p.get(key) {
        None => default,
        Some(v) => v.as_real().map(|v| T::from_f64(v).unwrap()),
    }
}

/// Read an optional integer from the parameter map.
///
/// ### Params
///
/// * `p` - The R parameter list as a map
/// * `key` - Key to read
/// * `default` - Value when the key is absent
///
/// ### Returns
///
/// `default` if the key is absent, `None` if it is `NULL`, the value otherwise.
fn nullable_usize(p: &FxHashMap<&str, Robj>, key: &str, default: Option<usize>) -> Option<usize> {
    match p.get(key) {
        None => default,
        Some(v) => v.as_integer().map(|v| v as usize),
    }
}

/// Read a string from the parameter map.
///
/// ### Params
///
/// * `p` - The R parameter list as a map
/// * `key` - Key to read
/// * `default` - Value when the key is absent or not a string
///
/// ### Returns
///
/// The string.
fn string_or(p: &FxHashMap<&str, Robj>, key: &str, default: &str) -> String {
    p.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

////////////
// Params //
////////////

/// Parse the GPU PHATE parameters from an R list.
///
/// ### Params
///
/// * `r_list` - Merged `params_nn_gpu()` and `params_phate_gpu()` list, plus
///   `knn_method`.
/// * `n_dim` - Number of output dimensions
/// * `k` - Neighbours that set the kernel bandwidth
///
/// ### Returns
///
/// `(PhateParams, NearestNeighbourParamsGpu)`. The CPU `nn_params` inside
/// `PhateParams` stay at their defaults; `phate_gpu` only reads the metric
/// from the GPU parameters.
fn get_params_phate_gpu<T>(
    r_list: List,
    n_dim: usize,
    k: usize,
) -> Result<(PhateParams<T>, NearestNeighbourParamsGpu<T>)>
where
    T: ManifoldsFloat,
{
    let nn_params = get_params_nn_ann_gpu::<T>(r_list.clone())?;
    let p: FxHashMap<&str, Robj> = r_list.iter().collect();

    let def = PhateParams::<T>::default();
    let def_diff = &def.diffusion_params;
    let t_max = match def_diff.t {
        PhateTime::Auto { t_max } => Some(t_max),
        PhateTime::Fixed(_) => None,
    };

    let diffusion_params = PhateDiffusionParams::new(
        nullable_real(&p, "decay", def_diff.decay),
        nullable_real(&p, "bandwidth_scale", Some(def_diff.bandwidth_scale))
            .unwrap_or(def_diff.bandwidth_scale),
        nullable_real(&p, "thresh", Some(def_diff.thresh)).unwrap_or(def_diff.thresh),
        string_or(&p, "graph_symmetry", &def_diff.graph_symmetry),
        nullable_usize(&p, "n_landmarks", def_diff.n_landmarks),
        string_or(&p, "landmark_method", &def_diff.landmark_method),
        nullable_usize(&p, "n_svd", def_diff.n_svd),
        string_or(&p, "k_means", &def_diff.k_means),
        nullable_usize(&p, "t_max", t_max),
        nullable_usize(&p, "t_custom", None),
        nullable_real(&p, "gamma", Some(def_diff.gamma)).unwrap_or(def_diff.gamma),
    );

    let phate_params = PhateParams::new(
        n_dim,
        k,
        string_or(&p, "knn_method", &def.ann_type),
        def.nn_params.clone(),
        diffusion_params,
        string_or(&p, "mds_method", &def.mds_method),
        nullable_usize(&p, "mds_iter", def.mds_iter),
        p.get("randomised")
            .and_then(|v| v.as_bool())
            .unwrap_or(def.randomised),
    );

    Ok((phate_params, nn_params))
}

///////////
// PHATE //
///////////

/// Wrapper into the GPU PHATE of `manifolds-rs`.
///
/// The kNN search and, for spectral landmarks, the randomised SVD and k-means
/// run on the device. With `Backend::Mlx` both move to MLX: the kNN method gets
/// the `_mlx` suffix and `"spectral"` becomes `"spectral_mlx"`.
///
/// ### Params
///
/// * `data` - Input data, samples x features
/// * `pre_computed_knn` - Optional kNN graph excluding self. Should be
///   `6k + 5` wide.
/// * `n_dim` - Number of output dimensions
/// * `k` - Neighbours that set the kernel bandwidth
/// * `phate_params` - Named R list with the PHATE and GPU kNN parameters
/// * `seed` - Random seed for reproducibility
/// * `backend` - Device backend
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The embedding as samples x `n_dim`.
#[allow(clippy::too_many_arguments)]
pub fn phate_manifold_gpu<T>(
    data: MatRef<T>,
    pre_computed_knn: PreComputedKnn<T>,
    n_dim: usize,
    k: usize,
    phate_params: List,
    seed: usize,
    backend: Backend,
    verbose: usize,
) -> Result<Mat<T>>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    let device: WgpuDevice = Default::default();

    let (mut params, nn_params) = get_params_phate_gpu::<T>(phate_params, n_dim, k)?;
    if backend == Backend::Mlx {
        params.ann_type = knn_method_for_backend(&params.ann_type, backend);
        if params
            .diffusion_params
            .landmark_method
            .eq_ignore_ascii_case("spectral")
        {
            params.diffusion_params.landmark_method = "spectral_mlx".to_string();
        }
    }

    let res = phate_gpu::<T>(
        data,
        pre_computed_knn,
        params,
        &nn_params,
        device.clone().into(),
        seed,
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    if backend == Backend::Wgpu {
        let client = cubecl::Device::from(device.clone()).client();
        let _ = client.memory_cleanup();
    }

    let ncol = res.len();
    let nrow = res[0].len();

    Ok(Mat::from_fn(nrow, ncol, |i, j| res[j][i]))
}
