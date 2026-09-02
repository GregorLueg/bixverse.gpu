use bixverse_rs::prelude::IntoExtendrErr;
use cubecl::prelude::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use extendr_api::*;
use faer::{Mat, MatRef};
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashMap;

#[cfg(not(target_os = "windows"))]
use manifolds_rs::utils::fft::FftwFloat;

use crate::embeddings::utils::get_params_nn_ann_gpu;

/////////////
// Helpers //
/////////////

/// Helper function to generate t-SNE optimisation parameters
///
/// # Arguments
///
/// * `r_list` - List with t-SNE optimisation parameters
///
/// # Returns
///
/// `TsneOptimParams` with sensible defaults if not found in the list
fn get_params_tsne_optim<T>(r_list: List) -> Result<TsneOptimParams<T>>
where
    T: ManifoldsFloat,
{
    let optim_params: HashMap<&str, Robj> = r_list.try_into()?;

    let lr = optim_params
        .get("lr")
        .and_then(|v| v.as_real())
        .map(|v| T::from_f64(v).unwrap());

    let late_exag_factor = optim_params
        .get("late_exag_factor")
        .and_then(|v| v.as_real())
        .map(|v| T::from_f64(v).unwrap());

    let n_epochs = optim_params
        .get("n_epochs")
        .and_then(|v| v.as_integer())
        .unwrap_or(1000) as usize;

    let early_exag_iter = optim_params
        .get("early_exag_iter")
        .and_then(|v| v.as_integer())
        .unwrap_or(250) as usize;

    let early_exag_factor = optim_params
        .get("early_exag_factor")
        .and_then(|v| v.as_real())
        .map(|v| T::from_f64(v).unwrap())
        .unwrap_or(T::from_f64(12.0).unwrap());

    let theta = optim_params
        .get("theta")
        .and_then(|v| v.as_real())
        .map(|v| T::from_f64(v).unwrap())
        .unwrap_or(T::from_f64(0.5).unwrap());

    let n_interp_points = optim_params
        .get("n_interp_points")
        .and_then(|v| v.as_integer())
        .unwrap_or(3) as usize;

    Ok(TsneOptimParams {
        n_epochs,
        lr,
        early_exag_iter,
        early_exag_factor,
        late_exag_factor,
        theta,
        n_interp_points,
    })
}

////////////
// Params //
////////////

/// InternalTsneParams
///
/// Overall wrapper over various parameters needed for tSNE
#[derive(Debug)]
pub struct InternalTsneParamsGpu<T> {
    /// Which of the approximate nearest neighbour searches to use.
    pub knn_method: String,
    /// The nearest neighbour parameters that are forwarded to the approximate
    /// nearest neighbour methods.
    pub param_knn: NearestNeighbourParamsGpu<T>,
    /// Which initialisation to use. One of `"spectral"`, `"pca"`, or
    /// `"random"`.
    pub init: String,
    /// When setting initialisation to `"pca"` shall randomised SVD be used (can
    /// make it faster on large data sets).
    pub randomised: bool,
    /// The TsneOptimParameters
    pub param_optimiser: TsneOptimParams<T>,
}

impl<T> InternalTsneParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Generate t-SNE parameters from an R list
    ///
    /// # Arguments
    ///
    /// * `r_list` - R list with all needed parameters for t-SNE
    ///
    /// # Returns
    ///
    /// The `InternalTsneParams`
    pub fn from_r_list(r_list: List) -> Result<Self> {
        let nn_params = get_params_nn_ann_gpu(r_list.clone())?;
        let optim_params = get_params_tsne_optim(r_list.clone())?;

        let tsne_params: HashMap<&str, Robj> = r_list.try_into()?;

        let init = std::string::String::from(
            tsne_params
                .get("init")
                .and_then(|v| v.as_str())
                .unwrap_or("pca"),
        );

        let knn_method = std::string::String::from(
            tsne_params
                .get("knn_method")
                .and_then(|v| v.as_str())
                .unwrap_or("kmknn"),
        );

        let randomised = tsne_params
            .get("randomised")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Self {
            param_knn: nn_params,
            knn_method,
            init,
            randomised,
            param_optimiser: optim_params,
        })
    }
}

//////////
// tSNE //
//////////

/// Wrapper function into the t-SNE implementation in `manifolds-rs`
///
/// This function wraps around the `manifolds-rs` function and exposes it to
/// R via another function. It uses under the hood the GPU-accelerated kNN
/// searches.
///
/// ### Params
///
/// * `data` - Input data matrix for t-SNE
/// * `pre_computed_knn` - Optional pre-computed kNN to be used.
/// * `n_dim` - Number of dimensions to reduce to (typically 2)
/// * `approximation` - String. One of `"bh"` for the Barnes Hut approximation
///   or `"fft"` for the Fast Fourier Transformation-accelerated one.
/// * `perplexity` - Perplexity parameter (typical: 5-50)
/// * `tsne_params` - Named R list with all t-SNE parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// t-SNE embeddings as matrix
#[allow(clippy::too_many_arguments)]
#[cfg(not(target_os = "windows"))]
pub fn tsne_manifold_gpu<T>(
    data: MatRef<T>,
    pre_computed_knn: PreComputedKnn<T>,
    n_dim: usize,
    approx_type: &str,
    perplexity: T,
    tsne_params: List,
    seed: usize,
    verbose: usize,
) -> Result<Mat<T>>
where
    T: ManifoldsFloatGpu + FftwFloat,
    StandardNormal: Distribution<T>,
{
    assert!(
        n_dim == 2,
        "At the moment, this tSNE implementation only supports n_dim = 2"
    );

    let device: WgpuDevice = Default::default();

    let tsne_params_internal = InternalTsneParamsGpu::from_r_list(tsne_params)?;

    let tsne_params = TsneParamsGpu {
        n_dim,
        perplexity,
        ann_type: tsne_params_internal.knn_method,
        initialisation: tsne_params_internal.init,
        nn_params: tsne_params_internal.param_knn,
        optim_params: tsne_params_internal.param_optimiser,
        randomised_init: tsne_params_internal.randomised,
        init_range: Some(T::from_f64(1e-2).unwrap()),
    };

    let res = tsne_gpu::<T, WgpuRuntime>(
        data,
        pre_computed_knn,
        &tsne_params,
        approx_type,
        device.clone(),
        seed,
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let ncol = res.len();
    let nrow = res[0].len();

    Ok(Mat::from_fn(nrow, ncol, |i, j| res[j][i]))
}

/// Wrapper function into the t-SNE implementation in `manifolds-rs`
///
/// This function wraps around the `manifolds-rs` function and exposes it to
/// R via another function. It uses under the hood the GPU-accelerated kNN
/// searches.
///
/// ### Params
///
/// * `data` - Input data matrix for t-SNE
/// * `pre_computed_knn` - Optional pre-computed kNN to be used.
/// * `n_dim` - Number of dimensions to reduce to (typically 2)
/// * `approximation` - String. One of `"bh"` for the Barnes Hut approximation
///   or `"fft"` for the Fast Fourier Transformation-accelerated one.
/// * `perplexity` - Perplexity parameter (typical: 5-50)
/// * `tsne_params` - Named R list with all t-SNE parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// t-SNE embeddings as matrix
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "windows")]
pub fn tsne_manifold_gpu<T>(
    data: MatRef<T>,
    pre_computed_knn: PreComputedKnn<T>,
    n_dim: usize,
    approx_type: &str,
    perplexity: T,
    tsne_params: List,
    seed: usize,
    verbose: usize,
) -> Result<Mat<T>>
where
    T: ManifoldsFloatGpu + FftwFloat,
    StandardNormal: Distribution<T>,
{
    assert!(
        n_dim == 2,
        "At the moment, this tSNE implementation only supports n_dim = 2"
    );

    let device: WgpuDevice = Default::default();

    let tsne_params_internal = InternalTsneParamsGpu::from_r_list(tsne_params)?;

    let tsne_params = TsneParamsGpu {
        n_dim,
        perplexity,
        ann_type: tsne_params_internal.knn_method,
        initialisation: tsne_params_internal.init,
        nn_params: tsne_params_internal.param_knn,
        optim_params: tsne_params_internal.param_optimiser,
        randomised_init: tsne_params_internal.randomised,
        init_range: Some(T::from_f64(1e-2).unwrap()),
    };

    let res = tsne_gpu::<T, WgpuRuntime>(
        data,
        pre_computed_knn,
        &tsne_params,
        approx_type,
        device.clone(),
        seed,
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let ncol = res.len();
    let nrow = res[0].len();

    Ok(Mat::from_fn(nrow, ncol, |i, j| res[j][i]))
}
