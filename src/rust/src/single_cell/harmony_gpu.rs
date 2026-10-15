use crate::{parse_backend, Backend};
use bixverse_rs::gpu::sc_gpu::harmony_gpu::{harmony_v2_gpu, HarmonyParamsV2Gpu};
use bixverse_rs::prelude::*;
use cubecl::wgpu::WgpuDevice;
use extendr_api::*;

#[cfg(mlx_available)]
use bixverse_rs::mlx::sc::harmony::harmony_v2_mlx;
#[cfg(mlx_available)]
use bixverse_rs::single_cell::sc_batch_correction::harmony_v2::HarmonyParamsV2;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod harmony_gpu;
    // functions
    fn rs_harmony_v2_gpu;
}

/// Harmony batch correction in Rust (version 2, GPU-accelerated)
///
/// @description
/// `r lifecycle::badge("experimental")`
/// This function implements the GPU-accelerated version 2 Harmony algorithm
/// from Patikas, et al., 2026. Only a single batch covariate is supported.
///
/// @param pca Numerical matrix, i.e., the PCA matrix you want to correct.
/// @param harmony_params List. The parameters for the Harmony (v2) GPU algorithm.
/// @param batch_labels List. Must contain exactly one element: a 0-indexed
/// integer vector representing the batch effects you wish to regress out.
/// @param seed Integer. Seed for reproducibility purposes.
/// @param backend String. `"wgpu"` or `"mlx"` (Apple Silicon, needs a build
/// with the MLX backend, see [mlx_available()]).
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @return The batch-corrected Harmony (v2) embedding space.
///
/// @export
#[extendr]
fn rs_harmony_v2_gpu(
    pca: RMatrix<f64>,
    harmony_params: List,
    batch_labels: List,
    seed: usize,
    #[extendr(default = "\"wgpu\"")] backend: &str,
    verbose: usize,
) -> extendr_api::Result<RArray<f64, 2>> {
    let backend = parse_backend(backend)?;

    // prepare everything
    let mut batch_indices: Vec<Vec<usize>> = Vec::new();
    for i in 0..batch_labels.len() {
        let batch_indices_i = batch_labels.elt(i)?;
        let batch_indices_i = batch_indices_i.as_integer_vector().unwrap();
        batch_indices.push(batch_indices_i.r_int_convert());
    }
    let harmony_params = HarmonyParamsV2Gpu::from_r_list(harmony_params)?;
    let embd = r_matrix_to_faer_fp32(&pca);

    #[cfg(mlx_available)]
    if backend == Backend::Mlx {
        let res = harmony_v2_mlx(
            embd.as_ref(),
            &batch_indices,
            &HarmonyParamsV2::from(&harmony_params),
            seed,
            verbose,
        )
        .to_extendr()?;
        return Ok(faer_to_r_matrix(res.as_ref()));
    }
    debug_assert_eq!(backend, Backend::Wgpu);

    // run harmony v2
    let device: WgpuDevice = Default::default();
    let res = harmony_v2_gpu(
        embd.as_ref(),
        &batch_indices,
        &harmony_params,
        seed,
        device.clone().into(),
        verbose,
    )
    .to_extendr()?;

    // clean up vram
    let client = cubecl::Device::from(device.clone()).client();
    let _ = client.memory_cleanup();

    Ok(faer_to_r_matrix(res.as_ref()))
}
