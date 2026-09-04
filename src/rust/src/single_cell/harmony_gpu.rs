use crate::ensure_gpu;
use bixverse_rs::gpu::sc_gpu::harmony_gpu::{harmony_v2_gpu, HarmonyParamsV2Gpu};
use bixverse_rs::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

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
    verbose: usize,
) -> extendr_api::Result<RArray<f64, 2>> {
    ensure_gpu()?;

    // prepare everything
    let device: WgpuDevice = Default::default();
    let mut batch_indices: Vec<Vec<usize>> = Vec::new();
    for i in 0..batch_labels.len() {
        let batch_indices_i = batch_labels.elt(i)?;
        let batch_indices_i = batch_indices_i.as_integer_vector().unwrap();
        batch_indices.push(batch_indices_i.r_int_convert());
    }
    let harmony_params = HarmonyParamsV2Gpu::from_r_list(harmony_params)?;
    let embd = r_matrix_to_faer_fp32(&pca);

    // run harmony v2
    let res = harmony_v2_gpu::<WgpuRuntime>(
        embd.as_ref(),
        &batch_indices,
        &harmony_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // clean up vram
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(faer_to_r_matrix(res.as_ref()))
}
