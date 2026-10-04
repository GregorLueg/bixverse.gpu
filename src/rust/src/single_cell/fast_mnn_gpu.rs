//! GPU-accelerated fastMNN.
//!
//! Mirrors `bixverse::rs_mnn` minus the PCA, which has to be in the object
//! already. Only the neighbour searches move to the device; centring, MNN
//! pairing and the tricube correction are shared with the CPU path inside
//! `bixverse-rs`.

use crate::ensure_gpu;
use bixverse_rs::gpu::sc_gpu::fast_mnn_gpu::{fast_mnn_gpu, FastMnnParamsGpu};
use bixverse_rs::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod fast_mnn_gpu;
    // functions
    fn rs_fast_mnn_gpu;
}

/////////////
// fastMNN //
/////////////

/// GPU: fastMNN batch correction
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_mnn`, implementing the fast mutual nearest
/// neighbour correction from Haghverdi, et al. The MNN searches and the
/// tricube neighbour search run on the WGPU backend; everything else is
/// shared with the CPU implementation.
///
/// @param embd Numerical matrix. The embedding to correct, usually PCA. Rows
/// represent cells.
/// @param batch_labels Integer vector. These represent to which batch a given
/// cell belongs. Needs to be 0-indexed!
/// @param fastmnn_params List. Parameter list, see [params_sc_fastmnn_gpu()].
/// @param seed Integer. Seed for reproducibility purposes.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @return The batch-corrected embedding, cells x dimensions, in the input
/// cell order.
///
/// @export
#[extendr]
fn rs_fast_mnn_gpu(
    embd: RMatrix<f64>,
    batch_labels: Vec<i32>,
    fastmnn_params: List,
    seed: usize,
    verbose: usize,
) -> extendr_api::Result<RArray<f64, 2>> {
    ensure_gpu()?;

    let fastmnn_params = FastMnnParamsGpu::from_r_list(fastmnn_params)?;
    let embd = r_matrix_to_faer_fp32(&embd);
    let batch_labels = batch_labels
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<usize>>();

    let device: WgpuDevice = Default::default();

    let res = fast_mnn_gpu::<WgpuRuntime>(
        embd.as_ref(),
        &batch_labels,
        &fastmnn_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(faer_to_r_matrix(res.as_ref()))
}
