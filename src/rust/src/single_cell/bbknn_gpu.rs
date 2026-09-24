//! GPU-accelerated BBKNN.
//!
//! Mirrors `bixverse::rs_bbknn`. Only the neighbour search moves to the device:
//! one GPU index per batch, cross-queried by every cell. The UMAP connectivity
//! pipeline that turns those neighbours into the two sparse matrices is shared
//! verbatim with the CPU path inside `bixverse-rs`, so the returned lists are
//! identical in shape and the R side can reuse `bixverse::rs_bbknn_filtering()`.

use crate::ensure_gpu;
use bixverse_rs::gpu::sc_gpu::bbknn_gpu::{bbknn_gpu, BbknnParamsGpu};
use bixverse_rs::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod bbknn_gpu;
    // functions
    fn rs_bbknn_gpu;
}

///////////
// BBKNN //
///////////

/// GPU: BBKNN batch correction
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_bbknn`, implementing the BBKNN algorithm
/// from Polański, et al. One nearest neighbour index is built per batch on the
/// WGPU backend and queried by every cell, so each cell gets
/// `neighbours_within_batch` neighbours from every batch. The UMAP
/// connectivity calculations that follow stay on the CPU and are shared with
/// the CPU implementation.
///
/// @param embd Numerical matrix. The embedding matrix used to generate the
/// BBKNN results. Usually PCA. Rows represent cells.
/// @param batch_labels Integer vector. These represent to which batch a given
/// cell belongs. Needs to be 0-indexed!
/// @param bbknn_params List. Parameter list, see [params_sc_bbknn_gpu()].
/// @param seed Integer. Seed for reproducibility purposes.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list of two lists representing the sparse matrix representation
/// of the distances and the connectivities. Each of them contains
/// \itemize{
///   \item data - The values of the sparse matrix.
///   \item indptr - The index pointers. 0-indexed.
///   \item indices - The column indices. 0-indexed.
///   \item nrow - Number of rows.
///   \item ncol - Number of columns.
///   \item cs_type - The sparse format, `"csr"` here.
/// }
///
/// @export
///
/// @references Polański, et al., Bioinformatics, 2020
///
/// @keywords internal
#[extendr]
fn rs_bbknn_gpu(
    embd: RMatrix<f64>,
    batch_labels: Vec<i32>,
    bbknn_params: List,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let bbknn_params = BbknnParamsGpu::from_r_list(bbknn_params)?;
    let embd = r_matrix_to_faer_fp32(&embd);
    let batch_labels = batch_labels
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<usize>>();

    let device: WgpuDevice = Default::default();

    let (distances, connectivities) = bbknn_gpu::<WgpuRuntime>(
        embd.as_ref(),
        &batch_labels,
        &bbknn_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(list!(
        distances = sparse_data_to_list(distances).to_extendr()?,
        connectivities = sparse_data_to_list(connectivities).to_extendr()?
    ))
}
