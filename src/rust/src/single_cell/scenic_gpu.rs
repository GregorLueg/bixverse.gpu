use bixverse_rs::gpu::sc_gpu::scenic_gpu::{
    run_scenic_grn_gpu, run_scenic_grn_in_memory_gpu, run_scenic_grn_streaming_gpu,
    ScenicGpuParams,
};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::scenic::ScenicParams;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod scenic_gpu;
    // functions
    fn rs_scenic_grn_gpu;
    fn rs_scenic_grn_streaming_gpu;
    fn rs_mc_scenic_gpu;
}

////////////
// Params //
////////////

/// Build the GPU-side SCENIC params from a single R-provided knob.
fn create_gpu_scenic_params(wave_byte_budget: f64) -> ScenicGpuParams {
    ScenicGpuParams {
        wave_byte_budget: wave_byte_budget as usize,
    }
}

/// Cast the R-native `<f64, f64>` sparse layout into `<u32, f32>` as required
/// by the GPU in-memory driver.
fn cast_sparse_u32_f32(
    sparse_mat: CompressedSparseData2<f64, f64>,
) -> CompressedSparseData2<u32, f32> {
    let data_cast: Vec<u32> = sparse_mat.data.iter().map(|&x| x as u32).collect();
    let data_2_cast = sparse_mat.data_2.map(|x| x.r_float_convert());

    match sparse_mat.cs_type {
        CompressedSparseFormat::Csc => CompressedSparseData2::new_csc(
            &data_cast,
            &sparse_mat.indices,
            &sparse_mat.indptr,
            data_2_cast.as_deref(),
            sparse_mat.shape,
        ),
        CompressedSparseFormat::Csr => CompressedSparseData2::new_csr(
            &data_cast,
            &sparse_mat.indices,
            &sparse_mat.indptr,
            data_2_cast.as_deref(),
            sparse_mat.shape,
        ),
    }
}

///////////////////
// SingleCells   //
///////////////////

/// GPU: SCENIC gene-regulatory network inference (disk-backed)
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_scenic_grn`. Reads the target genes from
/// disk in chunks and dispatches per-batch tree fits to the WGPU backend.
/// GBM is not supported on GPU; use the CPU version.
///
/// @param f_path_genes String. Path to the `counts_genes.bin` file.
/// @param cell_indices Integer vector. 0-indexed positions of cells to
/// include.
/// @param gene_indices Integer vector. 0-indexed positions of target genes.
/// @param tf_indices Integer vector. 0-indexed positions of TF predictors.
/// @param scenic_params Named list. See [bixverse::params_scenic()].
/// @param wave_byte_budget Double. VRAM ceiling for per-wave histogram +
/// cumulative tensors (bytes). Default 4 GiB.
/// @param seed Integer. Random seed.
/// @param verbose Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.
///
/// @returns A gene x TF importance matrix.
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_scenic_grn_gpu(
    f_path_genes: String,
    cell_indices: Vec<i32>,
    gene_indices: Vec<i32>,
    tf_indices: Vec<i32>,
    scenic_params: List,
    wave_byte_budget: f64,
    seed: usize,
    verbose: usize,
) -> Result<RArray<f64, 2>> {
    let cell_indices = cell_indices.r_int_convert();
    let gene_indices = gene_indices.r_int_convert();
    let tf_indices = tf_indices.r_int_convert();

    let scenic_params = ScenicParams::from_r_list(scenic_params)?;
    let gpu_params = create_gpu_scenic_params(wave_byte_budget);

    let device: WgpuDevice = Default::default();

    let grn_matrix = run_scenic_grn_gpu::<WgpuRuntime>(
        &f_path_genes,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &scenic_params,
        &gpu_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(faer_to_r_matrix(grn_matrix.as_ref()))
}

/// GPU: SCENIC GRN inference (streaming; bounded host memory)
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Streaming GPU equivalent of `bixverse::rs_scenic_grn_streaming`. Reads
/// targets in I/O chunks and dispatches each in-chunk batch serially to the
/// GPU. Peak host memory is bounded to one chunk of sparse columns.
/// GBM is not supported on GPU; use the CPU version.
///
/// @param f_path_genes String. Path to the `counts_genes.bin` file.
/// @param cell_indices Integer vector. 0-indexed positions of cells to
/// include.
/// @param gene_indices Integer vector. 0-indexed positions of target genes.
/// @param tf_indices Integer vector. 0-indexed positions of TF predictors.
/// @param scenic_params Named list. See [bixverse::params_scenic()].
/// @param wave_byte_budget Double. VRAM ceiling for per-wave histogram +
/// cumulative tensors (bytes). Default 4 GiB.
/// @param seed Integer. Random seed.
/// @param verbose Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.
///
/// @returns A gene x TF importance matrix.
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_scenic_grn_streaming_gpu(
    f_path_genes: String,
    cell_indices: Vec<i32>,
    gene_indices: Vec<i32>,
    tf_indices: Vec<i32>,
    scenic_params: List,
    wave_byte_budget: f64,
    seed: usize,
    verbose: usize,
) -> Result<RArray<f64, 2>> {
    let cell_indices = cell_indices.r_int_convert();
    let gene_indices = gene_indices.r_int_convert();
    let tf_indices = tf_indices.r_int_convert();

    let scenic_params = ScenicParams::from_r_list(scenic_params)?;
    let gpu_params = create_gpu_scenic_params(wave_byte_budget);

    let device: WgpuDevice = Default::default();

    let grn_matrix = run_scenic_grn_streaming_gpu::<WgpuRuntime>(
        &f_path_genes,
        &cell_indices,
        &gene_indices,
        &tf_indices,
        &scenic_params,
        &gpu_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(faer_to_r_matrix(grn_matrix.as_ref()))
}

/////////////////
// MetaCells   //
/////////////////

/// GPU: SCENIC GRN inference on MetaCells (in-memory sparse)
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_mc_scenic`. Assumes the sparse data is
/// pre-filtered for the genes / cells to include. Indices are 0-indexed.
/// GBM is not supported on GPU; use the CPU version.
///
/// @param sparse_data Named list with `data`, `indptr`, `indices`, `nrow`,
/// `ncol`, `format`.
/// @param tf_indices Integer vector. 0-indexed positions of TF predictors.
/// @param scenic_params Named list. See [bixverse::params_scenic()].
/// @param wave_byte_budget Double. VRAM ceiling for per-wave histogram +
/// cumulative tensors (bytes). Default 4 GiB.
/// @param seed Integer. Random seed.
/// @param verbose Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.
///
/// @returns A gene x TF importance matrix.
///
/// @export
///
/// @keywords internal
#[extendr]
fn rs_mc_scenic_gpu(
    sparse_data: List,
    tf_indices: Vec<i32>,
    scenic_params: List,
    wave_byte_budget: f64,
    seed: usize,
    verbose: usize,
) -> Result<RArray<f64, 2>> {
    let tf_indices = tf_indices.r_int_convert();
    let sparse: CompressedSparseData2<f64, f64> =
        list_to_sparse_matrix(sparse_data, true).to_extendr()?;
    let sparse = cast_sparse_u32_f32(sparse);
    let scenic_params = ScenicParams::from_r_list(scenic_params)?;
    let gpu_params = create_gpu_scenic_params(wave_byte_budget);

    let device: WgpuDevice = Default::default();

    let grn_matrix = run_scenic_grn_in_memory_gpu::<WgpuRuntime, u32>(
        &sparse,
        &tf_indices,
        &scenic_params,
        &gpu_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(faer_to_r_matrix(grn_matrix.as_ref()))
}
