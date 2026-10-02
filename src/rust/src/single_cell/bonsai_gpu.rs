//! Bonsai with Sanity on the GPU.
//!
//! Mirrors `bixverse::rs_sc_bonsai` and `bixverse::rs_mc_bonsai` argument for
//! argument. Only Sanity moves to
//! the device, gene chunk by gene chunk, with the same keep test as the CPU
//! path; the tree search and the layout stay on the CPU. The returned list is
//! the one `bixverse::new_bonsai_tree()` builds its class from.

use crate::ensure_plane_ops;
use crate::single_cell::scenic_gpu::cast_sparse_u32_f32;
use bixverse_rs::gpu::sc_gpu::sanity_bonsai_gpu::{sanity_bonsai_mc_gpu, sanity_bonsai_sc_gpu};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::bonsai::BonsaiScParams;
use bixverse_rs::single_cell::sc_r_wrappers::bonsai_sc_to_r_list;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod bonsai_gpu;
    // functions
    fn rs_sc_bonsai_gpu;
    fn rs_mc_bonsai_gpu;
}

////////////
// Bonsai //
////////////

/// GPU: Bonsai tree from single cell counts
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_sc_bonsai`. Sanity runs on the WGPU
/// backend, streamed over chunks of genes and keeping only the ones that pass
/// Bonsai's ingest filters. The tree search and the layout run on the CPU.
///
/// @param f_path_gene String. Path to the `counts_genes.bin` file.
/// @param f_path_cell String. Path to the `counts_cells.bin` file. Supplies the
/// library sizes.
/// @param cell_indices Integer. The cell indices to use. (0-indexed!) Sets the
/// leaf order.
/// @param gene_indices Integer. The candidate genes. (0-indexed!)
/// @param bonsai_params List. Parameter list, see
/// `bixverse::params_sc_bonsai()`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns The same list as `bixverse::rs_sc_bonsai()`: `parent` (0-indexed,
/// `-1` for the root), `branch`, `x`, `y`, `n_leaves`, `loglik`, `steps`,
/// `timings` and `genes_used` (0-indexed).
///
/// @export
///
/// @references de Groot, et al., Nat Biotechnol, 2026; Breda, et al., Nat
/// Biotechnol, 2021.
///
/// @keywords internal
#[extendr]
fn rs_sc_bonsai_gpu(
    f_path_gene: &str,
    f_path_cell: &str,
    cell_indices: Vec<i32>,
    gene_indices: Vec<i32>,
    bonsai_params: List,
    verbose: usize,
) -> Result<List> {
    ensure_plane_ops()?;

    let verbosity = parse_verbosity_level(verbose);
    let cell_indices = cell_indices.r_int_convert();
    let gene_indices = gene_indices.r_int_convert();
    let params = BonsaiScParams::from_r_list(bonsai_params)?;

    let gene_reader = ParallelSparseReader::new(f_path_gene).to_extendr()?;
    let cell_reader = ParallelSparseReader::new(f_path_cell).to_extendr()?;

    let device: WgpuDevice = Default::default();

    let res = sanity_bonsai_sc_gpu::<WgpuRuntime, _, _>(
        &gene_reader,
        &cell_reader,
        &cell_indices,
        &gene_indices,
        &params,
        device.clone(),
        verbosity,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(bonsai_sc_to_r_list(res))
}

/// GPU: Bonsai tree from metacell counts
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_mc_bonsai`. Sanity runs on the WGPU backend
/// over the metacells' aggregated raw counts, the tree search and the layout
/// on the CPU. Every metacell is a leaf.
///
/// @param sparse_data List. The raw metacell counts, see
/// `bixverse::mc_counts_to_list()` with `assay = "raw"`.
/// @param gene_indices Integer. The candidate genes. (0-indexed!)
/// @param bonsai_params List. Parameter list, see
/// `bixverse::params_sc_bonsai()`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns The same list as `bixverse::rs_mc_bonsai()`.
///
/// @export
///
/// @references de Groot, et al., Nat Biotechnol, 2026; Breda, et al., Nat
/// Biotechnol, 2021.
///
/// @keywords internal
#[extendr]
fn rs_mc_bonsai_gpu(
    sparse_data: List,
    gene_indices: Vec<i32>,
    bonsai_params: List,
    verbose: usize,
) -> Result<List> {
    ensure_plane_ops()?;

    let verbosity = parse_verbosity_level(verbose);
    let gene_indices = gene_indices.r_int_convert();
    let params = BonsaiScParams::from_r_list(bonsai_params)?;
    let sparse: CompressedSparseData2<f64, f64> =
        list_to_sparse_matrix(sparse_data, false).to_extendr()?;
    let counts = cast_sparse_u32_f32(sparse);

    let device: WgpuDevice = Default::default();

    let res = sanity_bonsai_mc_gpu::<WgpuRuntime>(
        &counts,
        &gene_indices,
        &params,
        device.clone(),
        verbosity,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(bonsai_sc_to_r_list(res))
}
