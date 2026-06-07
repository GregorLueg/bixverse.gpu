use bixverse_rs::gpu::sc_gpu::pca_gpu::pca_on_sc_sparse_gpu;
use bixverse_rs::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod pca_gpu;
    // functions
    fn rs_sc_pca_sparse_gpu;
}

////////////////////////////////////////////
// GPU-accelerated sparse, randomised SVD //
////////////////////////////////////////////

/// Calculates sparse PCA for single cell
///
/// @description
/// Helper function that will calculate sparse PCA without scaling the data.
/// This has the advantage that you avoid creating a large dense matrix due
/// to scaling; however, it has the disadvantage that the first PC will be
/// heavily influenced by average expression. If random_svd is set to `FALSE`,
/// Lanczos iterations will be used to solve the SVD; if random_svd is set
/// to `TRUE`, the randomised version will be used with multiplication of the
/// initial sparse matrix with a much smaller random dense matrix, avoiding
/// holding a large dense matrix in memory.
///
/// @param f_path_gene String. Path to the `counts_genes.bin` file.
/// @param no_pcs Integer. Number of PCs to calculate.
/// @param random_svd Boolean. Shall randomised SVD be used.
/// @param cell_indices Integer. The cell indices to use. (0-indexed!)
/// @param gene_indices Integer. The gene indices to use. (0-indexed!)
/// @param seed Integer. Random seed for the randomised SVD.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with with the following items
/// \itemize{
///   \item scores - The samples projected on the PCA space (solved via sparse
///   SVD).
///   \item loadings - The loadings of the features for the PCA (solved via
///   sparse SVD).
///   \item singular_values - The singular values for the PCA (solved via sparse
///   SVD).
/// }
///
/// @export
#[extendr]
fn rs_sc_pca_sparse_gpu(
    f_path_gene: &str,
    no_pcs: usize,
    cell_indices: Vec<i32>,
    gene_indices: Vec<i32>,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    let device: WgpuDevice = Default::default();

    let cell_set = cell_indices.r_int_convert();
    let gene_indices = gene_indices.r_int_convert();

    let res = pca_on_sc_sparse_gpu::<WgpuRuntime>(
        f_path_gene,
        &cell_set,
        &gene_indices,
        no_pcs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(list!(
        scores = faer_to_r_matrix(res.0.as_ref()),
        loadings = faer_to_r_matrix(res.1.as_ref()),
        singular_values = res.2.r_float_convert()
    ))
}
