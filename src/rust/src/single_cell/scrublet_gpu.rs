//! GPU-accelerated Scrublet.
//!
//! Mirrors `bixverse::rs_sc_scrublet` step for step. Three stages move to the
//! device: the randomised sparse SVD of the observed cells, the SpMM
//! projection of the simulated doublets into that PC space, and (by default)
//! the kNN over the combined embedding. HVG selection, doublet simulation, the
//! kNN classifier and the Otsu threshold stay on the CPU and are reused
//! unchanged from the CPU module, so `FinalScrubletRes` and the R list built
//! from it are identical on both paths.

use bixverse_rs::gpu::sc_gpu::scrublet_gpu::{run_scrublet_gpu, ScrubletParamsGpu};
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_processing::scrublet::FinalScrubletRes;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod scrublet_gpu;
    // functions
    fn rs_sc_scrublet_gpu;
}

//////////////
// Scrublet //
//////////////

/// GPU: Scrublet doublet detection
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_sc_scrublet`. The PCA of the observed
/// cells, the projection of the simulated doublets and the kNN over the
/// combined embedding run on the WGPU backend. HVG selection, doublet
/// simulation, scoring and the Otsu threshold stay on the CPU. Which nearest
/// neighbour index runs is decided by the `knn_backend` element of
/// `scrublet_params`.
///
/// @param f_path_gene String. Path to the `counts_genes.bin` file.
/// @param f_path_cell String. Path to the `counts_cells.bin` file.
/// @param cells_to_keep Integer vector. The indices (0-indexed!) of the cells
/// to include in this analysis.
/// @param scrublet_params List. Parameter list, see
/// [params_scrublet_gpu()].
/// @param seed Integer. Seed for reproducibility purposes.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
/// @param streaming Boolean. Shall the data be streamed for the HVG
/// calculations.
/// @param return_combined_pca Boolean. Shall the generated PCA be returned.
/// @param return_pairs Boolean. Shall the parents of the simulated cells be
/// returned.
///
/// @returns A list with
/// \itemize{
///   \item predicted_doublets - Boolean vector indicating which observed cells
///   were predicted as doublets (TRUE = doublet, FALSE = singlet).
///   \item doublet_scores_obs - Numerical vector with the likelihood of being
///   a doublet for the observed cells.
///   \item doublet_scores_sim - Numerical vector with the likelihood of being
///   a doublet for the simulated cells.
///   \item doublet_errors_obs - Numerical vector with the standard errors of
///   the scores for the observed cells.
///   \item z_scores - Z-scores for the observed cells. Represents:
///   `score - threshold / error`.
///   \item threshold - Used threshold.
///   \item detected_doublet_rate - Fraction of cells that are called as
///   doublet.
///   \item detectable_doublet_fraction - Fraction of simulated doublets with
///   scores above the threshold.
///   \item overall_doublet_rate - Estimated overall doublet rate.
///   \item pca - Optional PCA embeddings across the original cells and
///   simulated doublets.
///   \item pair_1 - Optional index of the parent cell 1 of the simulated
///   doublets.
///   \item pair_2 - Optional index of the parent cell 2 of the simulated
///   doublets.
/// }
///
/// @export
///
/// @references Wolock, et al., Cell Syst, 2020
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_sc_scrublet_gpu(
    f_path_gene: &str,
    f_path_cell: &str,
    cells_to_keep: Vec<i32>,
    scrublet_params: List,
    seed: usize,
    verbose: usize,
    streaming: bool,
    return_combined_pca: bool,
    return_pairs: bool,
) -> Result<List> {
    let scrublet_params = ScrubletParamsGpu::from_r_list(scrublet_params)?;
    let cells_to_keep = cells_to_keep.r_int_convert();

    let gene_reader = ParallelSparseReader::new(f_path_gene).to_extendr()?;
    let cell_reader = ParallelSparseReader::new(f_path_cell).to_extendr()?;

    let device: WgpuDevice = Default::default();

    let (scrublet_res, pca, pair_1, pair_2): FinalScrubletRes = run_scrublet_gpu::<WgpuRuntime, _>(
        &gene_reader,
        &cell_reader,
        &scrublet_params,
        &cells_to_keep,
        streaming,
        seed,
        device.clone(),
        verbose,
        return_combined_pca,
        return_pairs,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let pca_out = pca.map(|m| faer_to_r_matrix(m.as_ref()));
    let pair_1_out: Robj = match pair_1 {
        Some(p) => p.r_int_convert().into(),
        None => NULL.into(),
    };
    let pair_2_out: Robj = match pair_2 {
        Some(p) => p.r_int_convert().into(),
        None => NULL.into(),
    };

    Ok(list!(
        predicted_doublets = scrublet_res.predicted_doublets,
        doublet_scores_obs = scrublet_res.doublet_scores_obs.r_float_convert(),
        doublet_scores_sim = scrublet_res.doublet_scores_sim.r_float_convert(),
        doublet_errors_obs = scrublet_res.doublet_errors_obs.r_float_convert(),
        z_scores = scrublet_res.z_scores.r_float_convert(),
        threshold = scrublet_res.threshold as f64,
        detected_doublet_rate = scrublet_res.detected_doublet_rate as f64,
        detectable_doublet_fraction = scrublet_res.detectable_doublet_fraction as f64,
        overall_doublet_rate = scrublet_res.overall_doublet_rate as f64,
        pca = pca_out,
        pair_1 = pair_1_out,
        pair_2 = pair_2_out
    ))
}
