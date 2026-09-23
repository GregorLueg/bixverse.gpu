//! GPU-accelerated NEBULA.
//!
//! Mirrors `bixverse::rs_nebula_sc` argument for argument. Only stage two, the
//! per-gene penalised fits, moves to the device; cell ordering, batching, the
//! dispersion shrinkage and the Wald test are the CPU code in `bixverse-rs`.
//! Stage two runs in `f32` on the device, so the answers sit close to the CPU
//! path rather than on it.

use crate::ensure_gpu;
use crate::single_cell::sc_utils::nebula_res_to_r_list;
use bixverse_rs::gpu::sc_gpu::nebula_gpu::run_nebula_gpu;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::nebula::NebulaScParams;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod nebula_gpu;
    // functions
    fn rs_nebula_sc_gpu;
}

////////////
// NEBULA //
////////////

/// GPU: fit the NEBULA negative binomial gamma mixed model over single cells
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_nebula_sc`. Stage two of NEBULA, the
/// per-gene penalised fits, is dispatched to the WGPU backend in `f32` and
/// finished on the host in `f64`. Everything else, including the streaming of
/// the counts out of the gene-major store in batches, the subject ordering, the
/// dispersion shrinkage and the Wald test, is the CPU code. REML is not
/// implemented on the device and is rejected.
///
/// @param f_path_genes String. Path to the `counts_genes.bin` file.
/// @param f_path_cells String. Path to the `counts_cells.bin` file. Only read
/// when `offset` is `NULL`, to take the library sizes.
/// @param cells_to_keep Integer vector. 0-indexed(!) global positions of the
/// cells to analyse, in any order. Must not hold duplicates.
/// @param gene_indices Integer vector. 0-indexed(!) positions of the genes to
/// fit.
/// @param subject_ids Integer vector. 0-indexed(!) subject label per global
/// cell. One entry per cell in the store, not per cell in `cells_to_keep`.
/// @param design Numeric matrix. Predictors of cells x coefficients, rows
/// aligned to `cells_to_keep` and including an intercept.
/// @param offset Optional numeric vector. Strictly positive scaling factor per
/// selected cell, aligned to `cells_to_keep`. `NULL` uses the library sizes.
/// @param nebula_params Named list. The NEBULA parameters, see
/// [params_nebula_gpu()], plus either `coef` (a 0-indexed(!) coefficient) or
/// `contrast` (one weight per coefficient).
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with the following elements
/// \itemize{
///   \item gene_idx - Integer. 0-indexed positions of the genes that survived
///   NEBULA's own expression filter.
///   \item coefficients - Numeric matrix of genes x coefficients. The fixed
///   effects on the design scale.
///   \item se - Numeric matrix of genes x coefficients. The standard errors.
///   \item subject_overdispersion - Numeric. NEBULA's `sigma^2`.
///   \item cell_overdispersion - Numeric. NEBULA's `phi^-1`.
///   \item cell_overdispersion_shrunk - Numeric or `NULL`. The cell-level
///   overdispersion after empirical Bayes shrinkage, when it was requested.
///   \item convergence - Integer. NEBULA's convergence code. At or below `-20`
///   is a likely failure.
///   \item sigma_at_bound - Boolean. Whether the subject-level variance
///   finished pinned on its lower bound.
///   \item log_fc - Numeric. Effect of the tested coefficient or contrast, on
///   the natural log scale.
///   \item effect_se - Numeric. Standard error of that effect.
///   \item z - Numeric. The Wald statistic.
///   \item p_values - Numeric. Two-sided p-values.
///   \item fdr - Numeric. Benjamini-Hochberg adjusted p-values.
/// }
///
/// @references He, et al., Commun Biol, 2021
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nebula_sc_gpu(
    f_path_genes: String,
    f_path_cells: String,
    cells_to_keep: Vec<i32>,
    gene_indices: Vec<i32>,
    subject_ids: Vec<i32>,
    design: RMatrix<f64>,
    offset: Nullable<Vec<f64>>,
    nebula_params: List,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let cells_to_keep: Vec<usize> = cells_to_keep.r_int_convert();
    let gene_indices: Vec<usize> = gene_indices.r_int_convert();
    let subject_ids: Vec<usize> = subject_ids.r_int_convert();

    let n_coef = design.ncols();
    // Column-major out of R, row-major into `run_nebula_gpu`.
    let design = mat_to_flat_row_major(r_matrix_to_faer(&design));

    let params = NebulaScParams::from_r_list(nebula_params)?;

    let gene_reader = ParallelSparseReader::new(&f_path_genes).to_extendr()?;
    let cell_reader = ParallelSparseReader::new(&f_path_cells).to_extendr()?;

    let offset = match offset {
        Nullable::NotNull(o) => Some(o),
        Nullable::Null => None,
    };

    let device: WgpuDevice = Default::default();

    let res = run_nebula_gpu::<WgpuRuntime, _>(
        &gene_reader,
        &cell_reader,
        &cells_to_keep,
        &gene_indices,
        &subject_ids,
        &design,
        n_coef,
        offset.as_deref(),
        &params,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(nebula_res_to_r_list(res))
}
