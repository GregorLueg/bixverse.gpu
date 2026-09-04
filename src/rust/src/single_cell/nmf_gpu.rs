//! GPU-accelerated NMF for single cells and meta cells.
//!
//! Thin bindings over the sparse GPU entry points in
//! [`bixverse_rs::gpu::methods_gpu::nmf_consensus_gpu`]. Those own the device
//! client, the pre-processing, the upload and the scratch sizing, so there is
//! nothing to do here beyond getting the counts into a
//! [`CompressedSparseData2`] and converting the result back to an R list.
//!
//! Only the HALS solver runs on the device. The consensus machinery (pooling,
//! the local-density filter, the k-means, the silhouette and the per-coordinate
//! median) stays on the host, shared verbatim with the CPU path.
//!
//! The two data sources differ only in where the matrix comes from: single
//! cells stream from the gene-based binary store, meta cells arrive as an
//! in-memory CSR list from R. Everything downstream is identical.
//!
//! Everything here is `f32`. The GPU kernels are `f32`-only, which matches the
//! CPU single-cell and meta-cell paths, so the result converters are concrete
//! rather than generic over the float.

use crate::ensure_gpu;
use bixverse_rs::gpu::methods_gpu::nmf_consensus_gpu::{
    nmf_consensus_run_sparse_gpu, nmf_k_sweep_run_sparse_gpu, nmf_multiple_run_sparse_gpu,
    nmf_single_run_sparse_gpu,
};
use bixverse_rs::methods::nmf_hals::consensus::{ConsensusNmfResult, ConsensusParams, KSweepEntry};
use bixverse_rs::methods::nmf_hals::{HalsOpts, NmfResult, StabilisedNmfResult};
use bixverse_rs::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::prelude::Rint;
use extendr_api::*;
use indexmap::IndexSet;
use std::time::Instant;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod nmf_gpu;
    // functions
    fn rs_nmf_single_sc_gpu;
    fn rs_nmf_multi_sc_gpu;
    fn rs_nmf_consensus_sc_gpu;
    fn rs_nmf_k_sweep_sc_gpu;
    fn rs_nmf_single_mc_gpu;
    fn rs_nmf_multi_mc_gpu;
    fn rs_nmf_consensus_mc_gpu;
    fn rs_nmf_k_sweep_mc_gpu;
}

/////////////
// Helpers //
/////////////

/// Load a cells x genes sparse matrix from the gene-based binary store.
///
/// Mirrors the private loader behind the CPU `nmf_*_run_sc` entry points. The
/// GPU entry points take an owned [`CompressedSparseData2`] rather than a
/// reader, so the read has to happen here.
///
/// Only the requested layer is populated: `use_second_layer` selects the
/// normalised counts, which is also what the solver reads downstream. Asking
/// for one layer and solving on the other would hand the kernels an empty
/// buffer, so the two decisions are made from the same flag.
///
/// ### Params
///
/// * `f_path_gene` - Path to the `counts_genes.bin` file.
/// * `gene_indices` - 0-indexed positions of the genes to include.
/// * `cell_indices` - 0-indexed positions of the cells to include.
/// * `use_second_layer` - Load the normalised counts instead of the raw ones.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The loaded CSC matrix, cells x genes.
fn load_sc_sparse(
    f_path_gene: &str,
    gene_indices: &[usize],
    cell_indices: &[usize],
    use_second_layer: bool,
    verbose: usize,
) -> Result<CompressedSparseData2<f32>> {
    let verbosity = parse_verbosity_level(verbose);
    let start = Instant::now();
    if verbosity.normal_verbosity() {
        println!("NMF (GPU): Loading data from disk ...")
    }

    let layer = if use_second_layer {
        DataLayerReturn::Norm
    } else {
        DataLayerReturn::Raw
    };
    let cell_set: IndexSet<u32> = cell_indices.iter().map(|&x| x as u32).collect();

    let reader = ParallelSparseReader::new(f_path_gene).to_extendr()?;
    let gene_chunks: Vec<CscGeneChunk> = reader
        .read_gene_parallel_filtered(gene_indices, &cell_set)
        .to_extendr()?;
    let csc: CompressedSparseData2<f32> =
        from_gene_chunks::<f32>(gene_chunks, &layer, cell_indices.len()).to_extendr()?;

    if verbosity.normal_verbosity() {
        println!(" ... done in {:.2?}", start.elapsed())
    }

    Ok(csc)
}

/// Cast the R-native `<f64, f64>` sparse layout into the `f32` the GPU wants.
///
/// ### Params
///
/// * `sparse_mat` - The sparse matrix as it came across from R.
///
/// ### Returns
///
/// The same matrix, same layout, narrowed to `f32`.
fn cast_sparse_f32(sparse_mat: CompressedSparseData2<f64, f64>) -> CompressedSparseData2<f32> {
    CompressedSparseData2::from_parts(
        sparse_mat.data.r_float_convert(),
        sparse_mat.indices,
        sparse_mat.indptr,
        sparse_mat.data_2.map(|x| x.r_float_convert()),
        sparse_mat.cs_type,
        sparse_mat.shape,
    )
}

/// Release the device buffers once a solve is done.
///
/// The scratch for a k sweep is sized at the largest rank in the range, so
/// leaving it around between calls is worth avoiding.
///
/// ### Params
///
/// * `device` - The device that ran the solve.
fn release_device(device: &WgpuDevice) {
    let client = WgpuRuntime::client(device);
    client.memory_cleanup();
}

/// Convert a single-run fit into an R list.
///
/// ### Params
///
/// * `res` - The fit to convert.
///
/// ### Returns
///
/// A list with `w`, `h`, `final_loss`, `n_iter` and `converged`. `final_loss`
/// is absolute, unlike the relative errors the consensus bindings return.
fn nmf_res_to_r_list(res: &NmfResult<f32>) -> List {
    list!(
        w = faer_to_r_matrix(res.w.as_ref()),
        h = faer_to_r_matrix(res.h.as_ref()),
        final_loss = res.final_loss as f64,
        n_iter = res.n_iter,
        converged = res.converged
    )
}

/// Convert a stabilised (multi-restart) fit into an R list.
///
/// ### Params
///
/// * `res` - The fit to convert.
///
/// ### Returns
///
/// A list with `w_all`, `h_per_run`, `losses`, `converged` and `best_idx`.
/// `best_idx` is shifted to 1-based on the way out.
fn stabilised_res_to_r_list(res: &StabilisedNmfResult<f32>) -> List {
    let h_per_run: List = res
        .h_per_run
        .iter()
        .map(|h| faer_to_r_matrix(h.as_ref()))
        .collect();

    list!(
        w_all = faer_to_r_matrix(res.w_all.as_ref()),
        h_per_run = h_per_run,
        losses = res.losses.as_slice().r_float_convert(),
        converged = res.converged.clone(),
        best_idx = (res.best_idx + 1) as i32
    )
}

/// Convert a consensus fit into a flat R list.
///
/// Port of the converter behind the CPU bindings, so both packages hand the
/// same shape to the shared `bixverse::new_consensus_nmf_result()`.
///
/// Cluster labels and the pooled indices of the survivors are shifted to
/// 1-based on the way out. A dropped component has no label, which becomes
/// `NA_integer_` rather than a sentinel the R side would have to know about.
///
/// ### Params
///
/// * `res` - The fit to convert.
///
/// ### Returns
///
/// A list with `w`, `h`, `rel_error`, `rel_run_errors`, `labels`,
/// `local_density`, `kept`, `silhouette`, `stability`, `cluster_sizes`,
/// `n_dropped` and `n_empty_clusters`. `rel_error` and `rel_run_errors` are
/// relative to the squared Frobenius norm of the input.
fn consensus_res_to_r_list(res: &ConsensusNmfResult<f32>) -> List {
    let labels: Vec<Rint> = res
        .clusters
        .labels
        .iter()
        .map(|label| match label {
            Some(cluster) => Rint::from((*cluster + 1) as i32),
            None => Rint::na(),
        })
        .collect();

    let kept: Vec<i32> = res
        .clusters
        .kept
        .iter()
        .map(|&idx| (idx + 1) as i32)
        .collect();

    let cluster_sizes: Vec<i32> = res.clusters.sizes.iter().map(|&s| s as i32).collect();

    list!(
        w = faer_to_r_matrix(res.w.as_ref()),
        h = faer_to_r_matrix(res.h.as_ref()),
        rel_error = res.error as f64,
        rel_run_errors = res.run_errors.as_slice().r_float_convert(),
        labels = labels,
        local_density = res.clusters.local_density.as_slice().r_float_convert(),
        kept = kept,
        silhouette = res.clusters.silhouette.as_slice().r_float_convert(),
        stability = res.clusters.stability as f64,
        cluster_sizes = cluster_sizes,
        n_dropped = res.clusters.n_dropped as i32,
        n_empty_clusters = res.clusters.n_empty_clusters as i32
    )
}

/// Convert a k sweep into a columnar R list.
///
/// One element per field rather than one per k, so the R side can hand the
/// whole thing to `data.table::as.data.table()` without a transpose.
///
/// ### Params
///
/// * `entries` - One entry per swept k, in the order they were requested.
///
/// ### Returns
///
/// A list with `k`, `stability`, `best_error`, `median_error`,
/// `consensus_failed`, `n_dropped`, `n_empty_clusters` and `n_converged`, each
/// a vector as long as `entries`. `stability` is `NaN` where the consensus step
/// failed; the R side turns that into `NA`.
fn k_sweep_to_r_list(entries: &[KSweepEntry<f32>]) -> List {
    let k: Vec<i32> = entries.iter().map(|e| e.k as i32).collect();
    let stability: Vec<f64> = entries.iter().map(|e| e.stability as f64).collect();
    let best_error: Vec<f64> = entries.iter().map(|e| e.best_error as f64).collect();
    let median_error: Vec<f64> = entries.iter().map(|e| e.median_error as f64).collect();
    let consensus_failed: Vec<bool> = entries.iter().map(|e| e.consensus_failed).collect();
    let n_dropped: Vec<i32> = entries.iter().map(|e| e.n_dropped as i32).collect();
    let n_empty_clusters: Vec<i32> = entries.iter().map(|e| e.n_empty_clusters as i32).collect();
    let n_converged: Vec<i32> = entries.iter().map(|e| e.n_converged as i32).collect();

    list!(
        k = k,
        stability = stability,
        best_error = best_error,
        median_error = median_error,
        consensus_failed = consensus_failed,
        n_dropped = n_dropped,
        n_empty_clusters = n_empty_clusters,
        n_converged = n_converged
    )
}

//////////////////
// Single cells //
//////////////////

/// Run NMF (HALS) on the GPU over a set of single cells and genes
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_single_sc()]. The counts are loaded
/// from the binary store once, uploaded to the device, and the whole HALS loop
/// runs there. The NNDSVD initialisation stays on the host.
///
/// @param f_path_gene Path to the `counts_genes.bin` file.
/// @param gene_indices Integer vector. 0-indexed(!) positions of the genes
/// to include.
/// @param cell_indices Integer vector. 0-indexed(!) positions of cells to
/// include in the analysis.
/// @param k Integer. Number of latent factors to return. At most 128, the GPU
/// solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`. Takes the
/// data as is, or scales by standard deviation or squared standard deviation
/// per feature.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on the normalised
/// counts; if `FALSE`, on the raw counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters.
/// @param seed Integer. Random seed for initialisation.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with the following items
/// \itemize{
///   \item w - The left factor matrix (n_cells x k)
///   \item h - The right factor matrix (k x n_genes)
///   \item final_loss - Loss at the final iteration
///   \item n_iter - Number of iterations the algorithm run for
///   \item converged - Did the NMF algorithm converge
/// }
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_single_sc_gpu(
    f_path_gene: &str,
    gene_indices: &[i32],
    cell_indices: &[i32],
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let gene_indices = gene_indices.r_int_convert();
    let cell_indices = cell_indices.r_int_convert();
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;

    let csc = load_sc_sparse(
        f_path_gene,
        &gene_indices,
        &cell_indices,
        use_second_layer,
        verbose,
    )?;

    let device: WgpuDevice = Default::default();
    let nmf_res = nmf_single_run_sparse_gpu::<WgpuRuntime>(
        csc,
        k,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(nmf_res_to_r_list(&nmf_res))
}

/// Run multiple NMF (HALS) restarts on the GPU over single cells and genes
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_multi_sc()]. Runs `n_runs` HALS NMF
/// with random initialisations seeded by `seed + i`. The `nmf_init` field in
/// `nmf_hals_params` is ignored; random init is always used. The matrix is
/// uploaded once and reused across every restart, but the restarts themselves
/// run one after the other on the single device.
///
/// @param f_path_gene Path to the `counts_genes.bin` file.
/// @param gene_indices Integer vector. 0-indexed(!) positions of the genes
/// to include.
/// @param cell_indices Integer vector. 0-indexed(!) positions of cells to
/// include in the analysis.
/// @param k Integer. Number of latent factors per run. At most 128, the GPU
/// solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on the normalised
/// counts; if `FALSE`, on the raw counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters.
/// @param n_runs Integer. Number of random restarts.
/// @param seed Integer. Base random seed. Run `i` uses `seed + i`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with the following items
/// \itemize{
///   \item w_all - Column-bound W matrices across all runs,
///   shape `n_cells x (k * n_runs)`. Columns `i*k+1..(i+1)*k` are run `i`'s
///   components (1-indexed).
///   \item h_per_run - List of H matrices, each `k x n_genes`.
///   \item losses - Numeric vector. Final reconstruction loss per run.
///   \item converged - Logical vector. Convergence flag per run.
///   \item best_idx - Integer. 1-indexed position of the run with the lowest
///   final loss.
/// }
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_multi_sc_gpu(
    f_path_gene: &str,
    gene_indices: &[i32],
    cell_indices: &[i32],
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let gene_indices = gene_indices.r_int_convert();
    let cell_indices = cell_indices.r_int_convert();
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;

    let csc = load_sc_sparse(
        f_path_gene,
        &gene_indices,
        &cell_indices,
        use_second_layer,
        verbose,
    )?;

    let device: WgpuDevice = Default::default();
    let nmf_res = nmf_multiple_run_sparse_gpu::<WgpuRuntime>(
        csc,
        k,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        n_runs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(stabilised_res_to_r_list(&nmf_res))
}

/// Run consensus NMF on the GPU over a set of single cells and genes
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_consensus_sc()]. Runs `n_runs` HALS
/// restarts on the device, then pools their components, drops unstable ones by
/// local density, k-means clusters the survivors and refits the partner factor
/// against the per-cluster median. Everything after the restarts runs on the
/// host, shared with the CPU implementation.
///
/// The restart factors are dense and all held at once, so `n_runs` times `k`
/// times the cell count is the memory to budget for.
///
/// @param f_path_gene Path to the `counts_genes.bin` file.
/// @param gene_indices Integer vector. 0-indexed(!) positions of the genes
/// to include.
/// @param cell_indices Integer vector. 0-indexed(!) positions of cells to
/// include in the analysis.
/// @param k Integer. Number of latent factors. Must be at least 2 and at most
/// 128, the GPU solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on the normalised
/// counts; if `FALSE`, on the raw counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters. The
/// `nmf_init` field is ignored, restarts always use random initialisation.
/// @param nmf_consensus_params Named list. Contains the consensus parameters.
/// @param n_runs Integer. Number of restarts. Must be at least 2.
/// @param seed Integer. Base random seed. Restart `i` uses `seed + i`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with the following items
/// \itemize{
///   \item w - The left factor matrix (n_cells x k)
///   \item h - The right factor matrix (k x n_genes)
///   \item rel_error - Reconstruction error relative to the squared Frobenius
///   norm of the input. Not comparable with the absolute `final_loss` the
///   single-run version returns.
///   \item rel_run_errors - The same, per restart.
///   \item labels - Integer vector of length `k * n_runs`. Cluster each pooled
///   component landed in, `NA` if it was dropped.
///   \item local_density - Mean cosine distance to the nearest neighbours per
///   pooled component.
///   \item kept - 1-indexed positions of the surviving pooled components.
///   \item silhouette - Silhouette per survivor, aligned with `kept`.
///   \item stability - Mean silhouette over the survivors.
///   \item cluster_sizes - Number of survivors per cluster.
///   \item n_dropped - Number of pooled components removed.
///   \item n_empty_clusters - Number of clusters left with no members.
/// }
///
/// @references Kotliar et al., eLife, 2019
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_consensus_sc_gpu(
    f_path_gene: &str,
    gene_indices: &[i32],
    cell_indices: &[i32],
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    nmf_consensus_params: List,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let gene_indices = gene_indices.r_int_convert();
    let cell_indices = cell_indices.r_int_convert();
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;
    let consensus_opt: ConsensusParams<f32> = ConsensusParams::from_r_list(nmf_consensus_params)?;

    let csc = load_sc_sparse(
        f_path_gene,
        &gene_indices,
        &cell_indices,
        use_second_layer,
        verbose,
    )?;

    let device: WgpuDevice = Default::default();
    let nmf_res = nmf_consensus_run_sparse_gpu::<WgpuRuntime>(
        csc,
        k,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        Some(consensus_opt),
        n_runs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(consensus_res_to_r_list(&nmf_res))
}

/// Sweep k on the GPU and report consensus stability against error
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_k_sweep_sc()]. Returns diagnostics
/// only, no factors, so a wide `k_range` stays cheap in memory. This is the
/// shape the GPU path is really for: the counts are uploaded once and serve
/// every one of the `length(k_range) * n_runs` solves. Pick the k where
/// stability is high and the error curve has not yet flattened, then call
/// [rs_nmf_consensus_sc_gpu()] there.
///
/// @param f_path_gene Path to the `counts_genes.bin` file.
/// @param gene_indices Integer vector. 0-indexed(!) positions of the genes
/// to include.
/// @param cell_indices Integer vector. 0-indexed(!) positions of cells to
/// include in the analysis.
/// @param k_range Integer vector. Ranks to evaluate, every entry at least 2 and
/// at most 128, the GPU solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on the normalised
/// counts; if `FALSE`, on the raw counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters.
/// @param nmf_consensus_params Named list. Contains the consensus parameters.
/// @param n_runs Integer. Number of restarts per k. Must be at least 2.
/// @param seed Integer. Base random seed.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list of equal-length vectors, one element per swept k
/// \itemize{
///   \item k - The rank.
///   \item stability - Mean silhouette of the consensus clusters. `NaN` where
///   the consensus step failed.
///   \item best_error - Lowest restart error, relative to the squared
///   Frobenius norm of the input.
///   \item median_error - Median restart error, same scale.
///   \item consensus_failed - Did the density filter leave fewer than `k`
///   components.
///   \item n_dropped - Number of pooled components removed.
///   \item n_empty_clusters - Number of clusters left with no members.
///   \item n_converged - Restarts that met the HALS tolerance.
/// }
///
/// @references Kotliar et al., eLife, 2019
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_k_sweep_sc_gpu(
    f_path_gene: &str,
    gene_indices: &[i32],
    cell_indices: &[i32],
    k_range: &[i32],
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    nmf_consensus_params: List,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let gene_indices = gene_indices.r_int_convert();
    let cell_indices = cell_indices.r_int_convert();
    let k_range = k_range.r_int_convert();
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;
    let consensus_opt: ConsensusParams<f32> = ConsensusParams::from_r_list(nmf_consensus_params)?;

    let csc = load_sc_sparse(
        f_path_gene,
        &gene_indices,
        &cell_indices,
        use_second_layer,
        verbose,
    )?;

    let device: WgpuDevice = Default::default();
    let sweep_res = nmf_k_sweep_run_sparse_gpu::<WgpuRuntime>(
        csc,
        &k_range,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        Some(consensus_opt),
        n_runs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(k_sweep_to_r_list(&sweep_res))
}

////////////////
// Meta cells //
////////////////

/// Run NMF (HALS) on the GPU over meta cells
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_single_mc()]. Assumes that the sparse
/// data is pre-filtered for the cells/genes you wish to include. Indices in the
/// sparse data need to be 0-indexed.
///
/// @param sparse_data A named list with `data`, `indptr`, `indices`, `nrow`,
/// `ncol` and `cs_type`.
/// @param k Integer. Number of latent factors to return. At most 128, the GPU
/// solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on normalised counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters.
/// @param seed Integer. Random seed for initialisation.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with `w`, `h`, `final_loss`, `n_iter`, `converged`.
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_single_mc_gpu(
    sparse_data: List,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let sparse: CompressedSparseData2<f64, f64> =
        list_to_sparse_matrix(sparse_data, true).to_extendr()?;
    let sparse = cast_sparse_f32(sparse);
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;

    let device: WgpuDevice = Default::default();
    let nmf_res = nmf_single_run_sparse_gpu::<WgpuRuntime>(
        sparse,
        k,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(nmf_res_to_r_list(&nmf_res))
}

/// Run multiple NMF (HALS) restarts on the GPU over meta cells
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_multi_mc()]. Assumes that the sparse
/// data is pre-filtered for the cells/genes you wish to include. Indices in the
/// sparse data need to be 0-indexed.
///
/// @param sparse_data A named list with `data`, `indptr`, `indices`, `nrow`,
/// `ncol` and `cs_type`.
/// @param k Integer. Number of latent factors per run. At most 128, the GPU
/// solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on normalised counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters.
/// @param n_runs Integer. Number of random restarts.
/// @param seed Integer. Base random seed. Run `i` uses `seed + i`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with `w_all`, `h_per_run`, `losses`, `converged`,
/// `best_idx` (1-indexed).
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_multi_mc_gpu(
    sparse_data: List,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let sparse: CompressedSparseData2<f64, f64> =
        list_to_sparse_matrix(sparse_data, true).to_extendr()?;
    let sparse = cast_sparse_f32(sparse);
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;

    let device: WgpuDevice = Default::default();
    let nmf_res = nmf_multiple_run_sparse_gpu::<WgpuRuntime>(
        sparse,
        k,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        n_runs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(stabilised_res_to_r_list(&nmf_res))
}

/// Run consensus NMF on the GPU over meta cells
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_consensus_mc()]. Assumes that the
/// sparse data is pre-filtered for the cells/genes you wish to include. Indices
/// in the sparse data need to be 0-indexed.
///
/// @param sparse_data A named list with `data`, `indptr`, `indices`, `nrow`,
/// `ncol` and `cs_type`.
/// @param k Integer. Number of latent factors. Must be at least 2 and at most
/// 128, the GPU solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on normalised counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters. The
/// `nmf_init` field is ignored, restarts always use random initialisation.
/// @param nmf_consensus_params Named list. Contains the consensus parameters.
/// @param n_runs Integer. Number of restarts. Must be at least 2.
/// @param seed Integer. Base random seed. Restart `i` uses `seed + i`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with `w`, `h`, `rel_error`, `rel_run_errors`, `labels`,
/// `local_density`, `kept`, `silhouette`, `stability`, `cluster_sizes`,
/// `n_dropped` and `n_empty_clusters`. The errors are relative to the squared
/// Frobenius norm of the input.
///
/// @references Kotliar et al., eLife, 2019
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_consensus_mc_gpu(
    sparse_data: List,
    k: usize,
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    nmf_consensus_params: List,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let sparse: CompressedSparseData2<f64, f64> =
        list_to_sparse_matrix(sparse_data, true).to_extendr()?;
    let sparse = cast_sparse_f32(sparse);
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;
    let consensus_opt: ConsensusParams<f32> = ConsensusParams::from_r_list(nmf_consensus_params)?;

    let device: WgpuDevice = Default::default();
    let nmf_res = nmf_consensus_run_sparse_gpu::<WgpuRuntime>(
        sparse,
        k,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        Some(consensus_opt),
        n_runs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(consensus_res_to_r_list(&nmf_res))
}

/// Sweep k on the GPU over meta cells
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU counterpart of [bixverse::rs_nmf_k_sweep_mc()]. Returns diagnostics
/// only, no factors. The matrix is uploaded once and serves every one of the
/// `length(k_range) * n_runs` solves, which is where the GPU path pays off.
///
/// @param sparse_data A named list with `data`, `indptr`, `indices`, `nrow`,
/// `ncol` and `cs_type`.
/// @param k_range Integer vector. Ranks to evaluate, every entry at least 2 and
/// at most 128, the GPU solver's rank cap.
/// @param preprocessing String. One of `c("none", "sd", "sqrt_sd")`.
/// @param use_second_layer Boolean. If `TRUE`, runs NMF on normalised counts.
/// @param nmf_hals_params Named list. Contains the NMF parameters.
/// @param nmf_consensus_params Named list. Contains the consensus parameters.
/// @param n_runs Integer. Number of restarts per k. Must be at least 2.
/// @param seed Integer. Base random seed.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list of equal-length vectors, one element per swept k: `k`,
/// `stability`, `best_error`, `median_error`, `consensus_failed`, `n_dropped`,
/// `n_empty_clusters` and `n_converged`.
///
/// @references Kotliar et al., eLife, 2019
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_nmf_k_sweep_mc_gpu(
    sparse_data: List,
    k_range: &[i32],
    preprocessing: &str,
    use_second_layer: bool,
    nmf_hals_params: List,
    nmf_consensus_params: List,
    n_runs: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let sparse: CompressedSparseData2<f64, f64> =
        list_to_sparse_matrix(sparse_data, true).to_extendr()?;
    let sparse = cast_sparse_f32(sparse);
    let k_range = k_range.r_int_convert();
    let nmf_hals_opt: HalsOpts<f32> = HalsOpts::from_r_list(nmf_hals_params, seed).to_extendr()?;
    let consensus_opt: ConsensusParams<f32> = ConsensusParams::from_r_list(nmf_consensus_params)?;

    let device: WgpuDevice = Default::default();
    let sweep_res = nmf_k_sweep_run_sparse_gpu::<WgpuRuntime>(
        sparse,
        &k_range,
        preprocessing,
        use_second_layer,
        Some(nmf_hals_opt),
        Some(consensus_opt),
        n_runs,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;
    release_device(&device);

    Ok(k_sweep_to_r_list(&sweep_res))
}
