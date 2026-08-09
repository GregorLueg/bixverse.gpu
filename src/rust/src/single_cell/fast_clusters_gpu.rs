//! GPU-accelerated fast Louvain clustering.
//!
//! Mirrors `bixverse::rs_fast_cluster_sc` and `rs_fast_cluster_sc_grid`. Only
//! stage one moves to the device: the k-means coarsening of the embedding. The
//! centroid kNN, the optional sNN pass and the Louvain runs all stay on the CPU
//! inside `bixverse-rs`, so the speedup tracks how much of the run k-means owns.
//!
//! There is no `km_type` argument here. The GPU k-means is full-batch Lloyd's
//! and has no mini-batch path. The distance metric comes from the kNN params, so
//! the coarsening and the centroid graph agree on the geometry.

use bixverse_rs::gpu::sc_gpu::fast_clusters_gpu::{
    fast_louvain_clusters_gpu, fast_louvain_clusters_grid_gpu, FastLouvainParamsGpu,
};
use bixverse_rs::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;

use crate::single_cell::sc_utils::{
    fast_cluster_unwrap_multiple, fast_cluster_unwrap_single, process_fc_louvain_results,
};

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod fast_clusters_gpu;
    // functions
    fn rs_fast_cluster_gpu;
    fn rs_fast_cluster_grid_gpu;
}

//////////////////
// Fast cluster //
//////////////////

/// GPU: fast Louvain clustering on the data
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_fast_cluster_sc`. Runs k-means clustering on
/// the WGPU backend, followed by a kNN detection on the centroids to then run
/// Louvain clustering on the graph and propagate the membership back to the
/// original data. Everything after the k-means stays on the CPU.
///
/// @param embd Numeric matrix. The original embedding.
/// @param resolutions Numeric vector. The Louvain resolutions to iterate
/// through.
/// @param n_centroids Optional integer. The number of clusters to find. If
/// not provided, defaults to `sqrt(nrow(embd))`.
/// @param fc_params Named list. See [params_sc_fast_cluster_gpu()].
/// @param snn Boolean. Shall the kNN graph be additionally transformed into
/// an sNN graph.
/// @param return_kmeans Boolean. Shall the k-means centroids and assignments
/// be returned alongside the memberships.
/// @param seed Integer. For reproducibility.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with the following elements:
/// \itemize{
///  \item membership - The memberships across the different resolutions.
///  \item k_means_cluster - Optional integer vector of k-means assignments.
///  \item centroids - Optional numeric matrix of k-means centroids.
/// }
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fast_cluster_gpu(
    embd: RMatrix<f64>,
    resolutions: &[f64],
    n_centroids: Option<usize>,
    fc_params: List,
    snn: bool,
    return_kmeans: bool,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    let embd = r_matrix_to_faer_fp32(&embd);
    let n_clusters = n_centroids.unwrap_or(((embd.nrows() as f32).sqrt()) as usize);
    let resolutions = resolutions.r_float_convert();

    let mut params = FastLouvainParamsGpu::from_r_list(fc_params)?;
    params.n_centroids = n_clusters;

    let device: WgpuDevice = Default::default();

    let fc_results = fast_louvain_clusters_gpu::<WgpuRuntime>(
        embd.as_ref(),
        &resolutions,
        &params,
        snn,
        return_kmeans,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let (memberships, k_means_cluster, centroids) =
        fast_cluster_unwrap_single(fc_results, return_kmeans)?;

    let mut memberships_ls = List::new(memberships.len());

    for (index, membership) in memberships.iter().enumerate() {
        let membership = membership.clone().r_int_convert();
        memberships_ls.set_elt(index, Robj::from(membership))?;
    }

    let k_means_cluster = k_means_cluster.map_or_else(|| r!(NULL), |v| r!(v));
    let centroids = centroids.map_or_else(|| r!(NULL), |m| r!(m));

    Ok(list!(
        membership = memberships_ls,
        k_means_cluster = r!(k_means_cluster),
        centroids = r!(centroids),
    ))
}

/// GPU: fast Louvain clustering on the data (with multiple seeds)
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_fast_cluster_sc_grid`. Builds the k-means to
/// kNN/sNN graph once, then runs Louvain with several seeds (derived from the
/// original one) for every resolution. Returns additional metrics around
/// cluster stability and community conductance. Only the k-means runs on the
/// GPU.
///
/// @param embd Numeric matrix. The original embedding.
/// @param resolutions Numeric vector. The Louvain resolutions to iterate
/// through.
/// @param n_centroids Optional integer. The number of clusters to find. If
/// not provided, defaults to `sqrt(nrow(embd))`.
/// @param fc_params Named list. See [params_sc_fast_cluster_gpu()].
/// @param snn Boolean. Shall the kNN graph be additionally transformed into
/// an sNN graph.
/// @param return_kmeans Boolean. Shall the k-means centroids and assignments
/// be returned alongside the grid results.
/// @param no_seeds Integer. Number of additional seeds to use. Should be >= 2.
/// @param seed Integer. For reproducibility.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
///
/// @returns A list with the following elements:
/// \itemize{
///  \item membership - A list with `memberships` (the labels from the seed with
///  the best conductance, per resolution) and `stats` (the metrics per
///  resolution).
///  \item k_means_cluster - Optional integer vector of k-means assignments.
///  \item centroids - Optional numeric matrix of k-means centroids.
/// }
///
/// @export
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fast_cluster_grid_gpu(
    embd: RMatrix<f64>,
    resolutions: &[f64],
    n_centroids: Option<usize>,
    fc_params: List,
    snn: bool,
    return_kmeans: bool,
    no_seeds: usize,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    let embd = r_matrix_to_faer_fp32(&embd);
    let n_clusters = n_centroids.unwrap_or(((embd.nrows() as f32).sqrt()) as usize);
    let resolutions = resolutions.r_float_convert();

    let mut params = FastLouvainParamsGpu::from_r_list(fc_params)?;
    params.n_centroids = n_clusters;

    let device: WgpuDevice = Default::default();

    // The driver always keeps the k-means block here and `return_kmeans` is
    // honoured at the unwrap step. Matches the CPU grid path.
    let fc_results = fast_louvain_clusters_grid_gpu::<WgpuRuntime>(
        embd.as_ref(),
        &resolutions,
        &params,
        snn,
        true,
        seed,
        no_seeds,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let (memberships, k_means_cluster, centroids) =
        fast_cluster_unwrap_multiple(fc_results, return_kmeans)?;

    let membership_ls = process_fc_louvain_results(memberships)?;

    let k_means_cluster = k_means_cluster.map_or_else(|| r!(NULL), |v| r!(v));
    let centroids = centroids.map_or_else(|| r!(NULL), |m| r!(m));

    Ok(list!(
        membership = membership_ls,
        k_means_cluster = r!(k_means_cluster),
        centroids = r!(centroids),
    ))
}
