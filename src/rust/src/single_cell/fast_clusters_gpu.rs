//! GPU-accelerated fast clustering.
//!
//! Mirrors `bixverse::rs_fast_cluster_sc`. Only stage one moves to the device:
//! the k-means coarsening of the embedding and, for the cell-mediated graph,
//! the top-m search of the cells against the centroids. The centroid graph and
//! the community detection stay on the CPU inside `bixverse-rs`.
//!
//! There is no `km_type` argument here. Both GPU k-means are full-batch Lloyd's
//! and have no mini-batch path.

use crate::{parse_backend, Backend};
use ann_search_rs::gpu::k_means_gpu::KMeansGpuParams;
use bixverse_r::gpu::sc_gpu::fast_clusters_gpu::GpuKMeans;
#[cfg(mlx_available)]
use bixverse_r::mlx::sc::fast_clusters::MlxKMeans;
use bixverse_r::prelude::*;
use bixverse_r::single_cell::sc_analysis::fast_clusters::{
    fast_clusters, FastClusterParams, FastClusterResult, GridStats,
};
use cubecl::wgpu::WgpuDevice;
use extendr_api::*;

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod fast_clusters_gpu;
    // functions
    fn rs_fast_cluster_gpu;
}

//////////////////
// Fast cluster //
//////////////////

/// GPU: fast clustering on the data
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_fast_cluster_sc`. Coarsens the cells with
/// k-means on the GPU, builds a graph over the centroids (a kNN graph with one
/// of the symmetrisations, or the cell-mediated graph) and runs Louvain or
/// Leiden on it. The community of each centroid is propagated back to its
/// cells. With `n_seeds > 1` every resolution is run once per seed, the labels
/// come from the seed with the lowest mean conductance and stability metrics
/// across the seeds are returned. Everything after stage one stays on the CPU.
///
/// @param embd Numeric matrix. The embedding, cells x features.
/// @param resolutions Numeric vector. The resolutions to iterate through.
/// @param fc_params Named list. See [params_sc_fast_cluster_gpu()]. Carries
/// the GPU k-means keys (`k_means_iter`, `k_means_init`, `fixed`, `quantise`)
/// alongside the fast clustering ones.
/// @param n_seeds Integer. Number of seeds per resolution. `1L` runs a single
/// pass without stability metrics.
/// @param seed Integer. For reproducibility.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
/// @param backend String. `"wgpu"` or `"mlx"` (Apple Silicon, needs a build
/// with the MLX backend, see [mlx_available()]). MLX ignores `fixed` and
/// `quantise`.
///
/// @returns A list with the following elements:
/// \itemize{
///  \item membership - List with one integer membership vector per
///  resolution (0-indexed).
///  \item k_means_cluster - Integer vector with the k-means centroid per cell
///  (0-indexed).
///  \item centroids - Numerical matrix with the k-means centroids.
///  \item stats - `NULL` for `n_seeds = 1L`, otherwise a list with
///  `mean_ari`, `median_ari`, `mean_conductance`, `median_conductance` and
///  `mean_n_comms`, one value per resolution.
/// }
///
/// @export
///
/// @keywords internal
#[extendr]
fn rs_fast_cluster_gpu(
    embd: RMatrix<f64>,
    resolutions: &[f64],
    fc_params: List,
    n_seeds: usize,
    seed: usize,
    verbose: usize,
    #[extendr(default = "\"wgpu\"")] backend: &str,
) -> Result<List> {
    let backend = parse_backend(backend)?;

    let embd = r_matrix_to_faer_fp32(&embd);
    let resolutions: Vec<f32> = resolutions.iter().map(|&r| r as f32).collect();

    let params = FastClusterParams::from_r_list(fc_params.clone())?;
    let km = KMeansGpuParams::from_r_list(fc_params)?;

    let res = match backend {
        #[cfg(mlx_available)]
        Backend::Mlx => {
            let km = ann_search_rs::utils::k_means_utils::KMeansTrainingParams::new(
                km.iters, km.init, None,
            );
            fast_clusters(
                embd.as_ref(),
                &MlxKMeans::new(Some(km)),
                &params,
                &resolutions,
                n_seeds,
                seed,
                verbose,
            )
            .to_extendr()?
        }
        #[cfg(not(mlx_available))]
        Backend::Mlx => unreachable!("parse_backend rejects mlx without the MLX build"),
        Backend::Wgpu => {
            let device: WgpuDevice = Default::default();
            let res = fast_clusters(
                embd.as_ref(),
                &GpuKMeans::new(Some(km), device.clone().into()),
                &params,
                &resolutions,
                n_seeds,
                seed,
                verbose,
            )
            .to_extendr()?;

            // force VRAM memory clean up to avoid memory leaks
            let client = cubecl::Device::from(device).client();
            let _ = client.memory_cleanup();

            res
        }
    };

    fast_cluster_res_to_r_list(res, n_seeds)
}

/////////////
// Helpers //
/////////////

/// Converts the fast clustering results into an R list.
///
/// Same shape as the list `bixverse::rs_fast_cluster_sc` returns, so the R
/// wrappers can treat both alike.
///
/// ### Params
///
/// * `res` - The [FastClusterResult]
/// * `n_seeds` - Number of seeds per resolution; stats are only returned for
///   more than one
///
/// ### Returns
///
/// A named list with `membership`, `k_means_cluster`, `centroids` and `stats`.
fn fast_cluster_res_to_r_list(res: FastClusterResult, n_seeds: usize) -> Result<List> {
    let mut membership = List::new(res.per_resolution.len());
    for (i, r) in res.per_resolution.iter().enumerate() {
        membership.set_elt(i, Robj::from(r.labels.clone().r_int_convert()))?;
    }

    let stats = if n_seeds > 1 {
        let get = |f: fn(&GridStats) -> f32| -> Vec<f64> {
            res.per_resolution
                .iter()
                .map(|r| r.stability.as_ref().map_or(f64::NAN, |s| f(s) as f64))
                .collect()
        };
        r!(list!(
            mean_ari = get(|s| s.mean_ari),
            median_ari = get(|s| s.median_ari),
            mean_conductance = get(|s| s.mean_conductance),
            median_conductance = get(|s| s.median_conductance),
            mean_n_comms = get(|s| s.mean_n_communities)
        ))
    } else {
        r!(NULL)
    };

    Ok(list!(
        membership = membership,
        k_means_cluster = res.assignments.r_int_convert(),
        centroids = faer_to_r_matrix(res.centroids.as_ref()),
        stats = stats
    ))
}
