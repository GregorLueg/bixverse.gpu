use bixverse_rs::gpu::ml::k_means_gpu::*;
use bixverse_rs::prelude::IntoExtendrErr;
use cubecl::wgpu::WgpuDevice;
use cubecl::wgpu::WgpuRuntime;
use cubecl::Runtime;
use faer::{Mat, MatRef};
use std::time::Instant;

/////////////////////////////
// GPU-accelerated k means //
/////////////////////////////

/// Standard k-means (GPU-accelerated)
///
/// Uses some approaches from the Flash-KMeans approach
///
/// ### Params
///
/// * `data` - The data of samples x features
/// * `dist` - The distance metric to use. Supports `"euclidean"` and
///   `"cosine"`.
/// * `n_centroids` - Number of centroids.
/// * `kmeans_params` - Optional user-specified [KMeansGpuParams].
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Result with (centroids, cluster_assignments)
pub fn kmeans_gpu_internal(
    data: MatRef<f32>,
    dist: &str,
    n_centroids: usize,
    kmeans_params: Option<KMeansGpuParams>,
    seed: usize,
    verbose: bool,
) -> Result<(Mat<f32>, Vec<usize>), extendr_api::Error> {
    let start = Instant::now();
    let device: WgpuDevice = Default::default();

    if verbose {
        println!("Running k-means clustering on the GPU via wgpu")
    }

    let (centroids, assignments) = k_means_clusters_gpu::<f32, WgpuRuntime>(
        data,
        dist,
        n_centroids,
        kmeans_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    if verbose {
        println!(
            "Finished the GPU-accelerated k-means clustering in {:.2?}",
            start.elapsed()
        )
    }

    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok((centroids, assignments))
}
