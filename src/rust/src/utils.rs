use bixverse_r::gpu::sc_gpu::knn_gpu::KnnParamsGpu;
use extendr_api::*;
use manifolds_rs::prelude::*;
use manifolds_rs::PreComputedKnn;
use rustc_hash::FxHashMap;

/// Parse the nearest neighbours to a Rust function
///
/// ### Params
///
/// * `List` - The NearestNeighbour list.
///
/// ### Returns
///
/// A PreComputedKnn<f32> for the 2D embedding methods
pub fn nearest_neighbours_to_rust<T>(nn: List) -> PreComputedKnn<T>
where
    T: ManifoldsFloat,
{
    let k = nn.dollar("k").unwrap().as_integer().unwrap() as usize;

    let indices: Vec<i32> = nn.dollar("indices").unwrap().as_integer_vector().unwrap();
    let dist: Vec<f64> = nn.dollar("dist").unwrap().as_real_vector().unwrap();

    let indices: Vec<Vec<usize>> = indices
        .chunks(k)
        .map(|chunk| chunk.iter().map(|&i| (i - 1) as usize).collect())
        .collect();

    let dist: Vec<Vec<T>> = dist
        .chunks(k)
        .map(|chunk| chunk.iter().map(|&d| T::from_f64(d).unwrap()).collect())
        .collect();

    Some((indices, dist))
}

/// Translate a `params_nn_gpu()` R list into bixverse-rs GPU kNN parameters
///
/// The R list keeps the manifolds-rs key names (`dist_metric`, `n_probes`,
/// `node_degree_final`) it shares with UMAP and tSNE, while [`KnnParamsGpu`]
/// uses the bixverse ones (`ann_dist`, `n_probe`, `graph_k`). Missing keys
/// fall back to [`KnnParamsGpu::default()`].
///
/// ### Params
///
/// * `r_list` - The nearest neighbour parameters.
/// * `knn_method` - One of `"exhaustive"`, `"ivf"`, `"nndescent"` or
///   `"cagra"`.
/// * `k` - Number of neighbours to return.
///
/// ### Returns
///
/// The [`KnnParamsGpu`] for the run.
pub fn nn_list_to_knn_params_gpu(r_list: List, knn_method: &str, k: usize) -> Result<KnnParamsGpu> {
    let nn: FxHashMap<&str, Robj> = r_list.iter().collect();
    let defaults = KnnParamsGpu::default();
    let count = |key: &str| nn.get(key).and_then(|v| v.as_integer()).map(|v| v as usize);

    Ok(KnnParamsGpu {
        knn_method: knn_method.to_string(),
        ann_dist: nn
            .get("dist_metric")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or(defaults.ann_dist),
        k,
        n_list: count("n_list"),
        n_probe: count("n_probes"),
        graph_k: count("node_degree_final"),
        k_build: count("k_build"),
        n_tree: count("n_tree"),
        delta: nn
            .get("delta")
            .and_then(|v| v.as_real())
            .map(|v| v as f32)
            .unwrap_or(defaults.delta),
        rho: nn.get("rho").and_then(|v| v.as_real()).map(|v| v as f32),
        refine_knn: defaults.refine_knn,
        beam_width: count("beam_width"),
        max_beam_iters: count("max_beam_iters"),
        n_entry_points: count("n_entry_points"),
    })
}
