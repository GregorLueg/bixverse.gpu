//! Utility functions (mostly copied over from manifoldsR)

use extendr_api::*;
use manifolds_rs::prelude::*;

/////////////
// Helpers //
/////////////

/// Helper function to generate the UMAP NN parameters
///
/// Ported over from `manifoldsR`.
///
/// ### Params
///
/// * `r_list` - The list that has the nearest neighbour graph generation
///   parameters.
///
/// ### Returns
///
/// The `NearestNeighbourParams` with sensible defaults if not found in the
/// list.
pub fn get_params_nn(r_list: List) -> NearestNeighbourParams<f32> {
    let nn_params = r_list.into_hashmap();

    // distance
    let dist_metric = std::string::String::from(
        nn_params
            .get("dist_metric")
            .and_then(|v| v.as_str())
            .unwrap_or("cosine"),
    );

    // annoy
    let n_tree = nn_params
        .get("n_tree")
        .and_then(|v| v.as_integer())
        .unwrap_or(50) as usize;

    let search_budget = nn_params
        .get("search_budget")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    // hnsw
    let m = nn_params
        .get("m")
        .and_then(|v| v.as_integer())
        .unwrap_or(16) as usize;

    let ef_construction = nn_params
        .get("ef_construction")
        .and_then(|v| v.as_integer())
        .unwrap_or(100) as usize;

    let ef_search = nn_params
        .get("ef_search")
        .and_then(|v| v.as_integer())
        .unwrap_or(100) as usize;

    // nn descent
    let diversify_prob = nn_params
        .get("diversify_prob")
        .and_then(|v| v.as_real())
        .unwrap_or(0.0) as f32;

    let delta = nn_params
        .get("delta")
        .and_then(|v| v.as_real())
        .unwrap_or(0.001) as f32;

    let ef_budget = nn_params
        .get("ef_budget")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    // balltree
    let bt_budget = nn_params
        .get("bt_budget")
        .and_then(|v| v.as_real())
        .unwrap_or(0.1) as f32;

    // ivf
    let n_list = nn_params
        .get("n_list")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    let n_probes = nn_params
        .get("n_probe")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    NearestNeighbourParams {
        dist_metric,
        n_tree,
        search_budget,
        m,
        ef_construction,
        ef_budget,
        ef_search,
        diversify_prob,
        delta,
        bt_budget,
        n_list,
        n_probes,
    }
}

/// Parse the parametric UMAP training parameters from an R list.
///
/// Computes `a` and `b` from `min_dist` and `spread` via curve fitting.
///
/// ### Params
///
/// * `r_list` - The R list containing the parameters
/// * `min_dist` - Minimum distance parameter for UMAP
/// * `spread` - The spread parameter.
///
/// ### Returns
///
/// The `TrainParametricParams`
pub fn get_params_parametric_train(
    r_list: List,
    min_dist: f32,
    spread: f32,
) -> TrainParametricParams<f32> {
    let params = r_list.into_hashmap();

    let corr_weight = params
        .get("corr_weight")
        .and_then(|v| v.as_real())
        .unwrap_or(0.0) as f32;

    let lr = params.get("lr").and_then(|v| v.as_real()).map(|v| v as f32);

    let n_epochs = params
        .get("n_epochs")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    let batch_size = params
        .get("batch_size")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    let neg_sample_rate = params
        .get("neg_sample_rate")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    TrainParametricParams::from_min_dist_spread(
        min_dist,
        spread,
        corr_weight,
        lr,
        n_epochs,
        batch_size,
        neg_sample_rate,
    )
}

/// Helper function to generate the UMAP graph construction parameters
///
/// Ported over from manifoldsR
///
/// ### Params
///
/// * `r_list` - The list that has the UMAP graph construction parameters.
///
/// ### Returns
///
/// The `UmapGraphParams` with sensible defaults if not found in the list.
pub fn get_params_umap_graph(r_list: List) -> UmapGraphParams<f32> {
    let graph_params = r_list.into_hashmap();

    let mix_weight = graph_params
        .get("mix_weight")
        .and_then(|v| v.as_real())
        .unwrap_or(1.0) as f32;

    let local_connectivity = graph_params
        .get("local_connectivity")
        .and_then(|v| v.as_real())
        .unwrap_or(1.0) as f32;

    let bandwidth = graph_params
        .get("bandwidth")
        .and_then(|v| v.as_real())
        .unwrap_or(1e-5) as f32;

    UmapGraphParams {
        bandwidth,
        local_connectivity,
        mix_weight,
    }
}
