//! Utility functions (mostly copied over from manifoldsR)

use ann_search_rs::prelude::AnnSearchFloat;
use extendr_api::*;
use manifolds_rs::prelude::*;
use std::collections::HashMap;

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
pub fn get_params_nn(r_list: List) -> Result<NearestNeighbourParams<f32>> {
    let nn_params: HashMap<&str, Robj> = r_list.try_into()?;

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
        .get("n_probes")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    // nndescent
    let extract_knn = nn_params
        .get("extract_knn")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Ok(NearestNeighbourParams {
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
        extract_knn,
    })
}

////////////////////////
// Nearest neighbours //
////////////////////////

/// Helper function to generate the GPU nearest neighbour parameters
///
/// ### Params
///
/// * `r_list` - The list that has the nearest neighbour graph generation
///   parameters.
///
/// ### Returns
///
/// The `NearestNeighbourParamsGpu` with sensible defaults if not found in
/// the list.
pub fn get_params_nn_ann_gpu<T>(r_list: List) -> Result<NearestNeighbourParamsGpu<T>>
where
    T: AnnSearchFloat,
{
    let nn_params: HashMap<&str, Robj> = r_list.try_into()?;
    let dist_metric = std::string::String::from(
        nn_params
            .get("dist_metric")
            .and_then(|v| v.as_str())
            .unwrap_or("euclidean"),
    );
    let n_list = nn_params
        .get("n_list")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    let n_probes = nn_params
        .get("n_probes")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    // this is the final node degree of the CAGRA graph
    let k = nn_params
        .get("node_degree_final")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    let k_build = nn_params
        .get("k_build")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    let n_tree = nn_params
        .get("n_tree")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    let delta = nn_params
        .get("delta")
        .and_then(|v| v.as_real())
        .map(|v| T::from(v).unwrap())
        .unwrap_or(T::from(0.001).unwrap());
    let rho = nn_params
        .get("rho")
        .and_then(|v| v.as_real())
        .map(|v| T::from(v).unwrap());
    let beam_width = nn_params
        .get("beam_width")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    let max_beam_iters = nn_params
        .get("max_beam_iters")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);
    let n_entry_points = nn_params
        .get("n_entry_points")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize);

    let extract_knn = nn_params
        .get("extract_knn")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Ok(NearestNeighbourParamsGpu {
        dist_metric,
        n_list,
        n_probes,
        k,
        k_build,
        n_tree,
        delta,
        rho,
        beam_width,
        max_beam_iters,
        n_entry_points,
        extract_knn,
    })
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
) -> Result<TrainParametricParams<f32>> {
    let params: HashMap<&str, Robj> = r_list.try_into()?;

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

    Ok(TrainParametricParams::from_min_dist_spread(
        min_dist,
        spread,
        corr_weight,
        lr,
        n_epochs,
        batch_size,
        neg_sample_rate,
    ))
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
pub fn get_params_umap_graph(r_list: List) -> Result<UmapGraphParams<f32>> {
    let graph_params: HashMap<&str, Robj> = r_list.try_into()?;

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

    Ok(UmapGraphParams {
        bandwidth,
        local_connectivity,
        mix_weight,
    })
}
