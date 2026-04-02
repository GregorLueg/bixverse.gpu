use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use burn::tensor::backend::AutodiffBackend;
use extendr_api::*;
use faer::{Mat, MatRef};
use manifolds_rs::parametric::model::TrainedUmapModel;
use manifolds_rs::prelude::*;
use manifolds_rs::*;

use crate::embeddings::utils::*;

////////////
// Params //
////////////

/// Internal representation of parametric UMAP parameters parsed from R.
#[derive(Debug)]
pub struct InternalParametricUmapParams {
    /// Which approximate nearest neighbour search to use.
    pub ann_type: String,
    /// Hidden layer sizes for the MLP encoder.
    pub hidden_layers: Vec<usize>,
    /// Nearest neighbour parameters.
    pub param_knn: NearestNeighbourParams<f32>,
    /// UMAP graph construction parameters.
    pub umap_graph: UmapGraphParams<f32>,
    /// Training parameters for the neural network.
    pub train_param: TrainParametricParams<f32>,
}

impl InternalParametricUmapParams {
    /// Parse parametric UMAP parameters from a merged R list.
    ///
    /// ### Params
    ///
    /// * `r_list` - Merged R list containing nn + parametric umap params.
    /// * `min_dist` - Minimum distance parameter.
    /// * `spread` - Spread parameter.
    ///
    /// ### Returns
    ///
    /// Parsed `InternalParametricUmapParams`.
    pub fn from_r_list(r_list: List, min_dist: f32, spread: f32) -> Self {
        let nn_params = get_params_nn(r_list.clone());
        let umap_graph_params = get_params_umap_graph(r_list.clone());
        let train_param = get_params_parametric_train(r_list.clone(), min_dist, spread);

        let params = r_list.into_hashmap();

        let ann_type = std::string::String::from(
            params
                .get("knn_method")
                .and_then(|v| v.as_str())
                .unwrap_or("hnsw"),
        );

        let hidden_layers = params
            .get("hidden_layers")
            .and_then(|v| v.as_integer_vector())
            .map(|v| v.into_iter().map(|x| x as usize).collect())
            .unwrap_or(vec![128, 128, 128]);

        Self {
            ann_type,
            hidden_layers,
            param_knn: nn_params,
            umap_graph: umap_graph_params,
            train_param,
        }
    }
}

/////////////////////
// Parametric UMAP //
/////////////////////

/// Builds `ParametricUmapParams` from R inputs and runs parametric UMAP
///
/// Parses the R parameter list into internal configuration, constructs the
/// parameter struct, runs parametric UMAP, and returns the result as a
/// column-major matrix.
///
/// ### Params
///
/// * `data` - Input data matrix (n_samples × n_features)
/// * `n_dim` - Number of output embedding dimensions
/// * `k` - Number of nearest neighbours used in graph construction
/// * `min_dist` - Minimum distance between points in the embedding
/// * `spread` - Scale of the embedding; controls how clustered the output is
/// * `parametric_params` - R list containing network and training configuration
/// * `device` - Burn backend device to run the model on
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Whether to print progress during training
///
/// ### Returns
///
/// The embedding generated from the trainings data and the trained
/// UmapMlpModel.
#[allow(clippy::too_many_arguments)]
pub fn parametric_umap_manifold<B>(
    data: MatRef<f32>,
    n_dim: usize,
    k: usize,
    min_dist: f32,
    spread: f32,
    parametric_params: List,
    device: &B::Device,
    seed: usize,
    verbose: bool,
) -> (Mat<f32>, TrainedUmapModel<B, f32>)
where
    B: AutodiffBackend,
    HnswIndex<f32>: HnswState<f32>,
    NNDescent<f32>: ApplySortedUpdates<f32> + NNDescentQuery<f32>,
{
    let internal = InternalParametricUmapParams::from_r_list(parametric_params, min_dist, spread);

    let umap_params = ParametricUmapParams::new(
        Some(n_dim),
        Some(k),
        Some(internal.ann_type),
        Some(internal.hidden_layers),
        Some(internal.param_knn),
        Some(internal.umap_graph),
        Some(internal.train_param),
    );

    let (embd_res, model) =
        train_parametric_umap_model::<f32, B>(data, &umap_params, device, seed, verbose);

    let ncol = embd_res.len();
    let nrow = embd_res[0].len();
    let res = Mat::from_fn(nrow, ncol, |i, j| embd_res[j][i]);

    (res, model)
}
