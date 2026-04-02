use bixverse_rs::prelude::*;
use burn::backend::{
    ndarray::{NdArray, NdArrayDevice},
    wgpu::{Wgpu, WgpuDevice},
    Autodiff,
};
use extendr_api::prelude::*;
use faer::{Mat, MatRef};
use manifolds_rs::parametric::model::TrainedUmapModel;

pub mod embeddings;
pub mod single_cell;

use crate::embeddings::parametric_umap::*;
use crate::single_cell::knn_gpu::*;

/////////////
// extendR //
/////////////

extendr_module! {
    mod bixverse_gpu;
    // knn
    fn rs_cagra_gpu_knn;
    fn rs_ivf_gpu_knn;
    fn rs_exhaustive_gpu_knn;
    // umap
    fn rs_parametric_umap;
    fn rs_parametric_umap_predict;
}

///////////
// Types //
///////////

type GpuBackend = Autodiff<Wgpu>;
type CpuBackend = Autodiff<NdArray<f32>>;

/// Backend-agnostic wrapper around `TrainedUmapModel`.
///
/// Allows the R-facing API to store a single `ExternalPtr<PUmapModel>` without
/// exposing the backend type parameter to the caller.
enum PUmapModel {
    /// Model trained on the WGPU backend
    Gpu(TrainedUmapModel<GpuBackend, f32>),
    /// Model trained on the NdArray CPU backend
    Cpu(TrainedUmapModel<CpuBackend, f32>),
}

impl PUmapModel {
    /// Run inference on new data, dispatching to the backend the model was
    /// trained on.
    ///
    /// ### Params
    ///
    /// * `data` - Reference to the input data matrix (samples x features)
    ///
    /// ### Returns
    ///
    /// Embeddings as `Vec<Vec<f32>>` in column-major layout
    /// `[n_components][n_samples]`
    fn predict(&self, data: MatRef<f32>) -> Vec<Vec<f32>> {
        match self {
            PUmapModel::Gpu(m) => m.predict(data),
            PUmapModel::Cpu(m) => m.predict(data),
        }
    }
}

/////////
// kNN //
/////////

/// Generate a CAGRA-style GPU-accelerated kNN graph
///
/// Builds a kNN graph from an embedding matrix using the CAGRA algorithm on
/// the wgpu backend. Supports two retrieval modes: direct extraction from the
/// NNDescent graph, or beam search over the pruned CAGRA graph.
///
/// @param embd Numeric matrix of embeddings, cells x features.
/// @param cagra_params A named list with the parameters, see
/// [bixverse.gpu::params_sc_cagra()]
/// @param extract_knn Logical. If \code{TRUE}, extracts the kNN graph directly
/// from the NNDescent result (faster, slightly lower precision). If
/// \code{FALSE}, runs beam search over the pruned CAGRA graph (slower, higher
/// precision).
/// @param seed Integer. Random seed for reproducibility.
/// @param verbose Logical. Whether to print progress messages.
///
/// @return A named list with:
/// \itemize{
///  \item `indices` - Integer matrix of shape cells x k_query with
///  0-based neighbour indices.
///  \item `dist` - Numeric matrix of shape cells x k_query with distances to
///  the neighbours.
///  \item `dist_metric` - Character. The distance metric used.
/// }
///
/// @export
#[extendr]
fn rs_cagra_gpu_knn(
    embd: RMatrix<f64>,
    cagra_params: List,
    extract_knn: bool,
    seed: usize,
    verbose: bool,
) -> List {
    let data = r_matrix_to_faer_fp32(&embd);
    let params = CagraParams::from_r_list(cagra_params);

    let (indices, dist) =
        cagra_knn_with_dist(data.as_ref(), &params, true, extract_knn, seed, verbose);

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), params.k_query, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), params.k_query, |i, j| knn_dist[i][j] as f64);

    list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = params.ann_dist
    )
}

/// Generate an IVF-GPU-accelerated kNN graph
///
/// Builds an IVF index over the provided embedding matrix and queries each
/// vector against it to produce a kNN graph. Runs on the wgpu backend.
///
/// @param embd Numeric matrix of embeddings, cells x features.
/// @param ivf_params A named list with the parameters, see
/// [bixverse.gpu::params_sc_ivf()]
/// @param seed Integer. Random seed for reproducibility.
/// @param verbose Logical. Whether to print progress messages.
///
/// @return A named list with:
/// \itemize{
///  \item `indices` - Integer matrix of shape cells x k with 0-based neighbour
///  indices.
///  \item `dist` - Numeric matrix of shape cells x k with distances to the
///  neighbours.
///  \item `dist_metric` - Character. The distance metric used.
/// }
///
/// @export
#[extendr]
fn rs_ivf_gpu_knn(embd: RMatrix<f64>, ivf_params: List, seed: usize, verbose: bool) -> List {
    let data = r_matrix_to_faer_fp32(&embd);
    let params = IvfGpuParams::from_r_list(ivf_params);

    let (indices, dist) = gpu_ivf_knn_with_dist(data.as_ref(), &params, true, seed, verbose);

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| knn_dist[i][j] as f64);

    list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = params.ann_dist
    )
}

/// Generate an GPU-accelerated kNN graph from an exhaustive search
///
/// Runs an exhaustive kNN search on the GPU.
///
/// @param embd Numeric matrix of embeddings, cells x features.
/// @param k Integer. Number of neighbours to return.
/// @param dist_metric String. Distance metric; one of
/// `c("euclidean", "cosine")`.
/// @param verbose Logical. Whether to print progress messages.
///
/// @return A named list with:
/// \itemize{
///  \item `indices` - Integer matrix of shape cells x k with 0-based neighbour
///  indices.
///  \item `dist` - Numeric matrix of shape cells x k with distances to the
///  neighbours.
///  \item `dist_metric` - Character. The distance metric used.
/// }
///
/// @export
#[extendr]
fn rs_exhaustive_gpu_knn(embd: RMatrix<f64>, k: usize, dist_metric: String, verbose: bool) -> List {
    let data = r_matrix_to_faer_fp32(&embd);

    let (indices, dist) =
        gpu_exhaustive_knn_with_dist(data.as_ref(), k, &dist_metric, true, verbose);

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), k, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), k, |i, j| knn_dist[i][j] as f64);

    list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = dist_metric
    )
}

/////////////////////
// parametric UMAP //
/////////////////////

/// Parametric UMAP implementation
///
/// Trains a neural network encoder to learn a mapping from the input space to a
/// low-dimensional embedding that preserves the UMAP graph structure. Supports
/// both GPU (wgpu) and CPU (NdArray) backends. For small to medium data sets
/// (fewer than ~10k samples or narrow hidden layers), the CPU backend is
/// typically faster owing to GPU kernel dispatch overhead.
///
/// @param data Numerical matrix. Data of dimensions samples x features.
/// @param n_dim Integer. Number of embedding dimensions.
/// @param k Integer. Number of nearest neighbours for graph construction.
/// @param min_dist Numeric. Minimum distance between embedded points.
/// @param spread Numeric. Effective scale of embedded points.
/// @param parametric_params Named list. Merged parametric UMAP parameters
/// containing nearest neighbour, graph, and training configuration.
/// @param seed Integer. Seed for reproducibility.
/// @param verbose Boolean. Controls verbosity.
/// @param use_gpu Logical. If \code{TRUE}, trains on the wgpu backend. If
/// \code{FALSE}, trains on the CPU via NdArray. Defaults to \code{TRUE}.
///
/// @return A named list with two elements: `embedding` (numerical matrix of
/// dimensions samples x n_dim) and `model` (external pointer to the trained
/// encoder for use with `rs_parametric_umap_predict`).
///
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_parametric_umap(
    data: RMatrix<f64>,
    n_dim: usize,
    k: usize,
    min_dist: f64,
    spread: f64,
    parametric_params: List,
    seed: usize,
    verbose: bool,
    use_gpu: bool,
) -> List {
    let data = r_matrix_to_faer_fp32(&data);

    if use_gpu {
        let device = WgpuDevice::default();
        let (res, model) = parametric_umap_manifold::<GpuBackend>(
            data.as_ref(),
            n_dim,
            k,
            min_dist as f32,
            spread as f32,
            parametric_params,
            &device,
            seed,
            verbose,
        );

        list!(
            embedding = faer_to_r_matrix(res.as_ref()),
            model = ExternalPtr::new(PUmapModel::Gpu(model))
        )
    } else {
        let device = NdArrayDevice::Cpu;
        let (res, model) = parametric_umap_manifold::<CpuBackend>(
            data.as_ref(),
            n_dim,
            k,
            min_dist as f32,
            spread as f32,
            parametric_params,
            &device,
            seed,
            verbose,
        );

        list!(
            embedding = faer_to_r_matrix(res.as_ref()),
            model = ExternalPtr::new(PUmapModel::Cpu(model))
        )
    }
}

/// Predict new data using a trained parametric UMAP model
///
/// Runs forward inference through the trained encoder network. The prediction
/// automatically uses whichever backend (GPU or CPU) the model was trained on.
///
/// @param model External pointer to the trained parametric UMAP model, as
/// returned by `rs_parametric_umap`.
/// @param data Numerical matrix. New data of dimensions samples x features.
/// The number of features must match the training data.
///
/// @return Numerical matrix of dimensions samples x n_dim with the predicted
/// embeddings.
///
/// @export
#[extendr]
fn rs_parametric_umap_predict(model: Robj, data: RMatrix<f64>) -> RMatrix<f64> {
    let model: ExternalPtr<PUmapModel> = model
        .try_into()
        .expect("failed to convert to ExternalPtr<PUmapModel>");
    let data = r_matrix_to_faer_fp32(&data);
    let res = model.predict(data.as_ref());

    let ncol = res.len();
    let nrow = res[0].len();
    let mat = Mat::from_fn(nrow, ncol, |i, j| res[j][i]);
    faer_to_r_matrix(mat.as_ref())
}
