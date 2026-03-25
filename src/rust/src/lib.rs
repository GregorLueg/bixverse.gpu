use bixverse_rs::prelude::*;
use extendr_api::prelude::*;
use faer::Mat;

pub mod single_cell;

use crate::single_cell::knn_gpu::*;

/////////////
// extendR //
/////////////

extendr_module! {
    mod bixverse_gpu;
    fn rs_cagra_gpu_knn;
    fn rs_ivf_gpu_knn;
    fn rs_exhaustive_gpu_knn;
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
