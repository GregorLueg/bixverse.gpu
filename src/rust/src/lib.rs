use extendr_api::prelude::*;
use faer::{Mat, MatRef};
use faer_entity::SimpleEntity;

pub mod single_cell;

use crate::single_cell::knn_gpu::*;

/////////////
// extendR //
/////////////

extendr_module! {
    mod bixverse_gpu;
    fn rs_cagra_knn;
    fn rs_ivf_gpu_knn;
}

/////////////
// Helpers //
/////////////

/// Transform an R matrix into a f32 one
///
/// ### Params
///
/// * `x` - R matrix with f64.
///
/// ### Returns
///
/// A faer Mat with f32
pub fn r_matrix_to_faer_fp32(x: &RMatrix<f64>) -> Mat<f32> {
    let ncol = x.ncols();
    let nrow = x.nrows();
    let data = x.data();
    let data_fp32 = data.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    Mat::from_fn(nrow, ncol, |i, j| data_fp32[i + j * nrow])
}

/// Transform a faer into an R matrix
///
/// ### Params
///
/// * `x` - faer `MatRef` matrix to transform into an R matrix
///
/// ###
///
/// The R matrix based on the faer matrix.
pub fn faer_to_r_matrix<T>(x: MatRef<T>) -> extendr_api::RArray<T::RType, [usize; 2]>
where
    T: FaerRType,
{
    T::to_r_matrix(x)
}

/// Bridge between faer matrix types and R matrix types.
///
/// Defines how to convert faer matrices to R-compatible arrays.
pub trait FaerRType: SimpleEntity + Copy + Clone + 'static {
    /// Type definition to allow R conversion
    type RType: Copy + Clone;

    /// Transform an faer matrix (f32/f64) into an R matrix (f64)
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<Self::RType, [usize; 2]>;
}

impl FaerRType for f64 {
    type RType = f64;
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<Self, [usize; 2]> {
        let nrow = x.nrows();
        let ncol = x.ncols();
        RArray::new_matrix(nrow, ncol, |row, column| x[(row, column)])
    }
}

impl FaerRType for i32 {
    type RType = i32;
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<Self, [usize; 2]> {
        let nrow = x.nrows();
        let ncol = x.ncols();
        RArray::new_matrix(nrow, ncol, |row, column| x[(row, column)])
    }
}

impl FaerRType for f32 {
    type RType = f64;
    fn to_r_matrix(x: faer::MatRef<Self>) -> extendr_api::RArray<f64, [usize; 2]> {
        let nrow = x.nrows();
        let ncol = x.ncols();
        RArray::new_matrix(nrow, ncol, |row, column| x[(row, column)] as f64)
    }
}

/////////
// kNN //
/////////

/// Generate a CAGRA kNN-based graph
///
/// @export
#[extendr]
fn rs_cagra_knn(
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

/// Generate a CAGRA kNN-based graph
///
/// @export
#[extendr]
fn rs_ivf_gpu_knn(embd: RMatrix<f64>, ivf_params: List, seed: usize, verbose: bool) -> List {
    let data = r_matrix_to_faer_fp32(&embd);
    let params = IvfGpuParams::from_r_list(ivf_params);

    let (indices, dist) = ivf_knn_with_dist(data.as_ref(), &params, true, seed, verbose);

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| knn_dist[i][j] as f64);

    list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = params.ann_dist
    )
}
