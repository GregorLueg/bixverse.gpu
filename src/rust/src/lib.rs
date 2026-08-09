use bixverse_rs::gpu::linalg::corr::{column_pairwise_cor_gpu, GpuCorCov};
use bixverse_rs::gpu::ml::k_means_gpu::KMeansGpuParams;
use bixverse_rs::prelude::*;
use burn::backend::{
    flex::{Flex, FlexDevice},
    wgpu::{Wgpu, WgpuDevice, WgpuRuntime},
    Autodiff,
};
use cubecl::Runtime;
use extendr_api::prelude::*;
use faer::{Mat, MatRef};
use manifolds_rs::parametric::model::TrainedUmapModel;

pub mod embeddings;
pub mod ml;
pub mod single_cell;
pub mod utils;

pub use single_cell::fast_clusters_gpu;
pub use single_cell::harmony_gpu;
pub use single_cell::pca_gpu;
pub use single_cell::scenic_gpu;
pub use single_cell::seacells_gpu;

use crate::embeddings::parametric_umap::*;
use crate::embeddings::tsne_gpu::tsne_manifold_gpu;
use crate::embeddings::umap_gpu::umap_manifold_gpu;
use crate::ml::clustering::*;
use crate::single_cell::knn_gpu::*;
use crate::utils::nearest_neighbours_to_rust;

/////////////
// extendR //
/////////////

extendr_module! {
    mod bixverse_gpu;
    // modules
    use pca_gpu;
    use harmony_gpu;
    use scenic_gpu;
    use seacells_gpu;
    use fast_clusters_gpu;
    // knn
    fn rs_cagra_gpu_knn;
    fn rs_ivf_gpu_knn;
    fn rs_exhaustive_gpu_knn;
    // umap parametric
    fn rs_parametric_umap;
    fn rs_parametric_umap_predict;
    fn rs_serialise_parametric_umap;
    fn rs_deserialise_parametric_umap;
    // clustering
    fn rs_kmeans_gpu;
    // correlations
    fn rs_cor_gpu;
    fn rs_cov_gpu;
    // umap gpu
    fn rs_umap_gpu;
    fn rs_umap_from_knn_gpu;
    // tsne gpu
    fn rs_tsne_gpu;
    fn rs_tsne_from_knn_gpu;
}

////////////
// Consts //
////////////

/// Threshold at which the switch from fp32 to fp64 happens. Some algorithms
/// (tSNE, cough) are very sensitive to precision differences.
const SAMPLE_THRESHOLD_HIGH_PRECISION: usize = 100_000;

///////////
// Types //
///////////

/// GPU backend via WGPU
type GpuBackend = Autodiff<Wgpu>;

/// CPU backend via flex
type CpuBackend = Autodiff<Flex<f32>>;

/// Backend-agnostic wrapper around `TrainedUmapModel`.
///
/// Allows the R-facing API to store a single `ExternalPtr<PUmapModel>` without
/// exposing the backend type parameter to the caller.
enum PUmapModel {
    /// Model trained on the WGPU backend
    Gpu(TrainedUmapModel<GpuBackend, f32>),
    /// Model trained on the flex CPU backend
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

///////////////
// Precision //
///////////////

///////////
// Enums //
///////////

/// Enum defining the floating point operation precision. Some algorithms suffer
/// from catastrophic cancellation and accumulation of precision errors
#[derive(Debug, Clone, Copy, Default)]
enum FloatingPointPrecision {
    /// FP32
    #[default]
    FP32,
    /// FP64
    FP64,
}

/// Auto detection of the precision based on sensible thresholds
///
/// ### Params
///
/// * `n` - Number of samples
///
/// ### Returns
///
/// The [FloatingPointPrecision]
fn auto_precision(n: usize) -> FloatingPointPrecision {
    if n > SAMPLE_THRESHOLD_HIGH_PRECISION {
        FloatingPointPrecision::FP64
    } else {
        FloatingPointPrecision::FP32
    }
}

/// Function to parse the precision to use
///
/// ### Params
///
/// * `use_high_precision` - Nullable Rbool. If provided, will use that to
///   determine precision to use
/// * `n` - Number of samples in the data. Used to determine in an auto
///   threshold which precision to use.
///
/// ### Returns
///
/// The determined [FloatingPointPrecision].
fn parse_precision(use_high_precision: Nullable<Rbool>, n: usize) -> FloatingPointPrecision {
    if use_high_precision != Nullable::Null {
        let val = use_high_precision.into_option().map(|x| x.to_bool()).unwrap_or_else(|| {
            println!("[WARNING] - The provided 'use_high_precision' could not be decoded into a boolean. Function will assume 'true'");
            true
        });
        if val {
            FloatingPointPrecision::FP64
        } else {
            FloatingPointPrecision::FP32
        }
    } else {
        auto_precision(n)
    }
}

/////////
// kNN //
/////////

/// Generate a CAGRA-style GPU-accelerated kNN graph
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Builds a kNN graph from an embedding matrix using the CAGRA algorithm on
/// the wgpu backend. Supports two retrieval modes: direct extraction from the
/// NNDescent graph, or beam search over the pruned CAGRA graph. The former
/// tends to have worse Recall.
///
/// @param embd Numeric matrix of embeddings, cells x features.
/// @param cagra_params A named list with the parameters, see
/// [bixverse.gpu::params_sc_cagra()]
/// @param extract_knn Logical. If \code{TRUE}, extracts the kNN graph directly
/// from the NNDescent result (faster, slightly lower precision). If
/// \code{FALSE}, runs beam search over the pruned CAGRA graph (slower, higher
/// precision).
/// @param seed Integer. Random seed for reproducibility.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
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
    verbose: usize,
) -> Result<List, extendr_api::Error> {
    let verbosity = parse_verbosity_level(verbose);

    let data = r_matrix_to_faer_fp32(&embd);
    let params = CagraParams::from_r_list(cagra_params)?;

    let (indices, dist) = cagra_knn_with_dist(
        data.as_ref(),
        &params,
        true,
        extract_knn,
        seed,
        verbosity.normal_verbosity(),
    )
    .to_extendr()?;

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| knn_dist[i][j] as f64);

    Ok(list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = params.ann_dist
    ))
}

/// Generate an IVF-GPU-accelerated kNN graph
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Builds an IVF index over the provided embedding matrix and queries each
/// vector against it to produce a kNN graph. Runs on the wgpu backend.
///
/// @param embd Numeric matrix of embeddings, cells x features.
/// @param ivf_params A named list with the parameters, see
/// [bixverse.gpu::params_sc_ivf()]
/// @param seed Integer. Random seed for reproducibility.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
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
fn rs_ivf_gpu_knn(
    embd: RMatrix<f64>,
    ivf_params: List,
    seed: usize,
    verbose: usize,
) -> Result<List, extendr_api::Error> {
    let verbosity = parse_verbosity_level(verbose);

    let data = r_matrix_to_faer_fp32(&embd);
    let params = IvfGpuParams::from_r_list(ivf_params)?;

    let (indices, dist) = gpu_ivf_knn_with_dist(
        data.as_ref(),
        &params,
        true,
        seed,
        verbosity.normal_verbosity(),
    )
    .to_extendr()?;

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), params.k, |i, j| knn_dist[i][j] as f64);

    Ok(list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = params.ann_dist
    ))
}

/// Generate an GPU-accelerated kNN graph from an exhaustive search
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Runs an exhaustive kNN search on the GPU.
///
/// @param embd Numeric matrix of embeddings, cells x features.
/// @param k Integer. Number of neighbours to return.
/// @param dist_metric String. Distance metric; one of
/// `c("euclidean", "cosine")`.
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
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
fn rs_exhaustive_gpu_knn(
    embd: RMatrix<f64>,
    k: usize,
    dist_metric: String,
    verbose: usize,
) -> Result<List, extendr_api::Error> {
    let verbosity = parse_verbosity_level(verbose);

    let data = r_matrix_to_faer_fp32(&embd);

    let (indices, dist) = gpu_exhaustive_knn_with_dist(
        data.as_ref(),
        k,
        &dist_metric,
        true,
        verbosity.normal_verbosity(),
    )
    .to_extendr()?;

    let knn_dist = dist.unwrap();

    let index_mat = Mat::from_fn(embd.nrows(), k, |i, j| indices[i][j] as i32);
    let dist_mat = Mat::from_fn(embd.nrows(), k, |i, j| knn_dist[i][j] as f64);

    Ok(list!(
        indices = faer_to_r_matrix(index_mat.as_ref()),
        dist = faer_to_r_matrix(dist_mat.as_ref()),
        dist_metric = dist_metric
    ))
}

/////////////////////
// Parametric UMAP //
/////////////////////

/// Parametric UMAP implementation
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Trains a neural network encoder to learn a mapping from the input space to a
/// low-dimensional embedding that preserves the UMAP graph structure. Supports
/// both GPU (wgpu) and CPU (burn flex CPU) backends. For small to medium data
/// sets (fewer than ~10k samples or narrow hidden layers), the CPU backend is
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
/// @param verbose Integer. `0L` - quiet; `1L` - normal verbosity; `2L` -
/// detailed verbosity.
/// @param use_gpu Logical. If \code{TRUE}, trains on the wgpu backend. If
/// \code{FALSE}, trains on the CPU via flex. Defaults to \code{TRUE}.
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
    verbose: usize,
    use_gpu: bool,
) -> Result<List, extendr_api::Error> {
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
        )?;

        Ok(list!(
            embedding = faer_to_r_matrix(res.as_ref()),
            model = ExternalPtr::new(PUmapModel::Gpu(model))
        ))
    } else {
        let device = FlexDevice;
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
        )?;

        Ok(list!(
            embedding = faer_to_r_matrix(res.as_ref()),
            model = ExternalPtr::new(PUmapModel::Cpu(model))
        ))
    }
}

/// Predict new data using a trained parametric UMAP model
///
/// @description
/// `r lifecycle::badge("experimental")`
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

/// Takes in a parametric UMAP and serialises it to raw bytes
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Serialises a trained parametric UMAP model to bytes for saving the data.
///
/// @param model Robject. The trained parametric UMAP model to serialise.
///
/// @returns The raw bytes of the model
///
/// @keywords internal
#[extendr]
fn rs_serialise_parametric_umap(model: Robj) -> Result<Vec<u8>, extendr_api::Error> {
    let model: ExternalPtr<PUmapModel> = model.try_into()?;
    let (tag, mut bytes) = match &*model {
        PUmapModel::Gpu(m) => (
            0u8,
            m.to_bytes()
                .map_err(|e| extendr_api::Error::Other(e.to_string()))?,
        ),
        PUmapModel::Cpu(m) => (
            1u8,
            m.to_bytes()
                .map_err(|e| extendr_api::Error::Other(e.to_string()))?,
        ),
    };
    let mut out = Vec::with_capacity(bytes.len() + 1);
    out.push(tag);
    out.append(&mut bytes);
    Ok(out)
}

/// Deserialises raw bytes to a trained UMAP model.
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Deserialises a trained parametric UMAP model from raw bytes and returns an R
/// object.
///
/// @param bytes The raw byte sequence. The leading byte tags the backend the
/// model was trained on: `0` for wgpu, `1` for the flex CPU backend.
///
/// @returns An external pointer to the restored model, for use with
/// `rs_parametric_umap_predict`.
///
/// @keywords internal
#[extendr]
fn rs_deserialise_parametric_umap(bytes: Vec<u8>) -> Result<Robj, extendr_api::Error> {
    if bytes.is_empty() {
        return Err(extendr_api::Error::Other("empty payload".into()));
    }
    let (tag, payload) = (bytes[0], &bytes[1..]);
    let model = match tag {
        0 => PUmapModel::Gpu(
            TrainedUmapModel::from_bytes(payload, WgpuDevice::default())
                .map_err(|e| extendr_api::Error::Other(e.to_string()))?,
        ),
        1 => PUmapModel::Cpu(
            TrainedUmapModel::from_bytes(payload, FlexDevice)
                .map_err(|e| extendr_api::Error::Other(e.to_string()))?,
        ),
        t => {
            return Err(extendr_api::Error::Other(format!(
                "unknown backend tag: {}",
                t
            )))
        }
    };
    Ok(ExternalPtr::new(model).into())
}

///////////////////
// K-means (GPU) //
///////////////////

/// GPU-accelerated k-means
///
/// @description
/// `r lifecycle::badge("experimental")`
/// A GPU-accelerated k-means version leveraging the wgpu backend via cubecl.
///
/// @param data Numeric matrix. Samples x features.
/// @param dist String. Distance metric to use. One of
/// `c("euclidean", "cosine")`.
/// @param n_centroids Integer. Number of clusters, centroids to identify.
/// @param kmeans_params Named list. Contains specific parameters for the GPU-
/// accelerated k-means.
/// @param seed Integer. Seed for reproducibility.
/// @param verbose Boolean. Controls verbosity of the function.
///
/// @returns A list with
/// \itemize{
///   \item centoids - The centroids matrix (centroids x features)
///   \item assignments - The cluster assignments of the data. (1-indexed.)
/// }
///
/// @export
#[extendr]
fn rs_kmeans_gpu(
    data: RMatrix<f64>,
    dist: &str,
    n_centroids: usize,
    kmeans_params: List,
    seed: usize,
    verbose: bool,
) -> Result<List, extendr_api::Error> {
    let data = r_matrix_to_faer_fp32(&data);

    let kmeans_params = KMeansGpuParams::from_r_list(kmeans_params)?;

    let (centroids, assignments) = kmeans_gpu_internal(
        data.as_ref(),
        dist,
        n_centroids,
        Some(kmeans_params),
        seed,
        verbose,
    )?;

    Ok(list!(
        centroids = faer_to_r_matrix(centroids.as_ref()),
        assignments = assignments.r_int_convert_shift()
    ))
}

/////////////
// Cor/Cov //
/////////////

/// GPU-accelerated correlation calculations
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU-accelerated pairwise column correlations. Has the options of Pearson and
/// Spearman correlation coefficient calculations.
///
/// @param x Numerical matrix. The matrix for which to calculate the column
/// pairwise correlation matrix.
/// @param spearman Boolean. Shall the Spearman correlation be calculated
/// instead of Pearson.
/// @param verbose Boolean. Controls verbosity of the function.
///
/// @returns The correlation matrix
///
/// @export
#[extendr]
fn rs_cor_gpu(
    x: RMatrix<f64>,
    spearman: bool,
    verbose: bool,
) -> Result<RMatrix<f64>, extendr_api::Error> {
    let data = r_matrix_to_faer_fp32(&x);
    let device: WgpuDevice = Default::default();

    let cor_type = if spearman {
        GpuCorCov::Spearman
    } else {
        GpuCorCov::Pearson
    };

    let mut res = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        cor_type,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    // set the diagonals to 1
    for i in 0..res.nrows() {
        for j in 0..res.ncols() {
            if i == j {
                res[(i, j)] = 1.0;
            }
        }
    }

    Ok(faer_to_r_matrix(res.as_ref()))
}

/// GPU-accelerated covariance calculations
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU-accelerated pairwise column co-variance calculation.
///
/// @param x Numerical matrix. The matrix for which to calculate the column
/// pairwise covariance matrix.
/// @param verbose Boolean. Controls verbosity of the function.
///
/// @returns The covariance matrix
///
/// @export
#[extendr]
fn rs_cov_gpu(x: RMatrix<f64>, verbose: bool) -> Result<RMatrix<f64>, extendr_api::Error> {
    let data = r_matrix_to_faer_fp32(&x);
    let device: WgpuDevice = Default::default();

    let res = column_pairwise_cor_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        GpuCorCov::Covariance,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    Ok(faer_to_r_matrix(res.as_ref()))
}

//////////////////////////
// GPU-accelerated UMAP //
//////////////////////////

/// UMAP implementation
///
/// @description
/// `r lifecycle::badge("experimental")`
/// This is the wrapper function into the Rust interface for GPU-accelerated
/// UMAP. Leverages GPU acceleration in the kNN search and also in the
/// optimisation.
///
/// @param embd Numerical matrix. The data to use to generate the embeddings.
/// Should be of dimensions samples x features.
/// @param n_dim Integer. Number of UMAP dimensions to return.
/// @param min_dist Numeric. Minimum distance to use.
/// @param spread Numeric. Spread parameter to use.
/// @param k Integer. Number of nearest neighbours to consider
/// @param umap_params Named list. List that contains all of the key parameters
/// for the UMAP generation.
/// @param seed Integer. Seed for reproducibility.
/// @param use_high_precision Optional logical. Controls `fp32` vs `fp64` for.
/// If `NULL` will use sensible default thresholding.
/// @param verbose Integer. If `0L` -> silent or `1L` for normal verbosity; `2L`
/// for detailed verbosity.
///
/// @return The UMAP embeddings.
///
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_umap_gpu(
    embd: RMatrix<f64>,
    n_dim: usize,
    min_dist: f64,
    spread: f64,
    k: usize,
    umap_params: List,
    seed: usize,
    use_high_precision: Nullable<Rbool>,
    verbose: usize,
) -> extendr_api::Result<RMatrix<f64>> {
    let verbosity = bixverse_rs::prelude::parse_verbosity_level(verbose);
    let precision = parse_precision(use_high_precision, embd.nrows());

    match precision {
        FloatingPointPrecision::FP32 => {
            if verbosity.detailed_verbosity() {
                println!("Lower precision (fp32) path chosen.")
            }

            let embd = r_matrix_to_faer_fp32(&embd);

            let res = umap_manifold_gpu::<f32, WgpuRuntime>(
                embd.as_ref(),
                None,
                n_dim,
                k,
                min_dist,
                spread,
                umap_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
        FloatingPointPrecision::FP64 => {
            if verbosity.detailed_verbosity() {
                println!("Higher precision (fp64) path chosen.")
            }

            let embd = r_matrix_to_faer(&embd);

            let res = umap_manifold_gpu::<f64, WgpuRuntime>(
                embd.as_ref(),
                None,
                n_dim,
                k,
                min_dist,
                spread,
                umap_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
    }
}

/// UMAP implementation from a pre-computed kNN graph
///
/// @description
/// `r lifecycle::badge("experimental")`
/// This is the wrapper function into the Rust interface for UMAP and takes a
/// pre-computed kNN graph, skipping the graph build. GPU acceleration applies
/// to the optimisation.
///
/// @param embd Numerical matrix. The data to use to generate the embeddings.
/// Should be of dimensions samples x features.
/// @param knn_data `NearestNeighbours` class from R.
/// @param n_dim Integer. Number of UMAP dimensions to return.
/// @param min_dist Numeric. Minimum distance to use.
/// @param spread Numeric. Spread parameter to use.
/// @param k Integer. Number of nearest neighbours to consider
/// @param umap_params Named list. List that contains all of the key parameters
/// for the UMAP generation.
/// @param seed Integer. Seed for reproducibility.
/// @param use_high_precision Optional logical. Controls `fp32` vs `fp64` for.
/// If `NULL` will use sensible default thresholding.
/// @param verbose Integer. If `0L` -> silent or `1L` for normal verbosity; `2L`
/// for detailed verbosity.
///
/// @return The UMAP embeddings.
///
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_umap_from_knn_gpu(
    embd: RMatrix<f64>,
    knn_data: List,
    n_dim: usize,
    min_dist: f64,
    spread: f64,
    k: usize,
    umap_params: List,
    seed: usize,
    use_high_precision: Nullable<Rbool>,
    verbose: usize,
) -> extendr_api::Result<RMatrix<f64>> {
    let verbosity = bixverse_rs::prelude::parse_verbosity_level(verbose);
    let precision = parse_precision(use_high_precision, embd.nrows());

    match precision {
        FloatingPointPrecision::FP32 => {
            if verbosity.detailed_verbosity() {
                println!("Lower precision (fp32) path chosen.")
            }

            let embd = r_matrix_to_faer_fp32(&embd);

            let knn = nearest_neighbours_to_rust(knn_data);

            let res = umap_manifold_gpu::<f32, WgpuRuntime>(
                embd.as_ref(),
                knn,
                n_dim,
                k,
                min_dist,
                spread,
                umap_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
        FloatingPointPrecision::FP64 => {
            if verbosity.detailed_verbosity() {
                println!("Higher precision (fp64) path chosen.")
            }

            let embd = r_matrix_to_faer(&embd);

            let knn = nearest_neighbours_to_rust(knn_data);

            let res = umap_manifold_gpu::<f64, WgpuRuntime>(
                embd.as_ref(),
                knn,
                n_dim,
                k,
                min_dist,
                spread,
                umap_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
    }
}

//////////
// tSNE //
//////////

/// tSNE implementation
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Wraps the tSNE implementation in manifolds-rs. You have two optimiser
/// options: `"bh"`, which tends to be faster on smaller data sets, and `"fft"`
/// for large data sets. The kNN search runs on the GPU; the optimiser stays on
/// the CPU.
///
/// @param embd Numerical matrix. The data to use to generate the embeddings.
/// Should be of dimensions samples x features.
/// @param n_dim Integer. Number of tSNE dimensions to return. Needs to be two,
/// others are not supported.
/// @param perplexity Numeric. The tSNE perplexity parameter.
/// @param approx_type String. One of `c("fft", "bh")`. Which of the two
/// approximations to use.
/// @param tsne_params Named list. List that contains all of the key parameters
/// for the tSNE generation.
/// @param seed Integer. Seed for reproducibility.
/// @param use_high_precision Optional logical. Controls `fp32` vs `fp64` for.
/// If `NULL` will use sensible default thresholding.
/// @param verbose Integer. If `0L` -> silent or `1L` for normal verbosity; `2L`
/// for detailed verbosity.
///
/// @return The tSNE embeddings.
///
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_tsne_gpu(
    embd: RMatrix<f64>,
    n_dim: usize,
    perplexity: f64,
    approx_type: String,
    tsne_params: List,
    seed: usize,
    use_high_precision: Nullable<Rbool>,
    verbose: usize,
) -> extendr_api::Result<RMatrix<f64>> {
    let verbosity = bixverse_rs::prelude::parse_verbosity_level(verbose);
    let precision = parse_precision(use_high_precision, embd.nrows());

    match precision {
        FloatingPointPrecision::FP64 => {
            if verbosity.detailed_verbosity() {
                println!("Larger data set. Using high precision floats.")
            }

            let embd = r_matrix_to_faer(&embd);

            let res = tsne_manifold_gpu(
                embd.as_ref(),
                None,
                n_dim,
                &approx_type,
                perplexity,
                tsne_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
        FloatingPointPrecision::FP32 => {
            if verbosity.detailed_verbosity() {
                println!("Lower precision (fp32) path chosen.")
            }
            let embd = r_matrix_to_faer_fp32(&embd);

            let res = tsne_manifold_gpu(
                embd.as_ref(),
                None,
                n_dim,
                &approx_type,
                perplexity as f32,
                tsne_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
    }
}

/// tSNE implementation from a pre-computed kNN graph
///
/// @description
/// `r lifecycle::badge("experimental")`
/// Wraps the tSNE implementation in manifolds-rs. You have two optimiser
/// options: `"bh"`, which tends to be faster on smaller data sets, and `"fft"`
/// for large data sets. This version takes a pre-computed kNN graph, please see
/// [new_nearest_neighbour()].
///
/// @param embd Numerical matrix. The data to use to generate the embeddings.
/// Should be of dimensions samples x features.
/// @param knn_data `NearestNeighbours` class from R.
/// @param n_dim Integer. Number of tSNE dimensions to return. Needs to be two,
/// others are not supported.
/// @param perplexity Numeric. The tSNE perplexity parameter.
/// @param approx_type String. One of `c("fft", "bh")`. Which of the two
/// approximations to use.
/// @param tsne_params Named list. List that contains all of the key parameters
/// for the tSNE generation.
/// @param seed Integer. Seed for reproducibility.
/// @param use_high_precision Optional logical. Controls `fp32` vs `fp64` for.
/// If `NULL` will use sensible default thresholding.
/// @param verbose Integer. If `0L` -> silent or `1L` for normal verbosity; `2L`
/// for detailed verbosity.
///
/// @return The tSNE embeddings.
///
/// @export
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_tsne_from_knn_gpu(
    embd: RMatrix<f64>,
    knn_data: List,
    n_dim: usize,
    perplexity: f64,
    approx_type: String,
    tsne_params: List,
    seed: usize,
    use_high_precision: Nullable<Rbool>,
    verbose: usize,
) -> extendr_api::Result<RMatrix<f64>> {
    let verbosity = bixverse_rs::prelude::parse_verbosity_level(verbose);
    let precision = parse_precision(use_high_precision, embd.nrows());

    match precision {
        FloatingPointPrecision::FP32 => {
            if verbosity.detailed_verbosity() {
                println!("Lower precision (fp32) path chosen.")
            }

            let embd = r_matrix_to_faer_fp32(&embd);

            let knn = nearest_neighbours_to_rust(knn_data);

            let res = tsne_manifold_gpu(
                embd.as_ref(),
                knn,
                n_dim,
                &approx_type,
                perplexity as f32,
                tsne_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
        FloatingPointPrecision::FP64 => {
            if verbosity.detailed_verbosity() {
                println!("Large data set. Using higher precision floats.")
            }

            let embd = r_matrix_to_faer(&embd);

            let knn = nearest_neighbours_to_rust(knn_data);

            let res = tsne_manifold_gpu(
                embd.as_ref(),
                knn,
                n_dim,
                &approx_type,
                perplexity,
                tsne_params,
                seed,
                verbose,
            )
            .to_extendr()?;

            Ok(faer_to_r_matrix(res.as_ref()))
        }
    }
}
