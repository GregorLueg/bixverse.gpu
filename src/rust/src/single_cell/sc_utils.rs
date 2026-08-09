//! Shared R-bridge helpers for the single cell GPU entry points.
//!
//! These are ports of helpers that live in the `bixverse` R package's own crate
//! (`single_cell/r_sc_metacells.rs` and `single_cell/utils.rs`) rather than in
//! `bixverse-rs`, so this crate cannot import them. Keep them in sync with the
//! CPU originals: a divergence here means the GPU and CPU paths silently
//! aggregate different cells, or the fast clustering results come back in a
//! shape the R side does not expect.

use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::sc_analysis::fast_clusters::{
    FastLouvainGridResult, FastLouvainResults,
};
use extendr_api::*;
use faer::Mat;
use std::collections::HashMap;

///////////
// Types //
///////////

/// Results data from the neighbours class.
///
/// ### Fields
///
/// * `0` - The kNN indices
/// * `1` - The kNN distances
/// * `2` - Number of neighbours
/// * `3` - Distance metric
pub type NeighboursData = Result<(Vec<Vec<usize>>, Vec<Vec<f32>>, usize, String)>;

///////////////////
// Index mapping //
///////////////////

/// Resolved mapping between working space and the original count file.
///
/// Two index spaces meet in the meta cell entry points and conflating them
/// silently aggregates the wrong cells. *Working space* is the row space of the
/// embedding or kNN graph handed over from R, holding one row per cell that
/// passed QC. *Original space* is the row space of `counts_cells.bin`, which is
/// what `aggregate_meta_cells` indexes into and what R reports back as
/// `original_cell_idx`. The two coincide only when every cell in the file
/// passed QC, so memberships are always translated through this mapping before
/// they are aggregated or returned.
pub struct CellMapping {
    /// Working rows to retain, in output order. `None` means every working row
    /// is used and the embedding needs no subsetting.
    pub rows_to_use: Option<Vec<usize>>,
    /// Maps a working row onto its row in the original count file.
    pub subset_to_orig: Vec<usize>,
    /// Number of cells in the original count file, i.e. the size of the space
    /// `subset_to_orig` points into.
    pub n_total: usize,
}

impl CellMapping {
    /// Whether the incoming embedding or kNN graph has to be subset before use.
    ///
    /// ### Returns
    ///
    /// `true` when a narrowed set of cells was requested, which also implies the
    /// kNN graph has to be rebuilt on those cells.
    pub fn needs_subsetting(&self) -> bool {
        self.rows_to_use.is_some()
    }
}

/// Work out how working-space rows map back onto the original count file.
///
/// Resolves the identity case (no QC filtering), the QC-filtered case, and the
/// narrowed case where only a subset of the QC-passing cells is requested.
/// Requested cells that are absent from `cells_to_keep` are dropped silently,
/// since the caller may legitimately ask for cells that failed QC.
///
/// ### Params
///
/// * `cells_to_keep` - Original row indices the embedding was built from, in
///   embedding row order, 0-indexed. `None` assumes the embedding covers the
///   count file one-to-one.
/// * `cells_to_use` - Optional further narrowing, given in original space and
///   0-indexed. `None` keeps every working row.
/// * `n_working_rows` - Number of embedding or kNN rows supplied by R.
/// * `n_total` - Number of cells in the count file.
///
/// ### Returns
///
/// A [`CellMapping`], or an error when `cells_to_keep` disagrees with
/// `n_working_rows` or when none of the requested cells survive the mapping.
pub fn resolve_cell_mapping(
    cells_to_keep: Option<&[i32]>,
    cells_to_use: Option<&[i32]>,
    n_working_rows: usize,
    n_total: usize,
) -> Result<CellMapping> {
    // Working row i corresponds to cells_to_keep[i]. Without it we can only
    // assume the embedding covers the file one-to-one.
    let keep: Vec<usize> = match cells_to_keep {
        Some(keep) => {
            if keep.len() != n_working_rows {
                return Err(format!(
                    "'cells_to_keep' has {} entries but the embedding/kNN has {} rows",
                    keep.len(),
                    n_working_rows
                )
                .into());
            }
            keep.iter().map(|&x| x as usize).collect()
        }
        None => (0..n_working_rows).collect(),
    };

    match cells_to_use {
        Some(use_cells) => {
            let orig_to_working: HashMap<usize, usize> = keep
                .iter()
                .enumerate()
                .map(|(working_row, &orig_idx)| (orig_idx, working_row))
                .collect();

            let mut rows_to_use = Vec::with_capacity(use_cells.len());
            let mut subset_to_orig = Vec::with_capacity(use_cells.len());

            for &orig_idx in use_cells {
                let orig_idx = orig_idx as usize;
                if let Some(&working_row) = orig_to_working.get(&orig_idx) {
                    rows_to_use.push(working_row);
                    subset_to_orig.push(orig_idx);
                }
            }

            if subset_to_orig.is_empty() {
                return Err(
                    "None of the requested 'cells_to_use' are part of 'cells_to_keep'".into(),
                );
            }

            Ok(CellMapping {
                rows_to_use: Some(rows_to_use),
                subset_to_orig,
                n_total,
            })
        }
        // Still a remap whenever cells_to_keep is not the identity.
        None => Ok(CellMapping {
            rows_to_use: None,
            subset_to_orig: keep,
            n_total,
        }),
    }
}

/// Gather selected rows of an R embedding matrix into a faer matrix.
///
/// R matrices are column-major, so the source element for row `i`, column `j`
/// sits at `rows_to_use[i] + j * nrow`. Downcast to `f32` matches the precision
/// the meta cell algorithms work in; the embedding is a PCA projection, so the
/// dynamic range is small enough for this to be safe.
///
/// ### Params
///
/// * `embd` - Column-major embedding matrix as handed over from R, with one row
///   per working-space cell.
/// * `rows_to_use` - Working row indices to gather, in output order.
///
/// ### Returns
///
/// A `rows_to_use.len()` by `embd.ncols()` matrix holding the selected rows.
pub fn subset_embedding(embd: &RMatrix<f64>, rows_to_use: &[usize]) -> Mat<f32> {
    let ncol = embd.ncols();
    let nrow = embd.nrows();
    let data = embd.data();

    Mat::from_fn(rows_to_use.len(), ncol, |i, j| {
        data[rows_to_use[i] + j * nrow] as f32
    })
}

/////////
// kNN //
/////////

/// Process R kNN indices to the Rust variant.
///
/// ### Params
///
/// * `knn_mat` - Samples x indices of the k-nearest neighbours (0-indexed!)
///
/// ### Returns
///
/// A `Vec<Vec<usize>>`, one entry per cell.
pub fn knn_indices_processing(knn_mat: RMatrix<i32>) -> Vec<Vec<usize>> {
    let nrow = knn_mat.nrows();
    let ncol = knn_mat.ncols();
    let data = knn_mat.data();

    let mut out: Vec<Vec<usize>> = (0..nrow).map(|_| Vec::with_capacity(ncol)).collect();
    for i in 0..ncol {
        let col_offset = i * nrow;
        for j in 0..nrow {
            out[j].push(data[col_offset + j] as usize);
        }
    }
    out
}

/// Process R kNN distances to the Rust variant.
///
/// ### Params
///
/// * `knn_dist` - Samples x distances of the k-nearest neighbours
///
/// ### Returns
///
/// A `Vec<Vec<f32>>`, one entry per cell.
pub fn knn_distances_processing(knn_dist: RMatrix<f64>) -> Vec<Vec<f32>> {
    let nrow = knn_dist.nrows();
    let ncol = knn_dist.ncols();
    let data = knn_dist.data();

    let mut out: Vec<Vec<f32>> = (0..nrow).map(|_| Vec::with_capacity(ncol)).collect();
    for i in 0..ncol {
        let col_offset = i * nrow;
        for j in 0..nrow {
            out[j].push(data[col_offset + j] as f32);
        }
    }
    out
}

/// Transform R kNN data to Rust data.
///
/// ### Params
///
/// * `knn_data` - R list with the kNN data, holding `indices`, `dist`,
///   `dist_metric` and `k`.
///
/// ### Returns
///
/// The [NeighboursData] or an error when a field is missing or has the wrong
/// type.
pub fn knn_data_to_rust(knn_data: List) -> NeighboursData {
    let data: HashMap<&str, Robj> = knn_data.try_into()?;

    let knn_indices: RArray<i32, 2> = data
        .get("indices")
        .ok_or_else(|| Error::Other("missing 'indices'".into()))?
        .as_matrix()
        .ok_or_else(|| Error::Other("'indices' is not a matrix".into()))?;

    let knn_dist: RArray<f64, 2> = data
        .get("dist")
        .ok_or_else(|| Error::Other("missing 'dist'".into()))?
        .as_matrix()
        .ok_or_else(|| Error::Other("'dist' is not a matrix".into()))?;

    let dist_metric = data
        .get("dist_metric")
        .ok_or_else(|| Error::Other("missing 'dist_metric'".into()))?
        .as_str()
        .ok_or_else(|| Error::Other("'dist_metric' is not a string".into()))?
        .to_string();

    let k = data
        .get("k")
        .ok_or_else(|| Error::Other("missing 'k'".into()))?
        .as_integer()
        .ok_or_else(|| Error::Other("'k' is not an integer".into()))? as usize;

    let knn_indices = knn_indices_processing(knn_indices);
    let knn_dist = knn_distances_processing(knn_dist);

    Ok((knn_indices, knn_dist, k, dist_metric))
}

/////////////////
// FastCluster //
/////////////////

/// Type for the fast clustering single results
///
/// ### Fields
///
/// * `0` - The Louvain membership per resolution
/// * `1` - Optional k-means cluster membership
/// * `2` - Optional k-means centroids
pub type FastClusterSingle = Result<(Vec<Vec<usize>>, Option<Vec<i32>>, Option<RMatrix<f64>>)>;

/// Type for the fast clustering grid results
///
/// ### Fields
///
/// * `0` - The [FastLouvainGridResult] per resolution
/// * `1` - Optional k-means cluster membership
/// * `2` - Optional k-means centroids
pub type FastClusterGrid = Result<(
    Vec<FastLouvainGridResult>,
    Option<Vec<i32>>,
    Option<RMatrix<f64>>,
)>;

/// Pull the optional k-means block out of a fast clustering result.
///
/// ### Params
///
/// * `res` - The [FastLouvainResults]
/// * `return_km` - Shall the k-means results be returned
///
/// ### Returns
///
/// `(k-means assignments, centroid matrix)`, both `None` when `return_km` is
/// `false`.
fn fast_cluster_kmeans_block(
    res: &FastLouvainResults,
    return_km: bool,
) -> Result<(Option<Vec<i32>>, Option<RMatrix<f64>>)> {
    if !return_km {
        return Ok((None, None));
    }

    let k_means_clusters = res.get_k_mean_clusters().to_extendr()?;
    let centroids = res.get_centroids().to_extendr()?;

    Ok((
        Some(k_means_clusters.r_int_convert()),
        Some(faer_to_r_matrix(centroids.as_ref())),
    ))
}

/// Helper function to extract the single-run results from the fast clustering
///
/// The variant is fixed by the driver that produced `res`, so the `Single`
/// match cannot fail by construction. It still returns an error rather than
/// panicking, in case the two ever drift apart.
///
/// ### Params
///
/// * `res` - The [FastLouvainResults]
/// * `return_km` - Shall the k-means results be returned
///
/// ### Returns
///
/// [FastClusterSingle]
pub fn fast_cluster_unwrap_single(res: FastLouvainResults, return_km: bool) -> FastClusterSingle {
    let (k_means_membership, centroids) = fast_cluster_kmeans_block(&res, return_km)?;

    let memberships = res
        .get_assignments()
        .left()
        .ok_or_else(|| Error::Other("expected the single fast clustering variant".into()))?;

    Ok((memberships, k_means_membership, centroids))
}

/// Helper function to extract the grid results from the fast clustering
///
/// Same by-construction invariant as [fast_cluster_unwrap_single], on the
/// `GridRes` variant.
///
/// ### Params
///
/// * `res` - The [FastLouvainResults]
/// * `return_km` - Shall the k-means results be returned
///
/// ### Returns
///
/// [FastClusterGrid]
pub fn fast_cluster_unwrap_multiple(res: FastLouvainResults, return_km: bool) -> FastClusterGrid {
    let (k_means_membership, centroids) = fast_cluster_kmeans_block(&res, return_km)?;

    let memberships = res
        .get_assignments()
        .right()
        .ok_or_else(|| Error::Other("expected the grid fast clustering variant".into()))?;

    Ok((memberships, k_means_membership, centroids))
}

/// Process the fast cluster Louvain grid results
///
/// ### Params
///
/// * `results` - Vector of [FastLouvainGridResult]
///
/// ### Returns
///
/// A list with `memberships` (the best labels per resolution) and `stats` (the
/// stability and conductance metrics per resolution, in the same order).
pub fn process_fc_louvain_results(results: Vec<FastLouvainGridResult>) -> Result<List> {
    let mut mean_ari: Vec<f32> = Vec::with_capacity(results.len());
    let mut median_ari: Vec<f32> = Vec::with_capacity(results.len());
    let mut mean_conductance: Vec<f32> = Vec::with_capacity(results.len());
    let mut median_conductance: Vec<f32> = Vec::with_capacity(results.len());
    let mut mean_n_comms: Vec<f32> = Vec::with_capacity(results.len());

    let mut res = List::new(results.len());

    for (index, louvain_res) in results.iter().enumerate() {
        let membership = louvain_res.best_labels.clone().r_int_convert();
        res.set_elt(index, Robj::from(membership))?;

        mean_ari.push(louvain_res.mean_ari);
        median_ari.push(louvain_res.median_ari);
        mean_conductance.push(louvain_res.mean_conductance);
        median_conductance.push(louvain_res.median_conductance);
        mean_n_comms.push(louvain_res.mean_n_communities);
    }

    let stats = list![
        mean_ari = mean_ari.r_float_convert(),
        median_ari = median_ari.r_float_convert(),
        mean_conductance = mean_conductance.r_float_convert(),
        median_conductance = median_conductance.r_float_convert(),
        mean_n_comms = mean_n_comms.r_float_convert(),
    ];

    Ok(list![memberships = res, stats = stats])
}
