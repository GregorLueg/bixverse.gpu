use extendr_api::List;

use ann_search_rs::prelude::*;
use ann_search_rs::*;
use cubecl::wgpu::WgpuDevice;
use cubecl::wgpu::WgpuRuntime;
use cubecl::Runtime;
use faer::MatRef;
use std::time::Instant;

/////////////
// Helpers //
/////////////

/// Helper to remove self
///
/// ### Params
///
/// * `indices` - Mutable version of the kNN indices
/// * `distances` - Option of the distances if returned
///
/// ### Returns
///
/// Tuple of (indices, Option<distances>) with self returned
fn remove_self(
    mut indices: Vec<Vec<usize>>,
    distances: Option<Vec<Vec<f32>>>,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    for idx_vec in indices.iter_mut() {
        idx_vec.remove(0);
    }
    let distances = distances.map(|mut dists| {
        for dist_vec in dists.iter_mut() {
            dist_vec.remove(0);
        }
        dists
    });
    (indices, distances)
}

///////////
// CAGRA //
///////////

////////////
// Params //
////////////

/// The parameters for the CAGRA-style kNN search
pub struct CagraParams {
    /// Number of neighbours to identify
    pub k_query: usize,
    /// Distance metric to use. One of `"euclidean"` or `"cosine"`.
    pub ann_dist: String,
    /// Final node degree of the CAGRA graph during index generation. If None,
    /// defaults to `30`.
    pub k: Option<usize>,
    /// Optional build k for the NNDescent iterations prior to CAGRA pruning.
    /// Defaults to `2 * k` if not provided.
    pub k_build: Option<usize>,
    /// Number of refinement sweeps during the generation of the
    pub refine_sweeps: usize,
    /// Maximum iterations for the NNDescent rounds
    pub max_iters: Option<usize>,
    /// Optional number of trees to use in the initial GPU-accelerated forest
    pub n_trees: Option<usize>,
    /// Termination criterium for the NNDescent iterations
    pub delta: f32,
    /// Optional sampling rate during NNDescent iterations.
    pub rho: Option<f32>,
    /// Beam width during querying
    pub beam_width: Option<usize>,
    /// Maximum beam iterations
    pub max_beam_iters: Option<usize>,
    /// Number of entry points into the graph
    pub n_entry_points: Option<usize>,
}

impl CagraParams {
    /// Generate the CagraParams params from an R list or default to sensible
    /// parameters
    ///
    /// ### Params
    ///
    /// * `r_list` - R list with the parameters
    ///
    /// ### Returns
    ///
    /// Self with the specified parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let cagra = r_list.into_hashmap();

        let k_query = cagra
            .get("k_query")
            .and_then(|v| v.as_integer())
            .unwrap_or(15) as usize;

        let k = cagra
            .get("k")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let k_build = cagra
            .get("k_build")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let ann_dist = std::string::String::from(
            cagra
                .get("ann_dist")
                .and_then(|v| v.as_str())
                .unwrap_or("cosine"),
        );

        let refine_sweeps = cagra
            .get("refine_sweeps")
            .and_then(|v| v.as_integer())
            .unwrap_or(1) as usize;

        let max_iters = cagra
            .get("max_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let n_trees = cagra
            .get("n_trees")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let delta = cagra
            .get("delta")
            .and_then(|v| v.as_real())
            .unwrap_or(0.001) as f32;

        let rho = cagra.get("rho").and_then(|v| v.as_real()).map(|v| v as f32);

        let beam_width = cagra
            .get("beam_width")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let max_beam_iters = cagra
            .get("max_beam_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let n_entry_points = cagra
            .get("n_entry_points")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        Self {
            k_query,
            ann_dist,
            k,
            k_build,
            refine_sweeps,
            max_iters,
            n_trees,
            delta,
            rho,
            beam_width,
            max_beam_iters,
            n_entry_points,
        }
    }
}

//////////
// Main //
//////////

/// Use CAGRA-style GPU-accelerated kNN search.
///
/// Leverages the CAGRA style algorithm to generate a kNN graph from the data.
/// You have two options:
///
/// - Extract the graph directly after the NNDescent iterations and potential
///   refinement. Usually slightly lower precision, but faster.
/// - Run the beam search over the generated, pruned CAGRA graph for high
///   precison.
///
/// The algorithm runs on the wgpu backend.
///
/// ### Params
///
/// * `embd` - The embedding matrix to use to approximate neighbours and
///   calculate distances. Cells x features.
/// * `cagra_params` - Structure with the parameters for the CAGRA-style kNN
///   search.
/// * `return_dist` - Return the distances.
/// * `extract_knn` - Do you wish to use the fast extraction method.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Tuple of `(indices of nearest neighbours, distances to these neighbours)`
pub fn cagra_knn_with_dist(
    embd: MatRef<f32>,
    cagra_params: &CagraParams,
    return_dist: bool,
    extract_knn: bool,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    let start = Instant::now();
    let device: WgpuDevice = Default::default();

    if verbose {
        println!("Starting to generate the CAGRA index.")
    }

    let mut cagra_idx = build_nndescent_index_gpu::<f32, WgpuRuntime>(
        embd.as_ref(),
        &cagra_params.ann_dist,
        cagra_params.k,
        cagra_params.k_build,
        cagra_params.max_iters,
        cagra_params.n_trees,
        Some(cagra_params.delta),
        cagra_params.rho,
        Some(cagra_params.refine_sweeps),
        seed,
        verbose,
        true,
        device.clone(),
    );

    if verbose {
        println!("Generated the CAGRA index in {:.2?}.", start.elapsed());
    }

    let (indices, distances) = if extract_knn {
        if verbose {
            println!("Extracting the generated kNN graph directly.")
        }
        let (n, d) = extract_nndescent_knn_gpu(&cagra_idx, return_dist);
        if verbose {
            println!(" Extraction done in {:.2?}.", start.elapsed())
        }
        (n, d)
    } else {
        if verbose {
            println!("Generating the kNN graph via beam search.")
        }
        let search_params = CagraGpuSearchParams::new(
            cagra_params.beam_width,
            cagra_params.max_beam_iters,
            cagra_params.n_entry_points,
        );

        let (n, d) = query_nndescent_index_gpu_self(
            &mut cagra_idx,
            cagra_params.k_query + 1, // because of self
            Some(search_params),
            return_dist,
        );
        if verbose {
            println!(" Beam search done in {:.2?}.", start.elapsed())
        }
        (n, d)
    };

    // manual drop here
    drop(cagra_idx);

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    remove_self(indices, distances)
}

/////////////
// IVF GPU //
/////////////

////////////
// Params //
////////////

/// Parameters for the IVF-GPU kNN search
pub struct IvfGpuParams {
    /// Number of neighbours to identify
    pub k: usize,
    /// Distance metric to use. One of `"euclidean"` or `"cosine"`.
    pub ann_dist: String,
    /// Number of clusters (lists) to partition the index into. Defaults to √n.
    pub nlist: Option<usize>,
    /// Number of clusters to probe at query time. Defaults to √nlist.
    pub nprobe: Option<usize>,
    /// Number of query vectors processed per GPU batch. Defaults to 100,000.
    pub nquery: Option<usize>,
    /// Maximum k-means iterations during index build. Defaults to 30.
    pub max_iters: Option<usize>,
}

impl IvfGpuParams {
    /// Construct `IvfGpuParams` from an R list, falling back to sensible
    /// defaults for any missing fields.
    ///
    /// ### Params
    ///
    /// * `r_list` - R list with the parameters
    ///
    /// ### Returns
    ///
    /// `Self` with the specified parameters.
    pub fn from_r_list(r_list: List) -> Self {
        let ivf = r_list.into_hashmap();

        let k = ivf.get("k").and_then(|v| v.as_integer()).unwrap_or(15) as usize;

        let ann_dist = std::string::String::from(
            ivf.get("ann_dist")
                .and_then(|v| v.as_str())
                .unwrap_or("cosine"),
        );

        let nlist = ivf
            .get("nlist")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let nprobe = ivf
            .get("nprobe")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let nquery = ivf
            .get("nquery")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        let max_iters = ivf
            .get("max_iters")
            .and_then(|v| v.as_integer())
            .map(|v| v as usize);

        Self {
            k,
            ann_dist,
            nlist,
            nprobe,
            nquery,
            max_iters,
        }
    }
}

//////////
// Main //
//////////

/// Use IVF-GPU-accelerated kNN self-search.
///
/// Builds an IVF index over the provided embedding matrix and queries each
/// vector against the index to produce a kNN graph. The closest neighbour
/// (the vector itself) is removed from every result row before returning.
///
/// The algorithm runs on the wgpu backend.
///
/// ### Params
///
/// * `embd` - Embedding matrix to build the index from. Cells x features.
/// * `ivf_params` - Parameters for the IVF-GPU index and query.
/// * `return_dist` - Whether to return distances alongside indices.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Tuple of `(indices of nearest neighbours, distances to these neighbours)`
pub fn gpu_ivf_knn_with_dist(
    embd: MatRef<f32>,
    ivf_params: &IvfGpuParams,
    return_dist: bool,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    let start = Instant::now();
    let device: WgpuDevice = Default::default();

    if verbose {
        println!("Building IVF-GPU index.");
    }

    let ivf_idx = build_ivf_index_gpu::<f32, WgpuRuntime>(
        embd,
        ivf_params.nlist,
        ivf_params.max_iters,
        &ivf_params.ann_dist,
        seed,
        verbose,
        device.clone(),
    );

    if verbose {
        println!("Built IVF-GPU index in {:.2?}.", start.elapsed());
    }

    let (indices, distances) = query_ivf_index_gpu_self(
        &ivf_idx,
        ivf_params.k + 1,
        ivf_params.nprobe,
        ivf_params.nquery,
        return_dist,
        verbose,
    );

    if verbose {
        println!("Self-query done in {:.2?}.", start.elapsed());
    }

    drop(ivf_idx);

    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    remove_self(indices, distances)
}

////////////////////////
// Exhaustive GPU kNN //
////////////////////////

/// Exhaustive GPU kNN search
///
/// The algorithm runs on the wgpu backend.
///
/// ### Params
///
/// * `embd` - Embedding matrix to build the index from. Cells x features.
/// * `return_dist` - Whether to return distances alongside indices.
/// * `verbose` - Controls verbosity of the function.
///
/// ### Returns
///
/// Tuple of `(indices of nearest neighbours, distances to these neighbours)`
pub fn gpu_exhaustive_knn_with_dist(
    embd: MatRef<f32>,
    k: usize,
    dist_metric: &str,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
    let start = Instant::now();
    let device: WgpuDevice = Default::default();

    if verbose {
        println!("Building Exhaustive-GPU index.");
    }

    let idx = build_exhaustive_index_gpu::<f32, WgpuRuntime>(embd, dist_metric, device.clone());

    if verbose {
        println!("Built IVF-GPU index in {:.2?}.", start.elapsed());
    }

    let (indices, distances) = query_exhaustive_index_gpu_self(&idx, k + 1, return_dist, verbose);

    if verbose {
        println!("Self-query done in {:.2?}.", start.elapsed());
    }

    drop(idx);

    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    remove_self(indices, distances)
}
