//! GPU-accelerated SEACells.
//!
//! Mirrors `bixverse::rs_get_seacells` step for step. Both Frank-Wolfe solves,
//! the B-gradient argmin and the per-cell A columns, move to the device; kernel
//! construction, archetype initialisation, the `K²B` bookkeeping and the RSS
//! evaluation stay on the CPU inside `bixverse-rs`. Each solve falls back to its
//! CPU sibling for that iteration when no workgroup tier covers `k`. The
//! aggregation into meta cell pseudo-bulk is the same CPU reader-based path the
//! CPU wrapper uses.

use crate::ensure_gpu;
use bixverse_rs::gpu::sc_gpu::seacells_gpu::seacells_fit_gpu;
use bixverse_rs::prelude::*;
use bixverse_rs::single_cell::mc_generation::cell_aggregation_utils::{
    aggregate_meta_cells, remap_assignments_to_original, remap_metacells_to_original,
};
use bixverse_rs::single_cell::mc_generation::seacells::SEACellsParams;
use bixverse_rs::single_cell::sc_r_wrappers::assignments_to_r_list;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use extendr_api::*;
use std::time::Instant;

use crate::single_cell::sc_utils::{knn_data_to_rust, resolve_cell_mapping, subset_embedding};

/////////////
// extendr //
/////////////

extendr_module! {
    // module
    mod seacells_gpu;
    // functions
    fn rs_seacells_gpu;
}

//////////////
// SEACells //
//////////////

/// GPU: SEACells meta cell generation
///
/// @description
/// `r lifecycle::badge("experimental")`
/// GPU equivalent of `bixverse::rs_get_seacells`. Both Frank-Wolfe solves, the
/// B-gradient argmin and the per-cell A columns, are dispatched to the WGPU
/// backend. The kNN graph, the kernel matrix, the RSS evaluation and the
/// aggregation into pseudo-bulk counts all stay on the CPU.
///
/// @param f_path String. Path to the `counts_cells.bin` file.
/// @param embd Numeric matrix. Cells x components embedding, one row per
/// QC-passing cell.
/// @param cells_to_keep Optional integer vector. 0-indexed original row indices
/// the embedding was built from, in embedding row order.
/// @param cells_to_use Optional integer vector. 0-indexed original row indices
/// to narrow the run to. Forces a kNN rebuild on that subset.
/// @param knn_data Optional list. Precomputed kNN graph with `indices`, `dist`,
/// `dist_metric` and `k`. Ignored when `cells_to_use` is set.
/// @param seacells_params Named list. See [bixverse::params_sc_seacells()].
/// @param target_size Double. Library target size the meta cells are
/// normalised to.
/// @param seed Integer. Random seed.
/// @param verbose Integer. `0L` - quiet; `1L` - normal; `2L` - detailed.
///
/// @returns A list with the cell assignments, the aggregated meta cell counts
/// in compressed sparse form, the RSS history and the archetype cell indices.
///
/// @export
///
/// @references Persad, et al., Nat. Biotechnol., 2023.
///
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_seacells_gpu(
    f_path: String,
    embd: RMatrix<f64>,
    cells_to_keep: Option<Vec<i32>>,
    cells_to_use: Option<Vec<i32>>,
    knn_data: Nullable<List>,
    seacells_params: List,
    target_size: f64,
    seed: usize,
    verbose: usize,
) -> Result<List> {
    ensure_gpu()?;

    let start_seacell = Instant::now();

    let seacells_params = SEACellsParams::from_r_list(seacells_params)?;
    let verbosity = parse_verbosity_level(verbose);
    let knn_provided = knn_data != Nullable::Null;

    let reader = ParallelSparseReader::new(&f_path).to_extendr()?;

    if cells_to_use.is_some() && knn_provided {
        println!(
            "[WARNING!] 'knn_data' is ignored when 'cells_to_use' is set; the kNN graph will be regenerated on the subset"
        );
    }

    let mapping = resolve_cell_mapping(
        cells_to_keep.as_deref(),
        cells_to_use.as_deref(),
        embd.nrows(),
        reader.get_header().total_cells,
    )?;

    let embd_mat = match &mapping.rows_to_use {
        Some(rows_to_use) => {
            if verbosity.normal_verbosity() {
                println!(
                    "Subsetting to {} cells (from {} QC-passing cells)",
                    rows_to_use.len(),
                    embd.nrows()
                );
            }
            subset_embedding(&embd, rows_to_use)
        }
        None => r_matrix_to_faer_fp32(&embd),
    };

    let is_subset = mapping.needs_subsetting();
    let subset_to_orig = mapping.subset_to_orig;
    let n_total_cells = mapping.n_total;

    let (knn_indices, knn_dist) = if knn_provided && !is_subset {
        if verbosity.normal_verbosity() {
            println!("Using provided kNN graph.")
        }

        let knn_data = knn_data
            .into_robj()
            .as_list()
            .ok_or_else(|| Error::Other("'knn_data' is not a list".into()))?;
        let (knn_indices, knn_dist, _, _) = knn_data_to_rust(knn_data)?;

        if knn_indices.len() != embd_mat.nrows() {
            return Err(format!(
                "kNN indices have {} rows but embedding has {}",
                knn_indices.len(),
                embd_mat.nrows()
            )
            .into());
        }

        (knn_indices, knn_dist)
    } else {
        let start_knn = Instant::now();

        if verbosity.normal_verbosity() {
            println!("Regenerating kNN graph.")
        }

        let (knn_indices, knn_dist) = generate_knn_with_dist(
            embd_mat.as_ref(),
            &seacells_params.knn_params,
            true,
            false,
            seed,
            verbosity.detailed_verbosity(),
        )
        .to_extendr()?;
        let knn_dist = knn_dist.ok_or_else(|| {
            Error::Other("kNN generation returned no distances despite being asked for".into())
        })?;

        if verbosity.normal_verbosity() {
            println!(
                "kNN generation done in : {:.2?} with {}",
                start_knn.elapsed(),
                seacells_params.knn_params.knn_method
            );
        }

        (knn_indices, knn_dist)
    };

    let device: WgpuDevice = Default::default();

    let (assignments_raw, groups_raw, archetypes_raw, rss) = seacells_fit_gpu::<WgpuRuntime>(
        embd_mat.as_ref(),
        &knn_indices,
        &knn_dist,
        &seacells_params,
        seed,
        device.clone(),
        verbose,
    )
    .to_extendr()?;

    // force VRAM memory clean up to avoid memory leaks
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    // Archetype initialisation dedups, so the fit can come back with fewer
    // archetypes than `n_sea_cells` were requested. Key everything off what came
    // back rather than off the requested count.
    let k = archetypes_raw.len();

    let mut id_remap: Vec<Option<usize>> = vec![None; k];
    let mut groups_kept: Vec<Vec<usize>> = Vec::new();
    let mut archetypes_kept: Vec<usize> = Vec::new();

    for (old_id, group) in groups_raw.into_iter().enumerate() {
        if !group.is_empty() {
            id_remap[old_id] = Some(groups_kept.len());
            archetypes_kept.push(archetypes_raw[old_id]);
            groups_kept.push(group);
        }
    }

    if verbosity.normal_verbosity() && groups_kept.len() < seacells_params.n_sea_cells {
        println!(
            "Dropped {} empty archetype(s); keeping {} of {} requested",
            seacells_params.n_sea_cells - groups_kept.len(),
            groups_kept.len(),
            seacells_params.n_sea_cells
        );
    }

    let assignments_subset_opt: Vec<Option<usize>> =
        assignments_raw.iter().map(|&old| id_remap[old]).collect();

    // Always remap: the aggregation reader and R both work in original space.
    let assignments_full: Vec<Option<usize>> =
        remap_assignments_to_original(&assignments_subset_opt, &subset_to_orig, n_total_cells);

    let archetypes_original: Vec<usize> = archetypes_kept
        .iter()
        .map(|&idx| subset_to_orig[idx])
        .collect();

    let assignment_list = assignments_to_r_list(&assignments_full, n_total_cells);

    if verbosity.normal_verbosity() {
        println!("Aggregating meta cells.");
    }

    let metacells_original: Vec<Vec<usize>> = remap_metacells_to_original(
        &groups_kept.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        &subset_to_orig,
    );

    let metacells_refs: Vec<&[usize]> = metacells_original.iter().map(|v| v.as_slice()).collect();

    let n_genes = reader.get_header().total_genes;

    let aggregated: CompressedSparseData2<u32, f32> =
        aggregate_meta_cells(&reader, &metacells_refs, target_size as f32, n_genes).to_extendr()?;

    if verbosity.normal_verbosity() {
        println!("SEACells (GPU) found in : {:.2?}", start_seacell.elapsed());
    }

    Ok(list!(
        assignments = assignment_list,
        aggregated = list!(
            indptr = aggregated.indptr.r_int_convert(),
            indices = aggregated.indices.r_int_convert(),
            raw_counts = aggregated.data.r_int_convert(),
            norm_counts = aggregated
                .data_2
                .ok_or_else(|| Error::Other("aggregation returned no normalised counts".into()))?
                .r_float_convert(),
            nrow = aggregated.shape.0,
            ncol = aggregated.shape.1
        ),
        rss = rss,
        archetypes = archetypes_original.r_int_convert()
    ))
}
