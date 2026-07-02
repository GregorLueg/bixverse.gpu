use extendr_api::*;
use manifolds_rs::prelude::*;
use manifolds_rs::PreComputedKnn;

/// Parse the nearest neighbours to a Rust function
///
/// ### Params
///
/// * `List` - The NearestNeighbour list.
///
/// ### Returns
///
/// A PreComputedKnn<f32> for the 2D embedding methods
pub fn nearest_neighbours_to_rust<T>(nn: List) -> PreComputedKnn<T>
where
    T: ManifoldsFloat,
{
    let k = nn.dollar("k").unwrap().as_integer().unwrap() as usize;

    let indices: Vec<i32> = nn.dollar("indices").unwrap().as_integer_vector().unwrap();
    let dist: Vec<f64> = nn.dollar("dist").unwrap().as_real_vector().unwrap();

    let indices: Vec<Vec<usize>> = indices
        .chunks(k)
        .map(|chunk| chunk.iter().map(|&i| (i - 1) as usize).collect())
        .collect();

    let dist: Vec<Vec<T>> = dist
        .chunks(k)
        .map(|chunk| chunk.iter().map(|&d| T::from_f64(d).unwrap()).collect())
        .collect();

    Some((indices, dist))
}
