use std::collections::{HashMap, HashSet};

use crate::config::SearchConfig;
use crate::types::{Tuple, TupleMap, TupleSet};

/// Return all unused targets that satisfy the AONT distance constraints at
/// one search level. Shared by the real DFS and the randomized estimator.
pub fn feasible_candidates_at_level<'a>(
    config: &SearchConfig,
    level: usize,
    ordered: &[(Tuple, TupleSet)],
    output_hamming_map: &TupleMap,
    all_tuples: &'a [Tuple],
    current_map: &HashMap<Tuple, Tuple>,
    used_targets: &HashSet<Tuple>,
) -> Vec<&'a Tuple> {
    if level >= ordered.len() || (config.fix_zero && level == 0) {
        return Vec::new();
    }

    let previous_neighbours = &ordered[level].1;
    let mut infeasible = HashSet::new();

    for neighbour_source in previous_neighbours {
        let mapped_target = current_map
            .get(neighbour_source)
            .expect("every previous neighbour should already be assigned");

        if let Some(ball) = output_hamming_map.get(mapped_target) {
            infeasible.extend(ball.iter().cloned());
        }
    }

    all_tuples
        .iter()
        .filter(|candidate| {
            !used_targets.contains(*candidate) && !infeasible.contains(*candidate)
        })
        .collect()
}
