/// Functions used to calulate hamming distance and create hamming spheres.
/// Used in the branch and bound for pruing the search space.
/// Contains functions that does the greedy ordering for vectors before the branch and bound search.
use std::collections::HashSet;

use crate::types::{Tuple, TupleMap, TupleSet};

pub fn generate_tuples(s: usize, q: usize) -> Vec<Tuple> {
    let total = q.pow(s as u32);
    let mut result = Vec::with_capacity(total);

    for mut index in 0..total {
        let mut tuple = vec![0; s];
        for pos in (0..s).rev() {
            tuple[pos] = index % q;
            index /= q;
        }
        result.push(tuple);
    }

    result
}

fn generate_hamming_ball_rec(
    base: &Tuple,
    pos: usize,
    remaining_changes: usize,
    q: usize,
    current: &mut Tuple,
    result: &mut TupleSet,
) {
    if pos == base.len() {
        result.insert(current.clone());
        return;
    }

    current[pos] = base[pos];
    generate_hamming_ball_rec(
        base,
        pos + 1,
        remaining_changes,
        q,
        current,
        result,
    );

    if remaining_changes > 0 {
        for symbol in 0..q {
            if symbol == base[pos] {
                continue;
            }
            current[pos] = symbol;
            generate_hamming_ball_rec(
                base,
                pos + 1,
                remaining_changes - 1,
                q,
                current,
                result,
            );
        }
        current[pos] = base[pos];
    }
}

pub fn build_hamming_map(tuples: &[Tuple], q: usize, radius: usize) -> TupleMap {
    let mut map = TupleMap::with_capacity(tuples.len());

    for tuple in tuples {
        let mut neighbours = HashSet::new();
        let mut current = tuple.clone();
        generate_hamming_ball_rec(
            tuple,
            0,
            radius,
            q,
            &mut current,
            &mut neighbours,
        );
        neighbours.remove(tuple);
        map.insert(tuple.clone(), neighbours);
    }

    map
}

pub fn order_tuples(tuples: &[Tuple], input_hamming_map: &TupleMap) -> Vec<(Tuple, TupleSet)> {
    if tuples.is_empty() {
        return Vec::new();
    }

    let zero = vec![0; tuples[0].len()];
    let mut ordered = Vec::with_capacity(tuples.len());
    let mut placed = HashSet::with_capacity(tuples.len());
    let mut unplaced: HashSet<Tuple> = tuples.iter().cloned().collect();

    ordered.push((zero.clone(), HashSet::new()));
    placed.insert(zero.clone());
    unplaced.remove(&zero);

    while !unplaced.is_empty() {
        let mut best_tuple: Option<Tuple> = None;
        let mut best_neighbours = HashSet::new();
        let mut best_count = 0usize;
        let mut first = true;

        let mut candidates: Vec<&Tuple> = unplaced.iter().collect();
        candidates.sort();

        for tuple in candidates {
            let intersection: HashSet<Tuple> = input_hamming_map[tuple]
                .iter()
                .filter(|neighbour| placed.contains(*neighbour))
                .cloned()
                .collect();

            if first || intersection.len() > best_count {
                first = false;
                best_count = intersection.len();
                best_tuple = Some(tuple.clone());
                best_neighbours = intersection;
            }
        }

        let chosen = best_tuple.expect("unplaced set unexpectedly empty");
        unplaced.remove(&chosen);
        placed.insert(chosen.clone());
        ordered.push((chosen, best_neighbours));
    }

    ordered
}
