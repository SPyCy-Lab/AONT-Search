/// Implementation of the main branch and bound algorithm described in the paper.
use std::collections::{HashMap, HashSet};

use crate::config::SearchConfig;
use crate::field::FiniteFieldOps;
use crate::hamming::{build_hamming_map, generate_tuples, order_tuples};
use crate::linearity::is_linear_aont;
use crate::mapping_io::OutputFiles;
use crate::search::feasible_candidates_at_level;
use crate::time_estimator::{
    ProgressTracker, estimate_complexity, estimate_search_tree_randomly, print_random_tree_estimate,
};
use crate::types::{Mapping, Tuple, TupleSet};

#[derive(Default)]
pub struct BnbStats {
    pub total: u64,
    pub linear_count: u64,
    pub nonlinear_count: u64,
    pub branches_visited: u128,
    pub first_nonlinear: Vec<Mapping>,
}

#[allow(clippy::too_many_arguments)]
fn dfs_branch_and_bound(
    field: &dyn FiniteFieldOps,
    config: &SearchConfig,
    level: usize,
    ordered: &[(Tuple, TupleSet)],
    output_hamming_map: &crate::types::TupleMap,
    all_tuples: &[Tuple],
    current_map: &mut HashMap<Tuple, Tuple>,
    used_targets: &mut TupleSet,
    stats: &mut BnbStats,
    output_files: &mut OutputFiles,
    progress: &mut ProgressTracker,
) -> std::io::Result<()> {
    stats.branches_visited += 1;
    progress.maybe_report(stats.branches_visited);

    if level == ordered.len() {
        stats.total += 1;

        let mapping: Mapping = ordered
            .iter()
            .map(|(source, _)| (source.clone(), current_map[source].clone()))
            .collect();

        let linear = is_linear_aont(&mapping, field, config.s, false);
        output_files.write_mapping(&mapping, linear, config.s, field.order())?;

        if linear {
            stats.linear_count += 1;
        } else {
            stats.nonlinear_count += 1;
            if stats.first_nonlinear.len() < config.first_nonlinear_to_keep {
                stats.first_nonlinear.push(mapping);
            }
        }

        return Ok(());
    }

    let source = &ordered[level].0;

    if config.fix_zero && level == 0 {
        let zero = vec![0; config.s];
        current_map.insert(source.clone(), zero.clone());
        used_targets.insert(zero.clone());

        dfs_branch_and_bound(
            field,
            config,
            level + 1,
            ordered,
            output_hamming_map,
            all_tuples,
            current_map,
            used_targets,
            stats,
            output_files,
            progress,
        )?;

        used_targets.remove(&zero);
        current_map.remove(source);
        return Ok(());
    }

    let feasible = feasible_candidates_at_level(
        config,
        level,
        ordered,
        output_hamming_map,
        all_tuples,
        current_map,
        used_targets,
    );

    // Clone here so recursion can mutably borrow `used_targets` and `current_map`.
    let feasible: Vec<Tuple> = feasible.into_iter().cloned().collect();

    for candidate in feasible {
        current_map.insert(source.clone(), candidate.clone());
        used_targets.insert(candidate.clone());

        dfs_branch_and_bound(
            field,
            config,
            level + 1,
            ordered,
            output_hamming_map,
            all_tuples,
            current_map,
            used_targets,
            stats,
            output_files,
            progress,
        )?;

        used_targets.remove(&candidate);
        current_map.remove(source);
    }

    Ok(())
}

pub fn enumerate_aonts(
    field: &dyn FiniteFieldOps,
    config: &SearchConfig,
) -> std::io::Result<BnbStats> {
    assert!(config.t <= config.s, "t must satisfy t <= s");

    let q = field.order();
    let all_tuples = generate_tuples(config.s, q);

    let input_hamming_map = build_hamming_map(&all_tuples, q, config.s - config.t);
    let ordered = order_tuples(&all_tuples, &input_hamming_map);
    let output_hamming_map = build_hamming_map(&all_tuples, q, config.t);

    if config.run_old_heuristic {
        estimate_complexity(&ordered, q, config.s, config.t, config.fix_zero);
    }

    let random_estimate = estimate_search_tree_randomly(
        config,
        &ordered,
        &output_hamming_map,
        &all_tuples,
        config.estimator_samples,
        config.estimator_seed,
    );
    print_random_tree_estimate(&random_estimate);

    let mut current_map = HashMap::with_capacity(all_tuples.len());
    let mut used_targets = HashSet::with_capacity(all_tuples.len());
    let mut stats = BnbStats::default();
    let mut output_files = OutputFiles::create(config)?;
    let mut progress = ProgressTracker::new(
        config.progress_report_interval,
        random_estimate.mean_nodes,
    );

    dfs_branch_and_bound(
        field,
        config,
        0,
        &ordered,
        &output_hamming_map,
        &all_tuples,
        &mut current_map,
        &mut used_targets,
        &mut stats,
        &mut output_files,
        &mut progress,
    )?;

    progress.final_report(stats.branches_visited);
    output_files.flush()?;

    for (index, mapping) in stats.first_nonlinear.iter().enumerate() {
        println!("Nonlinear mapping #{}:", index + 1);
        for (source, target) in mapping {
            println!("{:?} -> {:?}", source, target);
        }
        is_linear_aont(mapping, field, config.s, true);
        println!();
    }

    println!("------------------------------------");
    println!("Total AONTs found:     {}", stats.total);
    println!("Nonlinear AONTs:       {}", stats.nonlinear_count);
    println!("Linear AONTs:          {}", stats.linear_count);
    println!("Branches visited:      {}", stats.branches_visited);
    println!("Fixed f(0)=0:          {}", config.fix_zero);
    println!("------------------------------------");

    Ok(stats)
}
