/// Implementation of the Kunuth backtracking estimator 
/// it is used to find if the search algorithm will terminate in reasonable time or not.
/// does not affect the logic of the branch and bound in any way.
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::SearchConfig;
use crate::search::feasible_candidates_at_level;
use crate::types::{Tuple, TupleMap, TupleSet};

pub struct ProgressTracker {
    start_time: Instant,
    last_report_time: Instant,
    last_report_branches: u128,
    report_interval: u128,
    estimated_total: f64,
}

impl ProgressTracker {
    pub fn new(report_interval: u128, estimated_total: f64) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_report_time: now,
            last_report_branches: 0,
            report_interval: report_interval.max(1),
            estimated_total,
        }
    }

    pub fn maybe_report(&mut self, branches_visited: u128) {
        if branches_visited.saturating_sub(self.last_report_branches) < self.report_interval {
            return;
        }

        let now = Instant::now();
        let total_elapsed = now.duration_since(self.start_time);
        let interval_elapsed = now.duration_since(self.last_report_time);
        let total_seconds = total_elapsed.as_secs_f64();
        let interval_seconds = interval_elapsed.as_secs_f64();

        let average_rate = if total_seconds > 0.0 {
            branches_visited as f64 / total_seconds
        } else {
            0.0
        };

        let interval_branches = branches_visited.saturating_sub(self.last_report_branches);
        let recent_rate = if interval_seconds > 0.0 {
            interval_branches as f64 / interval_seconds
        } else {
            0.0
        };

        let estimated_completion = if self.estimated_total > 0.0 {
            100.0 * branches_visited as f64 / self.estimated_total
        } else {
            0.0
        };

        let estimated_total_seconds = if average_rate > 0.0 {
            self.estimated_total / average_rate
        } else {
            f64::INFINITY
        };

        let estimated_remaining_seconds = if average_rate > 0.0 {
            ((self.estimated_total - branches_visited as f64) / average_rate).max(0.0)
        } else {
            f64::INFINITY
        };

        println!();
        println!("========== Search progress ==========");
        println!("Branches visited:       {}", format_u128(branches_visited));
        println!("Heuristic total:        {:.6e}", self.estimated_total);
        println!("Heuristic completion:   {:.8}%", estimated_completion);
        println!("Elapsed time:           {}", format_duration(total_elapsed));
        println!("Average rate:           {:.3e} branches/sec", average_rate);
        println!("Recent rate:            {:.3e} branches/sec", recent_rate);

        if estimated_total_seconds.is_finite() {
            println!("Estimated total time:   {}", format_seconds(estimated_total_seconds));
            println!("Estimated time left:    {}", format_seconds(estimated_remaining_seconds));
        } else {
            println!("Estimated total time:   unavailable");
            println!("Estimated time left:    unavailable");
        }
        println!("=====================================");
        println!();

        self.last_report_time = now;
        self.last_report_branches = branches_visited;
    }

    pub fn final_report(&self, branches_visited: u128) {
        let elapsed = self.start_time.elapsed();
        let seconds = elapsed.as_secs_f64();
        let rate = if seconds > 0.0 {
            branches_visited as f64 / seconds
        } else {
            0.0
        };

        println!();
        println!("========== Final search report ==========");
        println!("Branches visited: {}", format_u128(branches_visited));
        println!("Elapsed time:     {}", format_duration(elapsed));
        println!("Average rate:     {:.3e} branches/sec", rate);
        println!("=========================================");
    }
}

#[derive(Debug, Clone)]
pub struct RandomTreeEstimate {
    pub samples: usize,
    pub mean_nodes: f64,
    pub median_nodes: f64,
    pub minimum_nodes: f64,
    pub maximum_nodes: f64,
    pub standard_deviation: f64,
    pub mean_complete_leaves: f64,
    pub completed_paths: usize,
    pub dead_end_paths: usize,
    pub average_depth: f64,
}

#[derive(Debug)]
struct SinglePathEstimate {
    estimated_nodes: f64,
    estimated_complete_leaves: f64,
    depth_reached: usize,
    reached_complete_mapping: bool,
}

fn sample_one_search_path<R: Rng + ?Sized>(
    config: &SearchConfig,
    ordered: &[(Tuple, TupleSet)],
    output_hamming_map: &TupleMap,
    all_tuples: &[Tuple],
    rng: &mut R,
) -> SinglePathEstimate {
    let mut current_map: HashMap<Tuple, Tuple> = HashMap::with_capacity(all_tuples.len());
    let mut used_targets: TupleSet = HashSet::with_capacity(all_tuples.len());

    let mut estimated_nodes = 1.0_f64;
    let mut path_weight = 1.0_f64;
    let mut level = 0usize;

    while level < ordered.len() {
        let source = &ordered[level].0;

        if config.fix_zero && level == 0 {
            let zero = vec![0; config.s];
            estimated_nodes += path_weight;
            current_map.insert(source.clone(), zero.clone());
            used_targets.insert(zero);
            level += 1;
            continue;
        }

        let feasible = feasible_candidates_at_level(
            config,
            level,
            ordered,
            output_hamming_map,
            all_tuples,
            &current_map,
            &used_targets,
        );

        let branching = feasible.len();
        if branching == 0 {
            return SinglePathEstimate {
                estimated_nodes,
                estimated_complete_leaves: 0.0,
                depth_reached: level,
                reached_complete_mapping: false,
            };
        }

        path_weight *= branching as f64;
        estimated_nodes += path_weight;

        let chosen = feasible[rng.gen_range(0..branching)].clone();
        current_map.insert(source.clone(), chosen.clone());
        used_targets.insert(chosen);
        level += 1;
    }

    SinglePathEstimate {
        estimated_nodes,
        estimated_complete_leaves: path_weight,
        depth_reached: level,
        reached_complete_mapping: true,
    }
}

pub fn estimate_search_tree_randomly(
    config: &SearchConfig,
    ordered: &[(Tuple, TupleSet)],
    output_hamming_map: &TupleMap,
    all_tuples: &[Tuple],
    number_of_samples: usize,
    random_seed: u64,
) -> RandomTreeEstimate {
    assert!(number_of_samples > 0, "number_of_samples must be positive");

    let mut rng = StdRng::seed_from_u64(random_seed);
    let mut node_estimates = Vec::with_capacity(number_of_samples);
    let mut total_leaf_estimate = 0.0_f64;
    let mut total_depth = 0usize;
    let mut completed_paths = 0usize;
    let mut dead_end_paths = 0usize;

    println!();
    println!("Running {} randomized tree samples...", number_of_samples);
    let progress_interval = (number_of_samples / 10).max(1);

    for sample_index in 0..number_of_samples {
        let sample = sample_one_search_path(
            config,
            ordered,
            output_hamming_map,
            all_tuples,
            &mut rng,
        );

        node_estimates.push(sample.estimated_nodes);
        total_leaf_estimate += sample.estimated_complete_leaves;
        total_depth += sample.depth_reached;

        if sample.reached_complete_mapping {
            completed_paths += 1;
        } else {
            dead_end_paths += 1;
        }

        if (sample_index + 1) % progress_interval == 0 || sample_index + 1 == number_of_samples {
            println!("Tree sampling: {}/{} paths", sample_index + 1, number_of_samples);
        }
    }

    let mean_nodes = node_estimates.iter().sum::<f64>() / number_of_samples as f64;
    let mean_complete_leaves = total_leaf_estimate / number_of_samples as f64;
    let average_depth = total_depth as f64 / number_of_samples as f64;

    let standard_deviation = if number_of_samples > 1 {
        let squared_deviation_sum: f64 = node_estimates
            .iter()
            .map(|estimate| {
                let difference = estimate - mean_nodes;
                difference * difference
            })
            .sum();
        (squared_deviation_sum / (number_of_samples - 1) as f64).sqrt()
    } else {
        0.0
    };

    node_estimates.sort_by(|left, right| left.total_cmp(right));
    let median_nodes = if number_of_samples % 2 == 1 {
        node_estimates[number_of_samples / 2]
    } else {
        let upper = node_estimates[number_of_samples / 2];
        let lower = node_estimates[number_of_samples / 2 - 1];
        (lower + upper) / 2.0
    };

    RandomTreeEstimate {
        samples: number_of_samples,
        mean_nodes,
        median_nodes,
        minimum_nodes: node_estimates[0],
        maximum_nodes: node_estimates[number_of_samples - 1],
        standard_deviation,
        mean_complete_leaves,
        completed_paths,
        dead_end_paths,
        average_depth,
    }
}

pub fn print_random_tree_estimate(estimate: &RandomTreeEstimate) {
    let completion_percentage =
        100.0 * estimate.completed_paths as f64 / estimate.samples as f64;

    println!();
    println!("========== Randomized tree estimate ==========");
    println!("Samples:                    {}", estimate.samples);
    println!("Mean estimated nodes:       {:.6e}", estimate.mean_nodes);
    println!("Median estimated nodes:     {:.6e}", estimate.median_nodes);
    println!("Minimum path estimate:      {:.6e}", estimate.minimum_nodes);
    println!("Maximum path estimate:      {:.6e}", estimate.maximum_nodes);
    println!("Standard deviation:         {:.6e}", estimate.standard_deviation);
    println!("Estimated complete leaves:  {:.6e}", estimate.mean_complete_leaves);
    println!(
        "Paths reaching full depth:  {} ({:.3}%)",
        estimate.completed_paths, completion_percentage
    );
    println!("Paths ending in dead ends:  {}", estimate.dead_end_paths);
    println!("Average sampled depth:      {:.3}", estimate.average_depth);
    println!("==============================================");
}

/// Older independence-based heuristic retained for comparison/debugging.
pub fn estimate_complexity(
    ordered: &[(Tuple, TupleSet)],
    q: usize,
    s: usize,
    t: usize,
    fix_zero: bool,
) -> f64 {
    let space_size = (q as f64).powi(s as i32);
    let mut hamming_ball = 0.0;

    for i in 1..=t.min(s) {
        let binomial = binomial_coefficient(s, i) as f64;
        let changed_symbols = ((q - 1) as f64).powi(i as i32);
        hamming_ball += binomial * changed_symbols;
    }

    let survival_probability = (1.0 - hamming_ball / space_size).max(0.0);
    let mut estimated_leaves = 1.0_f64;

    println!("Space size               = {space_size}");
    println!("Hamming ball, no center  = {hamming_ball}");
    println!("Single-ball survival     = {survival_probability}");
    println!("Fix f(0) = 0             = {fix_zero}");
    println!();

    for (level, (_, neighbours)) in ordered.iter().enumerate() {
        let placed = level;
        let neighbour_count = neighbours.len();

        let branching = if fix_zero && level == 0 {
            1.0
        } else {
            let unused_targets = space_size - placed as f64;
            let expected_feasible =
                unused_targets * survival_probability.powf(neighbour_count as f64);
            expected_feasible.max(1.0)
        };

        estimated_leaves *= branching;
        println!(
            "{}. neighbours = {}, branching ≈ {:.6e}, running product ≈ {:.6e}",
            level + 1,
            neighbour_count,
            branching,
            estimated_leaves,
        );
    }

    println!("\nEstimated leaves ≈ {:.6e}", estimated_leaves);
    estimated_leaves
}

fn binomial_coefficient(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }

    let k = k.min(n - k);
    let mut result = 1_u128;
    for i in 0..k {
        result = result * (n - i) as u128 / (i + 1) as u128;
    }
    result
}

fn format_u128(value: u128) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            output.push(',');
        }
        output.push(character);
    }
    output
}

fn format_duration(duration: Duration) -> String {
    format_seconds(duration.as_secs_f64())
}

fn format_seconds(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "unavailable".to_string();
    }

    let total_seconds = seconds.max(0.0).round() as u128;
    let days = total_seconds / 86_400;
    let hours = (total_seconds % 86_400) / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let secs = total_seconds % 60;

    if days > 0 {
        format!("{} days, {:02}:{:02}:{:02}", days, hours, minutes, secs)
    } else {
        format!("{:02}:{:02}:{:02}", hours, minutes, secs)
    }
}
