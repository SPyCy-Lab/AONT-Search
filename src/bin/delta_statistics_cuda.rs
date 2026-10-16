#![cfg(target_os = "linux")]
use std::{
    env,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

use rand::{seq::SliceRandom, thread_rng};

use aont_search::{
    field::{ArkPrimeField, FiniteFieldOps},
    types::Symbol,
    cuda_flat_verifier::cuda_flat_verify,
    field::cauchy_matrix,
};

use ark_ff::{Fp64, MontBackend, MontConfig};

// ------------------------------------------------------------
// Supported prime fields
// ------------------------------------------------------------

#[derive(MontConfig)]
#[modulus = "7"]
#[generator = "3"]
pub struct F7Config;
type F7 = Fp64<MontBackend<F7Config, 1>>;

#[derive(MontConfig)]
#[modulus = "11"]
#[generator = "2"]
pub struct F11Config;
type F11 = Fp64<MontBackend<F11Config, 1>>;

#[derive(MontConfig)]
#[modulus = "13"]
#[generator = "2"]
pub struct F13Config;
type F13 = Fp64<MontBackend<F13Config, 1>>;

#[derive(MontConfig)]
#[modulus = "17"]
#[generator = "3"]
pub struct F17Config;
type F17 = Fp64<MontBackend<F17Config, 1>>;

#[derive(MontConfig)]
#[modulus = "23"]
#[generator = "5"]
pub struct F23Config;
type F23 = Fp64<MontBackend<F23Config, 1>>;

#[derive(MontConfig)]
#[modulus = "31"]
#[generator = "3"]
pub struct F31Config;
type F31 = Fp64<MontBackend<F31Config, 1>>;

#[derive(MontConfig)]
#[modulus = "43"]
#[generator = "3"]
pub struct F43Config;
type F43 = Fp64<MontBackend<F43Config, 1>>;

#[derive(MontConfig)]
#[modulus = "47"]
#[generator = "5"]
pub struct F47Config;
    
pub type F47 = Fp64<MontBackend<F47Config, 1>>;
    
#[derive(MontConfig)]
#[modulus = "257"]
#[generator = "3"]
pub struct F257Config;
type F257 = Fp64<MontBackend<F257Config, 1>>;

// ------------------------------------------------------------
// Experiment configuration
// ------------------------------------------------------------

const TRIALS: usize = 100;
const BATCH_SIZE: usize = 16;

// ------------------------------------------------------------
// Statistics
// ------------------------------------------------------------

fn mean(values: &[usize]) -> f64 {
    values.iter().map(|&x| x as f64).sum::<f64>()
        / values.len() as f64
}

fn sample_std_dev(values: &[usize]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let avg = mean(values);

    let variance = values
        .iter()
        .map(|&x| {
            let difference = x as f64 - avg;
            difference * difference
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;

    variance.sqrt()
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

// ------------------------------------------------------------
// Random coordinate permutations
// ------------------------------------------------------------

fn random_permutations(
    q: usize,
    s: usize,
) -> Vec<Vec<Symbol>> {
    let mut rng = thread_rng();

    (0..s)
        .map(|_| {
            let mut permutation: Vec<Symbol> =
                (0..q).collect();

            // Fisher–Yates shuffle.
            permutation.shuffle(&mut rng);

            permutation
        })
        .collect()
}

// ------------------------------------------------------------
// Experiment
// ------------------------------------------------------------

fn run_experiment(
    q: usize,
    s: usize,
    field: &dyn FiniteFieldOps,
) -> Result<(), Box<dyn std::error::Error>> {
    if q < 2 * s {
        return Err(format!(
            "Cauchy construction requires q >= 2s; got q={q}, s={s}"
        ).into());
    }

    let filename = format!("delta_statistics_cuda_s{s}_q{q}.txt");

    let file = File::create(&filename)?;
    let mut writer = BufWriter::new(file);

    // Two disjoint lists of field symbols.
    let a: Vec<Symbol> = (0..s).collect();
    let b: Vec<Symbol> = (s..2 * s).collect();

    
    let matrix = cauchy_matrix(&a, &b, field);

    writeln!(writer, "Differential Uniformity Experiment")?;
    writeln!(writer, "q = {q}")?;
    writeln!(writer, "s = {s}")?;
    writeln!(writer, "Trials = {TRIALS}")?;
    writeln!(writer, "CUDA batch size = {BATCH_SIZE}")?;
    writeln!(writer, "Cauchy a = {a:?}")?;
    writeln!(writer, "Cauchy b = {b:?}")?;
    writeln!(writer)?;

    let mut deltas: Vec<usize> = Vec::with_capacity(TRIALS);
    let experiment_start = Instant::now();

    println!();
    println!("========================================");
    println!("Differential Uniformity Statistics");
    println!("========================================");
    println!("q:                 {q}");
    println!("s:                 {s}");
    println!("N = q^s:           {}", q.pow(s as u32));
    println!("Trials:            {TRIALS}");
    println!("CUDA batch size:  {BATCH_SIZE}");
    println!("Output file:       {filename}");
    println!("========================================");

    for trial in 1..=TRIALS {
        let trial_start = Instant::now();

        let input_perms = random_permutations(q, s);
        let output_perms = random_permutations(q, s);

        let result = cuda_flat_verify(
            &matrix,
            &input_perms,
            &output_perms,
            field,
            BATCH_SIZE,
        )?;

        let delta = result.delta;
        deltas.push(delta);

        // Save every result immediately.
        writeln!(writer, "================================")?;
        writeln!(writer, "Trial {trial}")?;
        writeln!(writer, "Delta: {delta}")?;
        writeln!(
            writer,
            "Best input difference: {:?}",
            result.best_input_difference
        )?;
        writeln!(
            writer,
            "Best output difference: {:?}",
            result.best_output_difference
        )?;
        writeln!(writer, "Input permutations:")?;

        for (i, perm) in input_perms.iter().enumerate() {
            writeln!(writer, "  tau[{i}] = {perm:?}")?;
        }

        writeln!(writer, "Output permutations:")?;

        for (j, perm) in output_perms.iter().enumerate() {
            writeln!(writer, "  pi[{j}] = {perm:?}")?;
        }

        writeln!(
            writer,
            "Elapsed: {:.3} seconds",
            trial_start.elapsed().as_secs_f64()
        )?;
        writeln!(writer)?;

        // Important for long-running experiments:
        // preserve completed trials on disk.
        writer.flush()?;

        let current_mean = mean(&deltas);
        let current_std = sample_std_dev(&deltas);
        let elapsed = experiment_start.elapsed();

        let average_trial_time =
            elapsed.as_secs_f64() / trial as f64;

        let remaining = Duration::from_secs_f64(
            average_trial_time * (TRIALS - trial) as f64
        );

        println!(
            "[{trial:3}/{TRIALS}] \
             Delta={delta:<8} \
             Mean={current_mean:<10.3} \
             Std={current_std:<10.3} \
             Trial={:.2}s \
             ETA={}",
            trial_start.elapsed().as_secs_f64(),
            format_duration(remaining)
        );
    }

    // --------------------------------------------------------
    // Final statistics
    // --------------------------------------------------------

    let final_mean = mean(&deltas);
    let final_std = sample_std_dev(&deltas);

    let min_delta = *deltas.iter().min().unwrap();
    let max_delta = *deltas.iter().max().unwrap();

    let mut sorted = deltas.clone();
    sorted.sort_unstable();

    let median = if TRIALS % 2 == 0 {
        (sorted[TRIALS / 2 - 1] as f64
            + sorted[TRIALS / 2] as f64)
            / 2.0
    } else {
        sorted[TRIALS / 2] as f64
    };

    let total_elapsed = experiment_start.elapsed();

    writeln!(writer, "================================")?;
    writeln!(writer, "FINAL STATISTICS")?;
    writeln!(writer, "================================")?;
    writeln!(writer, "Mean: {final_mean:.6}")?;
    writeln!(writer, "Sample std dev: {final_std:.6}")?;
    writeln!(writer, "Median: {median:.6}")?;
    writeln!(writer, "Minimum: {min_delta}")?;
    writeln!(writer, "Maximum: {max_delta}")?;
    writeln!(writer, "All deltas: {deltas:?}")?;
    writeln!(
        writer,
        "Total elapsed: {:.3} seconds",
        total_elapsed.as_secs_f64()
    )?;
    writer.flush()?;

    println!();
    println!("========================================");
    println!("FINAL RESULTS");
    println!("========================================");
    println!("Mean:        {final_mean:.6}");
    println!("Std dev:     {final_std:.6}");
    println!("Median:      {median:.6}");
    println!("Minimum:     {min_delta}");
    println!("Maximum:     {max_delta}");
    println!("Total time:  {}", format_duration(total_elapsed));
    println!("Saved to:    {filename}");
    println!("========================================");

    Ok(())
}

// ------------------------------------------------------------
// Field dispatch
// ------------------------------------------------------------

fn run_for_field(
    q: usize,
    s: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    match q {
        7 => run_experiment(q, s, &ArkPrimeField::<F7>::new(F7::from(3u64))),
        11 => run_experiment(q, s, &ArkPrimeField::<F11>::new(F11::from(2u64))),
        13 => run_experiment(q, s, &ArkPrimeField::<F13>::new(F13::from(2u64))),
        17 => run_experiment(q, s, &ArkPrimeField::<F17>::new(F17::from(3u64))),
        23 => run_experiment(q, s, &ArkPrimeField::<F23>::new(F23::from(5u64))),
        31 => run_experiment(q, s, &ArkPrimeField::<F31>::new(F31::from(3u64))),
        43 => run_experiment(q, s, &ArkPrimeField::<F43>::new(F43::from(3u64))),
        47 => run_experiment(q, s, &ArkPrimeField::<F47>::new(F47::from(5u64))),
        257 => run_experiment(q, s, &ArkPrimeField::<F257>::new(F257::from(3u64))),
        _ => Err(format!(
            "Unsupported field size q={q}. Add it to run_for_field."
        ).into()),
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!(
            "Usage: cargo run --release --bin delta_statistics_cuda -- <q> <s>"
        );
        std::process::exit(1);
    }

    let q: usize = args[1].parse()?;
    let s: usize = args[2].parse()?;

    run_for_field(q, s)
}

// cargo run --release --bin delta_statistics_cuda -- 11 5
