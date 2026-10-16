use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{
    AtomicU64,
    AtomicUsize,
    Ordering,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;

use crate::field::{
    FieldMatrix,
    FiniteFieldOps,
};
use crate::types::Symbol;


// ============================================================
// Configuration
// ============================================================

#[derive(Debug, Clone)]
pub struct SampledRandomSearchConfig {
    /// Number of random candidates to test.
    pub max_trials: u64,

    /// Number of fixed representative DDT rows used to score
    /// every candidate.
    ///
    /// Larger = stronger screening, but slower.
    pub sampled_rows: usize,

    /// Master random seed.
    pub seed: u64,

    /// File where every new sampled best is stored.
    pub output_path: String,

    /// Print progress every this many completed candidates.
    ///
    /// 0 disables progress output.
    pub progress_interval: u64,
}


impl Default for SampledRandomSearchConfig {
    fn default() -> Self {
        Self {
            max_trials: 10_000,

            // Strict default.
            sampled_rows: 1024,

            seed: 12345,

            output_path:
                "sampled_random_search_bests.txt"
                    .to_string(),

            progress_interval: 100,
        }
    }
}


// ============================================================
// Statistics
// ============================================================

#[derive(Debug)]
pub struct SampledRandomSearchStats {
    pub trials: u64,

    /// Best exact maximum over the fixed sampled rows.
    ///
    /// This is NOT necessarily the true DU.
    pub best_sampled_delta: usize,

    /// Number of times the global sampled record improved.
    pub improvements: u64,

    /// Candidates that completed all sampled rows.
    pub fully_sampled: u64,

    /// Candidates rejected before completing all sampled rows.
    pub early_aborted: u64,

    pub elapsed: Duration,
}


// ============================================================
// Candidate
// ============================================================

#[derive(Debug, Clone)]
struct Candidate {
    input_perms: Vec<Vec<Symbol>>,
    output_perms: Vec<Vec<Symbol>>,
}


/// Generate one random permutation of the alphabet for
/// every coordinate.
fn random_coordinate_permutations(
    s: usize,
    q: usize,
    rng: &mut StdRng,
) -> Vec<Vec<Symbol>> {
    let mut result =
        Vec::with_capacity(s);

    for _ in 0..s {
        let mut permutation:
            Vec<Symbol> =
            (0..q).collect();

        permutation.shuffle(rng);

        result.push(permutation);
    }

    result
}


/// Generate random input and output coordinate permutations.
fn random_candidate(
    s: usize,
    q: usize,
    rng: &mut StdRng,
) -> Candidate {
    Candidate {
        input_perms:
            random_coordinate_permutations(
                s,
                q,
                rng,
            ),

        output_perms:
            random_coordinate_permutations(
                s,
                q,
                rng,
            ),
    }
}


// ============================================================
// Deterministic per-trial RNG
// ============================================================

#[inline]
fn splitmix64(
    mut x: u64,
) -> u64 {
    x =
        x.wrapping_add(
            0x9E3779B97F4A7C15
        );

    let mut z = x;

    z =
        (z ^ (z >> 30))
            .wrapping_mul(
                0xBF58476D1CE4E5B9
            );

    z =
        (z ^ (z >> 27))
            .wrapping_mul(
                0x94D049BB133111EB
            );

    z ^ (z >> 31)
}


#[inline]
fn trial_seed(
    master_seed: u64,
    trial: u64,
) -> u64 {
    splitmix64(
        master_seed ^ trial
    )
}


// ============================================================
// Fast field tables
// ============================================================

struct FastField {
    q: usize,

    add: Vec<Symbol>,
    sub: Vec<Symbol>,
    neg: Vec<Symbol>,
}


impl FastField {
    fn new(
        field: &dyn FiniteFieldOps,
    ) -> Self {
        let q =
            field.order();

        let mut add =
            vec![0; q * q];

        let mut sub =
            vec![0; q * q];

        let mut neg =
            vec![0; q];

        for a in 0..q {
            neg[a] =
                field.sub(
                    0,
                    a,
                );

            for b in 0..q {
                add[a * q + b] =
                    field.add(
                        a,
                        b,
                    );

                sub[a * q + b] =
                    field.sub(
                        a,
                        b,
                    );
            }
        }

        Self {
            q,
            add,
            sub,
            neg,
        }
    }


    #[inline(always)]
    fn add(
        &self,
        a: Symbol,
        b: Symbol,
    ) -> Symbol {
        self.add[
            a * self.q + b
        ]
    }


    #[inline(always)]
    fn sub(
        &self,
        a: Symbol,
        b: Symbol,
    ) -> Symbol {
        self.sub[
            a * self.q + b
        ]
    }


    #[inline(always)]
    fn neg(
        &self,
        a: Symbol,
    ) -> Symbol {
        self.neg[a]
    }
}


// ============================================================
// Vector-space helpers
// ============================================================

fn vector_space_size(
    q: usize,
    s: usize,
) -> usize {
    q.checked_pow(
        s as u32
    )
    .expect(
        "q^s does not fit in usize"
    )
}


/// Flat table containing every vector in F_q^s.
///
/// Vector with index i occupies:
///
///     table[i*s .. (i+1)*s]
fn build_vector_table(
    q: usize,
    s: usize,
    n: usize,
) -> Vec<Symbol> {
    let mut table =
        vec![0; n * s];

    for index in 0..n {
        let mut value =
            index;

        for coordinate
            in (0..s).rev()
        {
            table[
                index * s
                    + coordinate
            ] =
                value % q;

            value /= q;
        }
    }

    table
}


/// Base-q positional weights.
///
/// Example:
///
/// q = 31, s = 4
///
///     [31^3, 31^2, 31, 1]
fn build_place_values(
    q: usize,
    s: usize,
) -> Vec<usize> {
    let mut place =
        vec![1usize; s];

    if s >= 2 {
        for i
            in (0..s - 1).rev()
        {
            place[i] =
                place[i + 1]
                    .checked_mul(q)
                    .expect(
                        "base-q place value overflow"
                    );
        }
    }

    place
}


fn decode_index(
    index: usize,
    vector_table: &[Symbol],
    s: usize,
) -> Vec<Symbol> {
    vector_table[
        index * s
            ..
        (index + 1) * s
    ]
    .to_vec()
}


// ============================================================
// Matrix multiplication lookup
// ============================================================

/// Precompute
///
///     matrix[i][j] * x
///
/// for every matrix entry and every field symbol x.
fn build_matrix_product_table(
    matrix: &FieldMatrix,
    field: &dyn FiniteFieldOps,
) -> Vec<Symbol> {
    let s =
        matrix.len();

    let q =
        field.order();

    let mut table =
        vec![0; s * s * q];

    for i in 0..s {
        for j in 0..s {
            let coefficient =
                matrix[i][j];

            for x in 0..q {
                table[
                    (i * s + j)
                        * q
                        + x
                ] =
                    field.mul(
                        x,
                        coefficient,
                    );
            }
        }
    }

    table
}


// ============================================================
// Difference representatives
// ============================================================

/// Return the base-q index of -a.
fn negative_vector_index(
    a_index: usize,
    vector_table: &[Symbol],
    place_values: &[usize],
    fast_field: &FastField,
    s: usize,
) -> usize {
    let offset =
        a_index * s;

    let mut result =
        0usize;

    for i in 0..s {
        let digit =
            vector_table[
                offset + i
            ];

        result +=
            fast_field.neg(digit)
                * place_values[i];
    }

    result
}


/// Keep one representative from each pair {a,-a}.
///
/// For odd characteristic this approximately halves the
/// number of input differences.
///
/// In characteristic 2, a = -a, so all nonzero differences
/// remain.
fn build_difference_representatives(
    vector_table: &[Symbol],
    place_values: &[usize],
    fast_field: &FastField,
    s: usize,
    n: usize,
) -> Vec<usize> {
    let mut representatives =
        Vec::with_capacity(
            n - 1
        );

    for a_index in 1..n {
        let neg_index =
            negative_vector_index(
                a_index,
                vector_table,
                place_values,
                fast_field,
                s,
            );

        if a_index <= neg_index {
            representatives.push(
                a_index
            );
        }
    }

    representatives
}


// ============================================================
// Fixed DDT sample
// ============================================================

/// Choose the DDT rows once.
///
/// Every candidate is evaluated on exactly the same rows.
///
/// This is important because it makes sampled scores directly
/// comparable between candidates.
fn choose_fixed_sample_rows(
    difference_representatives: &[usize],
    sampled_rows: usize,
    seed: u64,
) -> Vec<usize> {
    let rows_to_choose =
        sampled_rows.min(
            difference_representatives.len()
        );

    let sample_seed =
        splitmix64(
            seed
                ^ 0xA5A5_A5A5_A5A5_A5A5
        );

    let mut rng =
        StdRng::seed_from_u64(
            sample_seed
        );

    let mut rows =
        difference_representatives
            .to_vec();

    rows.shuffle(
        &mut rng
    );

    rows.truncate(
        rows_to_choose
    );

    rows
}


// ============================================================
// Worker workspace
// ============================================================

/// Private reusable memory for each Rayon worker.
///
/// No locks are required for these large arrays.
struct WorkerWorkspace {
    /// f(x) for every x.
    output_cache: Vec<Symbol>,

    /// DDT histogram.
    counts: Vec<u32>,

    /// Epoch stamps avoid clearing counts for every row.
    stamps: Vec<u32>,

    epoch: u32,
}


impl WorkerWorkspace {
    fn new(
        n: usize,
        s: usize,
    ) -> Self {
        Self {
            output_cache:
                vec![0; n * s],

            counts:
                vec![0; n],

            stamps:
                vec![0; n],

            epoch: 0,
        }
    }
}


// ============================================================
// Candidate mapping
// ============================================================

/// Build
///
///     f = output_perms o M o input_perms
///
/// for every x in F_q^s.
///
/// The complete output mapping is cached because every sampled
/// DDT row repeatedly accesses f(x).
fn build_candidate_outputs(
    candidate: &Candidate,
    vector_table: &[Symbol],
    matrix_products: &[Symbol],
    fast_field: &FastField,
    s: usize,
    n: usize,
    output_cache: &mut [Symbol],
) {
    let q =
        fast_field.q;

    for x_index in 0..n {
        let x_offset =
            x_index * s;

        for j in 0..s {
            let mut sum =
                0;

            for i in 0..s {
                let x_i =
                    vector_table[
                        x_offset + i
                    ];

                let permuted =
                    candidate
                        .input_perms[i]
                        [x_i];

                let product =
                    matrix_products[
                        (i * s + j)
                            * q
                            + permuted
                    ];

                sum =
                    fast_field.add(
                        sum,
                        product,
                    );
            }

            output_cache[
                x_offset + j
            ] =
                candidate
                    .output_perms[j]
                    [sum];
        }
    }
}


// ============================================================
// Sample result
// ============================================================

#[derive(Debug)]
struct SampledResult {
    /// Exact maximum DDT count among the fixed sampled rows.
    max_sampled_delta: usize,

    /// Sampled a producing the largest count.
    worst_sampled_a: usize,

    /// b producing the largest count.
    worst_sampled_b: usize,
}


// ============================================================
// Strict sampled DU
// ============================================================

/// Compute the exact DDT maximum over the fixed sample rows.
///
/// Returns:
///
///     Some(result)
///
/// only if the candidate finishes every sampled row without
/// reaching `cutoff`.
///
/// Returns:
///
///     None
///
/// immediately once any DDT entry reaches cutoff.
///
/// Therefore, when cutoff is the current global sampled best,
/// None proves that the candidate cannot improve the sampled
/// record.
fn sampled_differential_uniformity(
    output_cache: &[Symbol],
    vector_table: &[Symbol],
    place_values: &[usize],
    fixed_sample_rows: &[usize],
    fast_field: &FastField,
    s: usize,
    n: usize,
    cutoff: usize,
    counts: &mut [u32],
    stamps: &mut [u32],
    epoch: &mut u32,
) -> Option<SampledResult> {
    let mut max_sampled_delta =
        0usize;

    let mut worst_sampled_a =
        0usize;

    let mut worst_sampled_b =
        0usize;

    // ========================================================
    // Every selected input difference a
    // ========================================================

    for &a_index
        in fixed_sample_rows
    {
        // ----------------------------------------------------
        // New logical histogram
        // ----------------------------------------------------

        *epoch =
            epoch.wrapping_add(1);

        if *epoch == 0 {
            stamps.fill(0);

            *epoch = 1;
        }

        let current_epoch =
            *epoch;

        let a_offset =
            a_index * s;

        // ====================================================
        // Every x
        // ====================================================

        for x_index in 0..n {
            let x_offset =
                x_index * s;

            // ------------------------------------------------
            // Calculate x + a and encode directly as an index.
            // ------------------------------------------------

            let mut xa_index =
                0usize;

            for i in 0..s {
                let x_i =
                    vector_table[
                        x_offset + i
                    ];

                let a_i =
                    vector_table[
                        a_offset + i
                    ];

                let xa_i =
                    fast_field.add(
                        x_i,
                        a_i,
                    );

                xa_index +=
                    xa_i
                        * place_values[i];
            }

            let xa_offset =
                xa_index * s;

            // ------------------------------------------------
            // Calculate
            //
            //     b = f(x+a) - f(x)
            //
            // and encode b directly as a base-q index.
            // ------------------------------------------------

            let mut b_index =
                0usize;

            for j in 0..s {
                let fx =
                    output_cache[
                        x_offset + j
                    ];

                let fxa =
                    output_cache[
                        xa_offset + j
                    ];

                let difference =
                    fast_field.sub(
                        fxa,
                        fx,
                    );

                b_index +=
                    difference
                        * place_values[j];
            }

            // ------------------------------------------------
            // Increment this DDT entry.
            // ------------------------------------------------

            if stamps[b_index]
                != current_epoch
            {
                stamps[b_index] =
                    current_epoch;

                counts[b_index] =
                    1;
            } else {
                counts[b_index] +=
                    1;
            }

            let count =
                counts[b_index]
                    as usize;

            // ------------------------------------------------
            // Dynamic early abort.
            //
            // If the current sampled best is 20 and this
            // candidate reaches 20, it cannot improve.
            // ------------------------------------------------

            if count >= cutoff {
                return None;
            }

            // ------------------------------------------------
            // Track sampled maximum.
            // ------------------------------------------------

            if count >
                max_sampled_delta
            {
                max_sampled_delta =
                    count;

                worst_sampled_a =
                    a_index;

                worst_sampled_b =
                    b_index;
            }
        }
    }

    Some(
        SampledResult {
            max_sampled_delta,
            worst_sampled_a,
            worst_sampled_b,
        }
    )
}


// ============================================================
// Output
// ============================================================

fn write_header(
    writer: &mut BufWriter<File>,
    matrix: &FieldMatrix,
    q: usize,
    s: usize,
    n: usize,
    config: &SampledRandomSearchConfig,
    difference_count: usize,
    fixed_sample_rows: &[usize],
    vector_table: &[Symbol],
) -> std::io::Result<()> {
    writeln!(
        writer,
        "========== STRICT SAMPLED RANDOM SEARCH =========="
    )?;

    writeln!(
        writer,
        "q: {}",
        q
    )?;

    writeln!(
        writer,
        "s: {}",
        s
    )?;

    writeln!(
        writer,
        "q^s: {}",
        n
    )?;

    writeln!(
        writer,
        "trials: {}",
        config.max_trials
    )?;

    writeln!(
        writer,
        "seed: {}",
        config.seed
    )?;

    writeln!(
        writer,
        "representative differences: {}",
        difference_count
    )?;

    writeln!(
        writer,
        "fixed sampled rows: {}",
        fixed_sample_rows.len()
    )?;

    writeln!(
        writer,
        "sample fraction: {:.10}%",
        100.0
            * fixed_sample_rows.len()
                as f64
            / difference_count
                .max(1)
                as f64
    )?;

    writeln!(
        writer,
        "Cauchy matrix:"
    )?;

    for row in matrix {
        writeln!(
            writer,
            "{:?}",
            row
        )?;
    }

    writeln!(
        writer,
        "Fixed input differences:"
    )?;

    for &a_index
        in fixed_sample_rows
    {
        writeln!(
            writer,
            "{:?}",
            decode_index(
                a_index,
                vector_table,
                s,
            )
        )?;
    }

    writeln!(writer)?;

    writer.flush()
}


fn write_best_record(
    writer: &mut BufWriter<File>,
    trial: u64,
    elapsed: Duration,
    previous_best: usize,
    result: &SampledResult,
    candidate: &Candidate,
    vector_table: &[Symbol],
    s: usize,
    sampled_rows: usize,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "=========================================="
    )?;

    writeln!(
        writer,
        "NEW SAMPLED BEST"
    )?;

    writeln!(
        writer,
        "trial: {}",
        trial
    )?;

    writeln!(
        writer,
        "elapsed_seconds: {:.6}",
        elapsed.as_secs_f64()
    )?;

    writeln!(
        writer,
        "previous_sampled_best: {}",
        previous_best
    )?;

    writeln!(
        writer,
        "new_sampled_best: {}",
        result.max_sampled_delta
    )?;

    writeln!(
        writer,
        "fixed_rows_tested: {}",
        sampled_rows
    )?;

    writeln!(
        writer,
        "worst_sampled_input_difference: {:?}",
        decode_index(
            result.worst_sampled_a,
            vector_table,
            s,
        )
    )?;

    writeln!(
        writer,
        "corresponding_output_difference: {:?}",
        decode_index(
            result.worst_sampled_b,
            vector_table,
            s,
        )
    )?;

    writeln!(
        writer,
        "input permutations:"
    )?;

    for (i, permutation)
        in candidate
            .input_perms
            .iter()
            .enumerate()
    {
        writeln!(
            writer,
            "tau_{} = {:?}",
            i,
            permutation
        )?;
    }

    writeln!(
        writer,
        "output permutations:"
    )?;

    for (i, permutation)
        in candidate
            .output_perms
            .iter()
            .enumerate()
    {
        writeln!(
            writer,
            "pi_{} = {:?}",
            i,
            permutation
        )?;
    }

    writeln!(writer)?;

    writer.flush()
}


// ============================================================
// Main search
// ============================================================

pub fn parallel_sampled_random_search(
    matrix: &FieldMatrix,
    field: &dyn FiniteFieldOps,
    config: &SampledRandomSearchConfig,
) -> std::io::Result<SampledRandomSearchStats> {
    // ========================================================
    // Validate input
    // ========================================================

    let s =
        matrix.len();

    assert!(
        s > 0,
        "matrix must not be empty"
    );

    assert!(
        config.sampled_rows > 0,
        "sampled_rows must be > 0"
    );

    for row in matrix {
        assert_eq!(
            row.len(),
            s,
            "search expects an s x s matrix"
        );
    }

    let q =
        field.order();

    let n =
        vector_space_size(
            q,
            s,
        );

    assert!(
        n <= u32::MAX as usize,
        "this implementation expects q^s <= u32::MAX"
    );

    // ========================================================
    // Shared lookup tables
    // ========================================================

    println!();

    println!(
        "========== Strict sampled low-DU search =========="
    );

    println!(
        "q = {}",
        q
    );

    println!(
        "s = {}",
        s
    );

    println!(
        "q^s = {}",
        n
    );

    println!(
        "Trials = {}",
        config.max_trials
    );

    println!(
        "Requested sampled rows = {}",
        config.sampled_rows
    );

    println!(
        "Seed = {}",
        config.seed
    );

    println!(
        "Rayon threads = {}",
        rayon::current_num_threads()
    );

    println!(
        "Building lookup tables..."
    );

    let fast_field =
        FastField::new(
            field
        );

    let vector_table =
        build_vector_table(
            q,
            s,
            n,
        );

    let place_values =
        build_place_values(
            q,
            s,
        );

    let matrix_products =
        build_matrix_product_table(
            matrix,
            field,
        );

    let difference_representatives =
        build_difference_representatives(
            &vector_table,
            &place_values,
            &fast_field,
            s,
            n,
        );

    let fixed_sample_rows =
        choose_fixed_sample_rows(
            &difference_representatives,
            config.sampled_rows,
            config.seed,
        );

    println!(
        "Nonzero differences = {}",
        n - 1
    );

    println!(
        "Representative differences = {}",
        difference_representatives.len()
    );

    println!(
        "Fixed sampled rows = {}",
        fixed_sample_rows.len()
    );

    println!(
        "Sample fraction = {:.8}%",
        100.0
            * fixed_sample_rows.len()
                as f64
            / difference_representatives
                .len()
                .max(1)
                as f64
    );

    // ========================================================
    // Output file
    // ========================================================

    let file =
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(
                &config.output_path
            )?;

    let mut initial_writer =
        BufWriter::new(
            file
        );

    write_header(
        &mut initial_writer,
        matrix,
        q,
        s,
        n,
        config,
        difference_representatives.len(),
        &fixed_sample_rows,
        &vector_table,
    )?;

    let writer =
        Mutex::new(
            initial_writer
        );

    // ========================================================
    // Shared search state
    // ========================================================

    // Start above the theoretical maximum so the first
    // candidate necessarily completes and establishes the
    // initial sampled record.
    let best_sampled_delta =
        AtomicUsize::new(
            n + 1
        );

    let completed_trials =
        AtomicU64::new(0);

    let improvements =
        AtomicU64::new(0);

    let fully_sampled =
        AtomicU64::new(0);

    let early_aborted =
        AtomicU64::new(0);

    let last_progress =
        AtomicU64::new(0);

    let start =
        Instant::now();

    // ========================================================
    // Parallel search
    // ========================================================

    (1..=config.max_trials)
        .into_par_iter()
        .map_init(
            || {
                WorkerWorkspace::new(
                    n,
                    s,
                )
            },

            |workspace, trial| {
                // ============================================
                // Candidate RNG
                // ============================================

                let seed =
                    trial_seed(
                        config.seed,
                        trial,
                    );

                let mut rng =
                    StdRng::seed_from_u64(
                        seed
                    );

                // Keep sequence stable if more RNG calls are
                // added around candidate generation later.
                let _ =
                    rng.next_u64();

                // ============================================
                // Random candidate
                // ============================================

                let candidate =
                    random_candidate(
                        s,
                        q,
                        &mut rng,
                    );

                // ============================================
                // Build f(x)
                // ============================================

                build_candidate_outputs(
                    &candidate,
                    &vector_table,
                    &matrix_products,
                    &fast_field,
                    s,
                    n,
                    &mut workspace
                        .output_cache,
                );

                // ============================================
                // Read current global sampled record
                // ============================================

                let cutoff =
                    best_sampled_delta.load(
                        Ordering::Relaxed
                    );

                // ============================================
                // Strict sampled DU
                // ============================================

                let result =
                    sampled_differential_uniformity(
                        &workspace.output_cache,
                        &vector_table,
                        &place_values,
                        &fixed_sample_rows,
                        &fast_field,
                        s,
                        n,
                        cutoff,
                        &mut workspace.counts,
                        &mut workspace.stamps,
                        &mut workspace.epoch,
                    );

                // ============================================
                // Process result
                // ============================================

                match result {
                    // ----------------------------------------
                    // Candidate reached the current sampled
                    // record before completing the sample.
                    // ----------------------------------------

                    None => {
                        early_aborted.fetch_add(
                            1,
                            Ordering::Relaxed,
                        );
                    }

                    // ----------------------------------------
                    // Candidate completed every sampled row.
                    // ----------------------------------------

                    Some(result) => {
                        fully_sampled.fetch_add(
                            1,
                            Ordering::Relaxed,
                        );

                        let sampled_delta =
                            result.max_sampled_delta;

                        let mut old_best =
                            best_sampled_delta.load(
                                Ordering::Relaxed
                            );

                        // ------------------------------------
                        // Atomic record update.
                        // ------------------------------------

                        loop {
                            if sampled_delta
                                >= old_best
                            {
                                break;
                            }

                            match best_sampled_delta
                                .compare_exchange_weak(
                                    old_best,
                                    sampled_delta,
                                    Ordering::SeqCst,
                                    Ordering::Relaxed,
                                )
                            {
                                // ----------------------------
                                // New global sampled record.
                                // ----------------------------

                                Ok(_) => {
                                    let improvement_number =
                                        improvements
                                            .fetch_add(
                                                1,
                                                Ordering::Relaxed,
                                            )
                                            + 1;

                                    let elapsed =
                                        start.elapsed();

                                    println!();

                                    println!(
                                        "******** NEW SAMPLED BEST #{} ********",
                                        improvement_number
                                    );

                                    println!(
                                        "Trial: {}",
                                        trial
                                    );

                                    println!(
                                        "Previous sampled delta: {}",
                                        old_best
                                    );

                                    println!(
                                        "New sampled delta: {}",
                                        sampled_delta
                                    );

                                    println!(
                                        "Rows tested: {}",
                                        fixed_sample_rows.len()
                                    );

                                    println!(
                                        "Elapsed: {:.2}s",
                                        elapsed.as_secs_f64()
                                    );

                                    println!(
                                        "***************************************"
                                    );

                                    println!();

                                    let mut guard =
                                        writer
                                            .lock()
                                            .expect(
                                                "output writer mutex poisoned"
                                            );

                                    write_best_record(
                                        &mut guard,
                                        trial,
                                        elapsed,
                                        old_best,
                                        &result,
                                        &candidate,
                                        &vector_table,
                                        s,
                                        fixed_sample_rows.len(),
                                    )
                                    .expect(
                                        "failed to write sampled best"
                                    );

                                    break;
                                }

                                // ----------------------------
                                // Another worker changed the
                                // record while we were working.
                                // ----------------------------

                                Err(actual_best) => {
                                    old_best =
                                        actual_best;
                                }
                            }
                        }
                    }
                }

                // ============================================
                // Progress
                // ============================================

                let completed =
                    completed_trials
                        .fetch_add(
                            1,
                            Ordering::Relaxed,
                        )
                        + 1;

                if config.progress_interval > 0 {
                    let milestone =
                        completed
                            / config
                                .progress_interval;

                    let previous =
                        last_progress.load(
                            Ordering::Relaxed
                        );

                    if milestone > previous
                        && last_progress
                            .compare_exchange(
                                previous,
                                milestone,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                    {
                        let elapsed =
                            start.elapsed();

                        let seconds =
                            elapsed
                                .as_secs_f64()
                                .max(
                                    f64::EPSILON
                                );

                        let rate =
                            completed as f64
                                / seconds;

                        println!(
                            concat!(
                                "completed {:>10} | ",
                                "best sampled {:>6} | ",
                                "improvements {:>5} | ",
                                "full {:>8} | ",
                                "aborted {:>10} | ",
                                "{:>8.2}/s"
                            ),
                            completed,
                            best_sampled_delta.load(
                                Ordering::Relaxed
                            ),
                            improvements.load(
                                Ordering::Relaxed
                            ),
                            fully_sampled.load(
                                Ordering::Relaxed
                            ),
                            early_aborted.load(
                                Ordering::Relaxed
                            ),
                            rate,
                        );
                    }
                }
            },
        )
        .for_each(drop);

    // ========================================================
    // Final statistics
    // ========================================================

    let elapsed =
        start.elapsed();

    let final_best =
        best_sampled_delta.load(
            Ordering::Relaxed
        );

    let final_improvements =
        improvements.load(
            Ordering::Relaxed
        );

    let final_fully_sampled =
        fully_sampled.load(
            Ordering::Relaxed
        );

    let final_early_aborted =
        early_aborted.load(
            Ordering::Relaxed
        );

    println!();

    println!(
        "========== Strict sampled search complete =========="
    );

    println!(
        "Trials: {}",
        config.max_trials
    );

    println!(
        "Best sampled delta: {}",
        final_best
    );

    println!(
        "Improvements: {}",
        final_improvements
    );

    println!(
        "Fully sampled: {}",
        final_fully_sampled
    );

    println!(
        "Early aborted: {}",
        final_early_aborted
    );

    println!(
        "Fixed sampled rows: {}",
        fixed_sample_rows.len()
    );

    println!(
        "Elapsed: {:.2} seconds",
        elapsed.as_secs_f64()
    );

    println!(
        "Average throughput: {:.2} candidates/s",
        config.max_trials as f64
            / elapsed
                .as_secs_f64()
                .max(
                    f64::EPSILON
                )
    );

    println!(
        "Best candidates written to: {}",
        config.output_path
    );

    println!(
        "NOTE: best sampled delta is exact over the sampled \
         rows, but is not necessarily the true differential \
         uniformity."
    );

    Ok(
        SampledRandomSearchStats {
            trials:
                config.max_trials,

            best_sampled_delta:
                final_best,

            improvements:
                final_improvements,

            fully_sampled:
                final_fully_sampled,

            early_aborted:
                final_early_aborted,

            elapsed,
        }
    )
}
