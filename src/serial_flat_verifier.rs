use std::time::{Duration, Instant};

use crate::field::{
    FieldMatrix,
    FiniteFieldOps,
};
use crate::types::Symbol;


// ============================================================
// Result
// ============================================================

#[derive(Debug)]
pub struct SerialFlatVerifierResult {
    pub delta: usize,

    pub best_input_difference: Vec<Symbol>,
    pub best_output_difference: Vec<Symbol>,

    pub rows_checked: usize,

    pub elapsed: Duration,
}


// ============================================================
// Basic helpers
// ============================================================

fn vector_space_size(
    q: usize,
    s: usize,
) -> usize {
    q.checked_pow(s as u32)
        .expect("q^s does not fit in usize")
}


/// Base-q positional weights.
///
/// q=11, s=4:
///
///     [1331, 121, 11, 1]
fn build_place_values(
    q: usize,
    s: usize,
) -> Vec<usize> {
    let mut place =
        vec![1usize; s];

    if s >= 2 {
        for i in (0..s - 1).rev() {
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


/// Decode a base-q integer into a vector.
///
fn decode_index(
    mut index: usize,
    q: usize,
    s: usize,
) -> Vec<Symbol> {
    let mut vector =
        vec![0; s];

    for i in (0..s).rev() {
        vector[i] =
            index % q;

        index /= q;
    }

    vector
}


/// Encode a vector as a base-q integer.
fn encode_vector(
    vector: &[Symbol],
    place_values: &[usize],
) -> usize {
    vector
        .iter()
        .zip(place_values)
        .map(
            |(&x, &place)|
                x * place
        )
        .sum()
}


// ============================================================
// Reference mapping
// ============================================================

/// Straightforward implementation of
///
///     f(x) = T2(T1(x) M)
/// 
fn reference_map_point(
    x: &[Symbol],
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
) -> Vec<Symbol> {
    let s =
        x.len();

    // --------------------------------------------------------
    // T1(x)
    // --------------------------------------------------------

    let mut permuted_input =
        vec![0; s];

    for i in 0..s {
        permuted_input[i] =
            input_perms[i][x[i]];
    }

    // --------------------------------------------------------
    // T1(x) M
    //
    // Row-vector convention:
    //
    // y[j] = sum_i x[i] * M[i][j]
    // --------------------------------------------------------

    let mut linear_output =
        vec![0; s];

    for j in 0..s {
        let mut sum =
            0;

        for i in 0..s {
            let product =
                field.mul(
                    permuted_input[i],
                    matrix[i][j],
                );

            sum =
                field.add(
                    sum,
                    product,
                );
        }

        linear_output[j] =
            sum;
    }

    // --------------------------------------------------------
    // T2(...)
    // --------------------------------------------------------

    let mut output =
        vec![0; s];

    for j in 0..s {
        output[j] =
            output_perms[j][
                linear_output[j]
            ];
    }

    output
}


// ============================================================
// Build flat output cache
// ============================================================

/// output_cache stores
///
///     f(x)
///
/// for every x.
///
/// f(x_index)[j] is stored at:
///
///     output_cache[x_index * s + j]
fn build_output_cache(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
    q: usize,
    s: usize,
    n: usize,
) -> Vec<Symbol> {
    let mut output_cache =
        vec![0; n * s];

    for x_index in 0..n {
        let x =
            decode_index(
                x_index,
                q,
                s,
            );

        let y =
            reference_map_point(
                &x,
                matrix,
                input_perms,
                output_perms,
                field,
            );

        for j in 0..s {
            output_cache[
                x_index * s + j
            ] =
                y[j];
        }
    }

    output_cache
}


// ============================================================
// Mapping diagnostic
// ============================================================

/// Independently rebuild every mapping value and compare it
/// with output_cache.
///
/// This is intentionally redundant.
///
/// If this fails, the problem is mapping construction rather
/// than DDT calculation.
fn verify_output_cache(
    output_cache: &[Symbol],
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
    q: usize,
    s: usize,
    n: usize,
) {
    println!();
    println!("Checking output cache...");

    for x_index in 0..n {
        let x =
            decode_index(
                x_index,
                q,
                s,
            );

        let expected =
            reference_map_point(
                &x,
                matrix,
                input_perms,
                output_perms,
                field,
            );

        let actual =
            &output_cache[
                x_index * s
                    ..
                (x_index + 1) * s
            ];

        if expected.as_slice()
            != actual
        {
            println!();
            println!("MAPPING MISMATCH");
            println!("x        = {:?}", x);
            println!("expected = {:?}", expected);
            println!("actual   = {:?}", actual);

            panic!(
                "output cache does not match reference mapping"
            );
        }
    }

    println!(
        "Mapping comparison passed: {} / {} outputs identical.",
        n,
        n
    );
}


// ============================================================
// Representative input differences
// ============================================================

/// Build one representative from {a,-a}.
fn build_difference_representatives(
    field: &dyn FiniteFieldOps,
    q: usize,
    s: usize,
    n: usize,
    place_values: &[usize],
) -> Vec<usize> {
    let mut representatives =
        Vec::new();

    for a_index in 1..n {
        let a =
            decode_index(
                a_index,
                q,
                s,
            );

        let mut negative =
            vec![0; s];

        for i in 0..s {
            negative[i] =
                field.sub(
                    0,
                    a[i],
                );
        }

        let negative_index =
            encode_vector(
                &negative,
                place_values,
            );

        if a_index <= negative_index {
            representatives.push(
                a_index
            );
        }
    }

    representatives
}


// ============================================================
// One exact DDT row
// ============================================================

#[derive(Debug)]
struct RowResult {
    delta: usize,
    best_b_index: usize,
}


/// Compute the entire DDT row corresponding to one nonzero a.
///
///
///     for every x
///         xa = x + a
///         b  = f(xa) - f(x)
///         counts[b] += 1
///
fn calculate_ddt_row(
    a_index: usize,
    output_cache: &[Symbol],
    field: &dyn FiniteFieldOps,
    q: usize,
    s: usize,
    n: usize,
    place_values: &[usize],
) -> RowResult {
    let a =
        decode_index(
            a_index,
            q,
            s,
        );

    let mut counts =
        vec![0usize; n];

    // ========================================================
    // Every x
    // ========================================================

    for x_index in 0..n {
        let x =
            decode_index(
                x_index,
                q,
                s,
            );

        // ----------------------------------------------------
        // x + a
        // ----------------------------------------------------

        let mut xa =
            vec![0; s];

        for i in 0..s {
            xa[i] =
                field.add(
                    x[i],
                    a[i],
                );
        }

        let xa_index =
            encode_vector(
                &xa,
                place_values,
            );

        // ----------------------------------------------------
        // b = f(x+a) - f(x)
        // ----------------------------------------------------

        let mut b_index =
            0usize;

        for j in 0..s {
            let fx =
                output_cache[
                    x_index * s + j
                ];

            let fxa =
                output_cache[
                    xa_index * s + j
                ];

            let difference =
                field.sub(
                    fxa,
                    fx,
                );

            b_index +=
                difference
                    * place_values[j];
        }

        counts[b_index] += 1;
    }

    // ========================================================
    // Find row maximum
    // ========================================================

    let mut row_delta =
        0usize;

    let mut best_b_index =
        0usize;

    for (
        b_index,
        &count,
    ) in counts.iter().enumerate()
    {
        if count > row_delta {
            row_delta =
                count;

            best_b_index =
                b_index;
        }
    }

    RowResult {
        delta:
            row_delta,

        best_b_index,
    }
}


// ============================================================
// Main serial verifier
// ============================================================

pub fn serial_flat_verify(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
    progress_interval: usize,
) -> SerialFlatVerifierResult {
    let start =
        Instant::now();

    // ========================================================
    // Validate
    // ========================================================

    let s =
        matrix.len();

    assert!(
        s > 0,
        "matrix cannot be empty"
    );

    let q =
        field.order();

    let n =
        vector_space_size(
            q,
            s,
        );

    assert_eq!(
        input_perms.len(),
        s
    );

    assert_eq!(
        output_perms.len(),
        s
    );

    for permutation in input_perms {
        assert_eq!(
            permutation.len(),
            q
        );
    }

    for permutation in output_perms {
        assert_eq!(
            permutation.len(),
            q
        );
    }

    println!();
    println!(
        "========== SERIAL FLAT EXACT VERIFIER =========="
    );

    println!("q = {}", q);
    println!("s = {}", s);
    println!("q^s = {}", n);

    // ========================================================
    // Base-q representation
    // ========================================================

    let place_values =
        build_place_values(
            q,
            s,
        );

    println!(
        "Place values = {:?}",
        place_values
    );

    // ========================================================
    // Mapping
    // ========================================================

    println!(
        "Building flat output cache..."
    );

    let output_cache =
        build_output_cache(
            matrix,
            input_perms,
            output_perms,
            field,
            q,
            s,
            n,
        );

    // Redundant correctness check.
    verify_output_cache(
        &output_cache,
        matrix,
        input_perms,
        output_perms,
        field,
        q,
        s,
        n,
    );

    // ========================================================
    // Differences
    // ========================================================

    println!(
        "Building representative differences..."
    );

    let representatives =
        build_difference_representatives(
            field,
            q,
            s,
            n,
            &place_values,
        );

    println!(
        "Nonzero differences = {}",
        n - 1
    );

    println!(
        "Representative differences = {}",
        representatives.len()
    );

    // ========================================================
    // Exact DDT
    // ========================================================

    let mut global_delta =
        0usize;

    let mut best_a_index =
        0usize;

    let mut best_b_index =
        0usize;

    for (
        row_number,
        &a_index,
    ) in representatives
        .iter()
        .enumerate()
    {
        let result =
            calculate_ddt_row(
                a_index,
                &output_cache,
                field,
                q,
                s,
                n,
                &place_values,
            );

        if result.delta
            > global_delta
        {
            global_delta =
                result.delta;

            best_a_index =
                a_index;

            best_b_index =
                result.best_b_index;

            println!();
            println!(
                "NEW DELTA = {}",
                global_delta
            );

            println!(
                "a = {:?}",
                decode_index(
                    best_a_index,
                    q,
                    s,
                )
            );

            println!(
                "b = {:?}",
                decode_index(
                    best_b_index,
                    q,
                    s,
                )
            );
            println!();
        }

        let completed =
            row_number + 1;

        if progress_interval > 0
            && completed
                % progress_interval
                == 0
        {
            let elapsed =
                start.elapsed()
                    .as_secs_f64();

            let rate =
                completed as f64
                    / elapsed.max(
                        f64::EPSILON
                    );

            println!(
                "rows {}/{} | delta {} | {:.2} rows/s",
                completed,
                representatives.len(),
                global_delta,
                rate,
            );
        }
    }

    // ========================================================
    // Result
    // ========================================================

    let elapsed =
        start.elapsed();

    let best_input_difference =
        decode_index(
            best_a_index,
            q,
            s,
        );

    let best_output_difference =
        decode_index(
            best_b_index,
            q,
            s,
        );

    println!();
    println!(
        "========== SERIAL VERIFICATION COMPLETE =========="
    );

    println!(
        "Exact differential uniformity: {}",
        global_delta
    );

    println!(
        "Best input difference: {:?}",
        best_input_difference
    );

    println!(
        "Best output difference: {:?}",
        best_output_difference
    );

    println!(
        "Representative rows checked: {} / {}",
        representatives.len(),
        representatives.len()
    );

    println!(
        "Elapsed: {:.3} seconds",
        elapsed.as_secs_f64()
    );

    println!(
        "=================================================="
    );

    SerialFlatVerifierResult {
        delta:
            global_delta,

        best_input_difference,

        best_output_difference,

        rows_checked:
            representatives.len(),

        elapsed,
    }
}
