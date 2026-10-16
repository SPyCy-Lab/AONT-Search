use crate::differential_uniformity::differential_uniformity;
use crate::field::{FieldMatrix, FiniteFieldOps};
use crate::types::{Mapping, Symbol, Tuple};
/// The verifier in this file is for Testing, and very slow in actual usage.
/// Use the serial_flat verifer or the GPU verifiers for real usage.

/// Generate every vector in F_q^s.
///
/// simple and un-optimized verifiver.
fn generate_vectors(q: usize, s: usize) -> Vec<Tuple> {
    let total = q.pow(s as u32);

    let mut vectors = Vec::with_capacity(total);

    for index in 0..total {
        let mut value = index;
        let mut vector = vec![0; s];

        for i in (0..s).rev() {
            vector[i] = value % q;
            value /= q;
        }

        vectors.push(vector);
    }

    vectors
}

/// Compute x * matrix using the row-vector convention.
fn multiply_by_matrix(
    x: &[Symbol],
    matrix: &FieldMatrix,
    field: &dyn FiniteFieldOps,
) -> Tuple {
    let s = x.len();

    let mut result = vec![0; s];

    for j in 0..s {
        let mut sum = 0;

        for i in 0..s {
            let product =
                field.mul(x[i], matrix[i][j]);

            sum = field.add(sum, product);
        }

        result[j] = sum;
    }

    result
}

/// Apply one independent symbol permutation to each coordinate.
fn apply_permutations(
    x: &[Symbol],
    permutations: &[Vec<Symbol>],
) -> Tuple {
    x.iter()
        .enumerate()
        .map(|(i, &value)| permutations[i][value])
        .collect()
}

/// Construct the mapping
///
///     f(x) = xC
///
/// for every x.
pub fn build_linear_mapping(
    matrix: &FieldMatrix,
    field: &dyn FiniteFieldOps,
) -> Mapping {
    let q = field.order();
    let s = matrix.len();

    let inputs = generate_vectors(q, s);

    inputs
        .into_iter()
        .map(|x| {
            let y =
                multiply_by_matrix(&x, matrix, field);

            (x, y)
        })
        .collect()
}

/// Construct
///
///     f'(x) = T2(T1(x) C)
///
/// explicitly for every x.
///
/// This is deliberately straightforward rather than optimized.
pub fn build_permuted_mapping(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
) -> Mapping {
    let q = field.order();
    let s = matrix.len();

    assert_eq!(input_perms.len(), s);
    assert_eq!(output_perms.len(), s);

    let inputs = generate_vectors(q, s);

    inputs
        .into_iter()
        .map(|x| {
            // T1(x)
            let permuted_input =
                apply_permutations(
                    &x,
                    input_perms,
                );

            // T1(x) C
            let linear_output =
                multiply_by_matrix(
                    &permuted_input,
                    matrix,
                    field,
                );

            // T2(T1(x) C)
            let y =
                apply_permutations(
                    &linear_output,
                    output_perms,
                );

            (x, y)
        })
        .collect()
}

/// Verify a result produced by the random search.
///
/// Calculates DU twice:
///
/// 1. Original Cauchy linear AONT
/// 2. After the supplied input/output permutations
pub fn verify_search_result(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
) -> std::io::Result<()> {
    let q = field.order();
    let s = matrix.len();

    let expected_linear_delta =
        q.pow(s as u32);

    println!();
    println!("========== VERIFY SEARCH RESULT ==========");
    println!("q = {}", q);
    println!("s = {}", s);
    println!("q^s = {}", expected_linear_delta);
    println!();

    // ========================================================
    // BEFORE PERMUTATIONS
    // ========================================================

    println!("Building original linear mapping...");

    let linear_mapping =
        build_linear_mapping(
            matrix,
            field,
        );

    println!(
        "Calculating differential uniformity before permutations..."
    );

    let (linear_delta, linear_a, linear_b) =
        differential_uniformity(
            &linear_mapping,
            field,
        )?;

    println!();
    println!("BEFORE PERMUTATIONS");
    println!("Delta = {}", linear_delta);
    println!("Best a = {:?}", linear_a);
    println!("Best b = {:?}", linear_b);

    assert_eq!(
        linear_delta,
        expected_linear_delta,
        "Linear Cauchy mapping should have Delta = q^s"
    );

    println!(
        "Check passed: Delta = q^s = {}",
        expected_linear_delta
    );

    // Free the first full mapping before constructing the second.
    drop(linear_mapping);

    // ========================================================
    // AFTER PERMUTATIONS
    // ========================================================

    println!();
    println!("Building permuted mapping...");

    let permuted_mapping =
        build_permuted_mapping(
            matrix,
            input_perms,
            output_perms,
            field,
        );

    println!(
        "Calculating differential uniformity after permutations..."
    );

    let (permuted_delta, permuted_a, permuted_b) =
        differential_uniformity(
            &permuted_mapping,
            field,
        )?;

    println!();
    println!("AFTER PERMUTATIONS");
    println!("Delta = {}", permuted_delta);
    println!("Best a = {:?}", permuted_a);
    println!("Best b = {:?}", permuted_b);

    println!();
    println!("==========================================");

    Ok(())
}
