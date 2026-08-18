/// Used to check if all NLAONTs are column-wise permutations of Linear AONTs.
use std::collections::HashMap;

use crate::mapping_io::parse_aont_mappings;
use crate::types::Mapping;

/// Summary returned by the column-wise equivalence checker.
#[derive(Debug, Clone)]
pub struct ColumnEquivalenceReport {
    pub linear_mappings: usize,
    pub nonlinear_mappings: usize,
    pub nonlinear_covered: usize,
    /// One-based indices in `nonlinear.txt` that were not found.
    pub missing_nonlinear_indices: Vec<usize>,
}

/// Return all permutations of `0..q`.
fn generate_symbol_permutations(q: usize) -> Vec<Vec<usize>> {
    fn recurse(
        q: usize,
        current: &mut Vec<usize>,
        used: &mut [bool],
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == q {
            result.push(current.clone());
            return;
        }

        for symbol in 0..q {
            if used[symbol] {
                continue;
            }

            used[symbol] = true;
            current.push(symbol);

            recurse(q, current, used, result);

            current.pop();
            used[symbol] = false;
        }
    }

    let mut result = Vec::new();
    let mut current = Vec::with_capacity(q);
    let mut used = vec![false; q];

    recurse(q, &mut current, &mut used, &mut result);

    result
}

/// Generate every choice of one symbol permutation for each of the `s` columns.
///
/// The number of returned choices is `(q!)^s`.
fn generate_column_perm_choices(q: usize, s: usize) -> Vec<Vec<Vec<usize>>> {
    fn recurse(
        depth: usize,
        s: usize,
        symbol_perms: &[Vec<usize>],
        current: &mut Vec<Vec<usize>>,
        result: &mut Vec<Vec<Vec<usize>>>,
    ) {
        if depth == s {
            result.push(current.clone());
            return;
        }

        for perm in symbol_perms {
            current.push(perm.clone());

            recurse(
                depth + 1,
                s,
                symbol_perms,
                current,
                result,
            );

            current.pop();
        }
    }

    let symbol_perms = generate_symbol_permutations(q);
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(s);

    recurse(
        0,
        s,
        &symbol_perms,
        &mut current,
        &mut result,
    );

    result
}

/// Apply independent symbol permutations to each input and output column.
fn apply_column_perms(
    mapping: &Mapping,
    input_perms: &[Vec<usize>],
    output_perms: &[Vec<usize>],
) -> Mapping {
    mapping
        .iter()
        .map(|(x, y)| {
            assert_eq!(
                x.len(),
                input_perms.len(),
                "input permutation count does not match mapping dimension",
            );

            assert_eq!(
                y.len(),
                output_perms.len(),
                "output permutation count does not match mapping dimension",
            );

            let new_x = x
                .iter()
                .enumerate()
                .map(|(column, &symbol)| input_perms[column][symbol])
                .collect();

            let new_y = y
                .iter()
                .enumerate()
                .map(|(column, &symbol)| output_perms[column][symbol])
                .collect();

            (new_x, new_y)
        })
        .collect()
}

/// Canonicalize a mapping so equality does not depend on the order in which
/// mapping rows appear in the file.
fn mapping_signature(mapping: &Mapping) -> Mapping {
    let mut signature = mapping.clone();

    signature.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    signature
}

/// Check whether every mapping in `nonlinear_file` occurs as a column-wise
/// symbol relabeling of at least one mapping in `linear_file`.
///
/// Flow:
///   1. all nonlinear mappings are canonicalized once and stored in a lookup;
///   2. one transformed linear mapping is generated;
///   3. it is immediately checked against the nonlinear lookup;
///   4. matching nonlinear mappings are marked as found;
///   5. the transformed linear mapping is then discarded;
///   6. the search stops early if every nonlinear mapping has been found.
pub fn check_nonlinear_are_columnwise_perms_of_linear(
    linear_file: &str,
    nonlinear_file: &str,
    q: usize,
    s: usize,
) -> std::io::Result<ColumnEquivalenceReport> {
    let linear_mappings = parse_aont_mappings(linear_file)?;
    let nonlinear_mappings = parse_aont_mappings(nonlinear_file)?;

    println!("linear mappings: {}", linear_mappings.len());
    println!("nonlinear mappings: {}", nonlinear_mappings.len());

    if nonlinear_mappings.is_empty() {
        println!("No nonlinear mappings to check.");

        return Ok(ColumnEquivalenceReport {
            linear_mappings: linear_mappings.len(),
            nonlinear_mappings: 0,
            nonlinear_covered: 0,
            missing_nonlinear_indices: Vec::new(),
        });
    }

    /*
    A signature can theoretically appear more than once in the nonlinear file,
    so each signature maps to all one-based indices where it occurs.
    */
    let mut nonlinear_lookup: HashMap<Mapping, Vec<usize>> =
        HashMap::with_capacity(nonlinear_mappings.len());

    for (zero_based_index, mapping) in nonlinear_mappings.iter().enumerate() {
        nonlinear_lookup
            .entry(mapping_signature(mapping))
            .or_default()
            .push(zero_based_index);
    }

    let mut found = vec![false; nonlinear_mappings.len()];
    let mut covered = 0usize;

    /*
    These are the possible choices of one symbol permutation per column.
    */
    let column_perm_choices = generate_column_perm_choices(q, s);

    println!(
        "column permutation choices per side: {}",
        column_perm_choices.len()
    );

    let transformations_per_linear =
        column_perm_choices.len() as u128
            * column_perm_choices.len() as u128;

    println!(
        "transformations per linear mapping: {}",
        transformations_per_linear
    );

    'linear_loop: for (linear_index, linear_mapping) in
        linear_mappings.iter().enumerate()
    {
        println!(
            "processing linear AONT {}/{}",
            linear_index + 1,
            linear_mappings.len(),
        );

        for input_perms in &column_perm_choices {
            for output_perms in &column_perm_choices {
        
                let transformed = apply_column_perms(
                    linear_mapping,
                    input_perms,
                    output_perms,
                );

                let signature = mapping_signature(&transformed);

                if let Some(indices) = nonlinear_lookup.get(&signature) {
                    for &index in indices {
                        if !found[index] {
                            found[index] = true;
                            covered += 1;

                            println!(
                                "  found nonlinear AONT {} ({}/{})",
                                index + 1,
                                covered,
                                nonlinear_mappings.len(),
                            );
                        }
                    }

                    if covered == nonlinear_mappings.len() {
                        println!(
                            "All nonlinear mappings found; stopping early."
                        );

                        break 'linear_loop;
                    }
                }

            }
        }
    }

    let missing_nonlinear_indices: Vec<usize> = found
        .iter()
        .enumerate()
        .filter_map(|(index, &was_found)| {
            if was_found {
                None
            } else {
                Some(index + 1)
            }
        })
        .collect();

    println!();
    println!("========== COLUMN EQUIVALENCE RESULT ==========");
    println!("nonlinear covered: {}", covered);
    println!(
        "nonlinear missing: {}",
        missing_nonlinear_indices.len()
    );

    if missing_nonlinear_indices.is_empty() {
        println!(
            "All nonlinear AONTs are column-wise permutations \
             of the linear AONTs."
        );
    } else {
        println!("missing nonlinear indices:");

        for index in &missing_nonlinear_indices {
            println!("{index}");
        }
    }

    println!("===============================================");

    Ok(ColumnEquivalenceReport {
        linear_mappings: linear_mappings.len(),
        nonlinear_mappings: nonlinear_mappings.len(),
        nonlinear_covered: covered,
        missing_nonlinear_indices,
    })
}
