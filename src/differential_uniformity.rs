/// Used to calulate the differential uniformity of each mapping.
/// Uses the standard definition of differetial uniformity.
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Error, ErrorKind, Write};

use crate::field::FiniteFieldOps;
use crate::hamming::generate_tuples;
use crate::mapping_io::{parse_aont_mappings, validate_complete_mapping};
use crate::types::{Mapping, Tuple};

#[derive(Debug, Clone)]
pub struct DifferentialUniformityResult {
    pub mapping_index: usize,
    pub delta: usize,
    pub best_input_difference: Tuple,
    pub best_output_difference: Tuple,
    pub mapping: Mapping,
}

pub fn differential_uniformity(
    mapping: &Mapping,
    field: &dyn FiniteFieldOps,
) -> std::io::Result<(usize, Tuple, Tuple)> {
    let Some((first_input, _)) = mapping.first() else {
        return Err(Error::new(ErrorKind::InvalidData, "Mapping is empty"));
    };

    let all_vectors = generate_tuples(first_input.len(), field.order());
    differential_uniformity_with_vectors(mapping, field, &all_vectors)
}

fn differential_uniformity_with_vectors(
    mapping: &Mapping,
    field: &dyn FiniteFieldOps,
    all_vectors: &[Tuple],
) -> std::io::Result<(usize, Tuple, Tuple)> {
    let Some((first_input, _)) = mapping.first() else {
        return Err(Error::new(ErrorKind::InvalidData, "Mapping is empty"));
    };

    let s = first_input.len();
    let lookup = validate_complete_mapping(mapping, s, field.order())?;
    let zero = vec![0; s];

    let mut best_delta = 0usize;
    let mut best_input_difference = zero.clone();
    let mut best_output_difference = zero;

    for input_difference in all_vectors
        .iter()
        .filter(|difference| difference.iter().any(|&x| x != 0))
    {
        let mut difference_counts: HashMap<Tuple, usize> = HashMap::new();

        for x in all_vectors {
            let x_plus_a = field.add_tuples(x, input_difference);
            let fx = lookup.get(x).expect("complete mapping is missing x");
            let f_x_plus_a = lookup
                .get(&x_plus_a)
                .expect("complete mapping is missing x+a");

            let output_difference = field.sub_tuples(f_x_plus_a, fx);
            let count = difference_counts.entry(output_difference.clone()).or_insert(0);
            *count += 1;

            if *count > best_delta {
                best_delta = *count;
                best_input_difference = input_difference.clone();
                best_output_difference = output_difference;
            }
        }
    }

    Ok((
        best_delta,
        best_input_difference,
        best_output_difference,
    ))
}

fn write_du_result(
    writer: &mut BufWriter<std::fs::File>,
    result: &DifferentialUniformityResult,
    s: usize,
    q: usize,
) -> std::io::Result<()> {
    writeln!(writer, "BEGIN_AONT_DU")?;
    writeln!(writer, "MAPPING_INDEX: {}", result.mapping_index)?;
    writeln!(writer, "SIZE: {s}  FIELD_SIZE: {q}")?;
    writeln!(writer, "DIFFERENTIAL_UNIFORMITY: {}", result.delta)?;
    writeln!(
        writer,
        "BEST_INPUT_DIFFERENCE: {:?}",
        result.best_input_difference
    )?;
    writeln!(
        writer,
        "BEST_OUTPUT_DIFFERENCE: {:?}",
        result.best_output_difference
    )?;
    writeln!(writer, "MAPPING:")?;

    for (source, target) in &result.mapping {
        writeln!(writer, "{source:?}->{target:?}")?;
    }

    writeln!(writer, "END_AONT_DU\n")?;
    Ok(())
}

pub fn analyze_nonlinear_differential_uniformity(
    field: &dyn FiniteFieldOps,
    input_path: &str,
    output_path: &str,
) -> std::io::Result<Vec<DifferentialUniformityResult>> {
    let mappings = parse_aont_mappings(input_path)?;

    if mappings.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("No AONT mappings were found in `{input_path}`"),
        ));
    }

    let q = field.order();
    let s = mappings[0]
        .first()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "The first mapping is empty"))?
        .0
        .len();
    let all_vectors = generate_tuples(s, q);

    let output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;
    let mut output = BufWriter::new(output_file);

    let mut results: Vec<DifferentialUniformityResult> = Vec::with_capacity(mappings.len());
    let mut best_result_index = 0usize;

    for (zero_based_index, mapping) in mappings.into_iter().enumerate() {
        let (delta, best_input_difference, best_output_difference) =
            differential_uniformity_with_vectors(&mapping, field, &all_vectors)?;

        let result = DifferentialUniformityResult {
            mapping_index: zero_based_index + 1,
            delta,
            best_input_difference,
            best_output_difference,
            mapping,
        };

        if results.is_empty() || result.delta < results[best_result_index].delta {
            best_result_index = results.len();
        }

        let mapping_s = result.mapping[0].0.len();
        write_du_result(&mut output, &result, mapping_s, q)?;
        results.push(result);
    }

    output.flush()?;

    let best = &results[best_result_index];
    println!("====================================");
    println!("Lowest differential uniformity: {}", best.delta);
    println!("Mapping index in {}: {}", input_path, best.mapping_index);
    println!("Best input difference: {:?}", best.best_input_difference);
    println!("Best output difference: {:?}", best.best_output_difference);
    println!("Mapping:");
    for (source, target) in &best.mapping {
        println!("{source:?} -> {target:?}");
    }
    println!("Full results written to: {}", output_path);
    println!("====================================");

    Ok(results)
}

pub fn analyze_default_nonlinear_file(
    field: &dyn FiniteFieldOps,
) -> std::io::Result<Vec<DifferentialUniformityResult>> {
    analyze_nonlinear_differential_uniformity(
        field,
        "nonlinear.txt",
        "nonlinear_differential_uniformity.txt",
    )
}
