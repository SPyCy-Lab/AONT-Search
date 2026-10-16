use std::mem::size_of;
use std::time::{Duration, Instant};

use metal::*;

use crate::field::{
    FieldMatrix,
    FiniteFieldOps,
};
use crate::types::Symbol;



// ============================================================
// Public result
// ============================================================

#[derive(Debug)]
pub struct MetalFlatVerifierResult {
    pub delta: usize,

    pub best_input_difference: Vec<Symbol>,
    pub best_output_difference: Vec<Symbol>,

    pub rows_checked: usize,

    pub batches: usize,

    pub elapsed: Duration,

    pub rows_per_second: f64,
}


// ============================================================
// Metal parameters
// ============================================================

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MetalParameters {
    q: u32,
    s: u32,
    n: u32,
    batch_size: u32,
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


// ============================================================
// Every vector in F_q^s
// ============================================================

fn build_vector_table(
    q: usize,
    s: usize,
    n: usize,
) -> Vec<u32> {
    let mut table =
        vec![0u32; n * s];

    for index in 0..n {
        let mut value =
            index;

        for i in (0..s).rev() {
            table[
                index * s + i
            ] =
                (value % q) as u32;

            value /= q;
        }
    }

    table
}


// ============================================================
// Field lookup tables
// ============================================================

fn build_add_table(
    field: &dyn FiniteFieldOps,
    q: usize,
) -> Vec<u32> {
    let mut table =
        vec![0u32; q * q];

    for a in 0..q {
        for b in 0..q {
            table[
                a * q + b
            ] =
                field.add(
                    a,
                    b,
                ) as u32;
        }
    }

    table
}


fn build_sub_table(
    field: &dyn FiniteFieldOps,
    q: usize,
) -> Vec<u32> {
    let mut table =
        vec![0u32; q * q];

    for a in 0..q {
        for b in 0..q {
            table[
                a * q + b
            ] =
                field.sub(
                    a,
                    b,
                ) as u32;
        }
    }

    table
}


// ============================================================
// Candidate mapping
// ============================================================

fn map_point(
    x: &[Symbol],
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
) -> Vec<Symbol> {
    let s =
        x.len();


    // ========================================================
    // Input permutations
    // ========================================================

    let mut permuted_input =
        vec![0; s];


    for i in 0..s {
        permuted_input[i] =
            input_perms[i][x[i]];
    }


    // ========================================================
    // Row-vector matrix multiplication
    //
    // y[j] = sum_i x[i] M[i][j]
    // ========================================================

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


    // ========================================================
    // Output permutations
    // ========================================================

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
// Output cache
// ============================================================

fn build_output_cache(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
    q: usize,
    s: usize,
    n: usize,
) -> Vec<u32> {
    let mut cache =
        vec![0u32; n * s];


    for x_index in 0..n {
        let x =
            decode_index(
                x_index,
                q,
                s,
            );


        let y =
            map_point(
                &x,
                matrix,
                input_perms,
                output_perms,
                field,
            );


        for j in 0..s {
            cache[
                x_index * s + j
            ] =
                y[j] as u32;
        }
    }


    cache
}


// ============================================================
// Representative differences
//
// Keep one from each:
//
//     {a, -a}
//
// ============================================================

fn build_difference_representatives(
    field: &dyn FiniteFieldOps,
    vector_table: &[u32],
    s: usize,
    n: usize,
    place_values: &[usize],
) -> Vec<u32> {
    let mut representatives =
        Vec::new();


    for a_index in 1..n {
        let a_offset =
            a_index * s;


        let mut negative_index =
            0usize;


        for i in 0..s {
            let a_i =
                vector_table[
                    a_offset + i
                ] as usize;


            let negative =
                field.sub(
                    0,
                    a_i,
                );


            negative_index +=
                negative
                * place_values[i];
        }


        if a_index <= negative_index {
            representatives.push(
                a_index as u32
            );
        }
    }


    representatives
}


// ============================================================
// Metal buffer helper
// ============================================================

fn create_u32_buffer(
    device: &DeviceRef,
    data: &[u32],
) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,

        std::mem::size_of_val(data)
            as u64,

        MTLResourceOptions::StorageModeShared,
    )
}


// ============================================================
// Main exact batched GPU verifier
// ============================================================

pub fn metal_flat_verify(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,

    // Number of DDT rows simultaneously processed.
    //
    // Start with 16.
    batch_size: usize,
) -> Result<MetalFlatVerifierResult, String> {
    let start =
        Instant::now();


    // ========================================================
    // Validate
    // ========================================================

    assert!(
        batch_size > 0,
        "batch_size must be positive"
    );


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


    if n > u32::MAX as usize {
        return Err(
            format!(
                "q^s={} exceeds GPU u32 indexing limit",
                n
            )
        );
    }


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


    // ========================================================
    // Report
    // ========================================================

    println!();

    println!(
        "========== METAL FLAT EXACT VERIFIER =========="
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
        "Requested batch size = {}",
        batch_size
    );


    // ========================================================
    // CPU preprocessing
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


    println!(
        "Building vector table..."
    );


    let vector_table =
        build_vector_table(
            q,
            s,
            n,
        );


    println!(
        "Building field lookup tables..."
    );


    let add_table =
        build_add_table(
            field,
            q,
        );


    let sub_table =
        build_sub_table(
            field,
            q,
        );


    println!(
        "Building output cache..."
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


    println!(
        "Building representative differences..."
    );


    let representatives =
        build_difference_representatives(
            field,
            &vector_table,
            s,
            n,
            &place_values,
        );


    let total_rows =
        representatives.len();


    println!(
        "Nonzero differences = {}",
        n - 1
    );


    println!(
        "Representative rows = {}",
        total_rows
    );


    // ========================================================
    // Convert place values to GPU representation
    // ========================================================

    let place_values_u32:
        Vec<u32> =
        place_values
            .iter()
            .map(
                |&value| {
                    u32::try_from(value)
                        .expect(
                            "place value exceeds u32"
                        )
                }
            )
            .collect();


    // ========================================================
    // Metal setup
    // ========================================================

    println!(
        "Creating Metal device..."
    );


    let device =
        Device::system_default()
            .ok_or_else(
                || "No Metal device available"
                    .to_string()
            )?;


    println!(
        "Metal device: {}",
        device.name()
    );


    let command_queue =
        device.new_command_queue();


    // ========================================================
    // Compile kernel
    // ========================================================

    println!(
        "Compiling Metal kernel..."
    );


    let source =
        include_str!(
            "metal_flat_verifier.metal"
        );


    let options =
        CompileOptions::new();


    let library =
        device
            .new_library_with_source(
                source,
                &options,
            )
            .map_err(
                |error| {
                    format!(
                        "Metal compilation failed: {}",
                        error
                    )
                }
            )?;


    let function =
        library
            .get_function(
                "ddt_batch",
                None,
            )
            .map_err(
                |error| {
                    format!(
                        "Could not load ddt_batch: {}",
                        error
                    )
                }
            )?;


    let pipeline =
        device
            .new_compute_pipeline_state_with_function(
                &function,
            )
            .map_err(
                |error| {
                    format!(
                        "Could not create Metal pipeline: {}",
                        error
                    )
                }
            )?;


    // ========================================================
    // Permanent GPU buffers
    // ========================================================

    println!(
        "Creating GPU buffers..."
    );


    let vector_buffer =
        create_u32_buffer(
            &device,
            &vector_table,
        );


    let output_buffer =
        create_u32_buffer(
            &device,
            &output_cache,
        );


    let add_buffer =
        create_u32_buffer(
            &device,
            &add_table,
        );


    let sub_buffer =
        create_u32_buffer(
            &device,
            &sub_table,
        );


    let place_buffer =
        create_u32_buffer(
            &device,
            &place_values_u32,
        );


    // ========================================================
    // Reusable batch buffers
    // ========================================================

    // a indices
    let a_buffer_bytes =
        batch_size
            .checked_mul(
                size_of::<u32>()
            )
            .expect(
                "a buffer size overflow"
            );


    let a_buffer =
        device.new_buffer(
            a_buffer_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );


    // --------------------------------------------------------
    // Histograms
    //
    // batch_size * n * sizeof(u32)
    // --------------------------------------------------------

    let histogram_entries =
        batch_size
            .checked_mul(n)
            .ok_or_else(
                || "histogram entry count overflow"
                    .to_string()
            )?;


    let histogram_bytes =
        histogram_entries
            .checked_mul(
                size_of::<u32>()
            )
            .ok_or_else(
                || "histogram byte count overflow"
                    .to_string()
            )?;


    println!(
        "Histogram memory = {:.2} MB",
        histogram_bytes as f64
            / (1024.0 * 1024.0)
    );


    let histogram_buffer =
        device.new_buffer(
            histogram_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );


    // ========================================================
    // Global best
    // ========================================================

    let mut global_delta =
        0usize;


    let mut best_a_index =
        0usize;


    let mut best_b_index =
        0usize;


    // ========================================================
    // Batch loop
    // ========================================================

    let number_of_batches =
        (
            total_rows
            + batch_size
            - 1
        )
        / batch_size;


    println!(
        "Number of batches = {}",
        number_of_batches
    );


    println!(
        "Starting exact GPU verification..."
    );


    for batch_number in 0..number_of_batches {
        let row_start =
            batch_number
                * batch_size;


        let row_end =
            (
                row_start
                + batch_size
            )
            .min(
                total_rows
            );


        let current_batch_size =
            row_end
                - row_start;


        // ====================================================
        // Copy a indices into shared GPU buffer
        // ====================================================

        unsafe {
            let destination =
                a_buffer
                    .contents()
                    as *mut u32;


            for row in 0..current_batch_size {
                *destination.add(row) =
                    representatives[
                        row_start + row
                    ];
            }
        }


        // ====================================================
        // Clear histogram
        // ====================================================

        let active_histogram_entries =
            current_batch_size
                * n;


        let active_histogram_bytes =
            active_histogram_entries
                * size_of::<u32>();


        unsafe {
            std::ptr::write_bytes(
                histogram_buffer
                    .contents()
                    as *mut u8,

                0,

                active_histogram_bytes,
            );
        }


        // ====================================================
        // Parameters
        // ====================================================

        let params =
            MetalParameters {
                q:
                    q as u32,

                s:
                    s as u32,

                n:
                    n as u32,

                batch_size:
                    current_batch_size
                        as u32,
            };


        let params_buffer =
            device.new_buffer_with_data(
                &params
                    as *const MetalParameters
                    as *const _,

                size_of::<MetalParameters>()
                    as u64,

                MTLResourceOptions::StorageModeShared,
            );


        // ====================================================
        // Command buffer
        // ====================================================

        let command_buffer =
            command_queue
                .new_command_buffer();


        let encoder =
            command_buffer
                .new_compute_command_encoder();


        encoder
            .set_compute_pipeline_state(
                &pipeline
            );


        encoder.set_buffer(
            0,
            Some(&vector_buffer),
            0,
        );


        encoder.set_buffer(
            1,
            Some(&output_buffer),
            0,
        );


        encoder.set_buffer(
            2,
            Some(&add_buffer),
            0,
        );


        encoder.set_buffer(
            3,
            Some(&sub_buffer),
            0,
        );


        encoder.set_buffer(
            4,
            Some(&place_buffer),
            0,
        );


        encoder.set_buffer(
            5,
            Some(&a_buffer),
            0,
        );


        encoder.set_buffer(
            6,
            Some(&histogram_buffer),
            0,
        );


        encoder.set_buffer(
            7,
            Some(&params_buffer),
            0,
        );


        // ====================================================
        // 2D dispatch
        //
        // width  = every x
        // height = every a in current batch
        // ====================================================

        let execution_width =
            pipeline
                .thread_execution_width();


        let threads_per_group =
            MTLSize {
                width:
                    execution_width,

                height:
                    1,

                depth:
                    1,
            };


        let grid =
            MTLSize {
                width:
                    n as u64,

                height:
                    current_batch_size
                        as u64,

                depth:
                    1,
            };


        encoder.dispatch_threads(
            grid,
            threads_per_group,
        );


        encoder.end_encoding();


        // ====================================================
        // Run batch
        // ====================================================

        command_buffer.commit();

        command_buffer
            .wait_until_completed();


        // ====================================================
        // Read histograms
        // ====================================================

        let histograms:
            &[u32] =
            unsafe {
                std::slice::from_raw_parts(
                    histogram_buffer
                        .contents()
                        as *const u32,

                    active_histogram_entries,
                )
            };


        // ====================================================
        // CPU reduction
        // ====================================================

        for row in 0..current_batch_size {
            let histogram_start =
                row * n;


            let histogram_end =
                histogram_start + n;


            let histogram =
                &histograms[
                    histogram_start
                        ..
                    histogram_end
                ];


            let mut row_delta =
                0usize;


            let mut row_best_b =
                0usize;


            // -----------------------------------------------
            // Sanity check:
            //
            // every x must contribute exactly once.
            // -----------------------------------------------

            let mut row_sum =
                0u64;


            for (
                b_index,
                &count,
            ) in histogram
                .iter()
                .enumerate()
            {
                let count_usize =
                    count as usize;


                row_sum +=
                    count as u64;


                if count_usize
                    > row_delta
                {
                    row_delta =
                        count_usize;


                    row_best_b =
                        b_index;
                }
            }


            assert_eq!(
                row_sum,
                n as u64,
                "GPU histogram row does not sum to q^s"
            );


            let a_index =
                representatives[
                    row_start + row
                ] as usize;


            // -----------------------------------------------
            // Global maximum
            // -----------------------------------------------

            if row_delta
                > global_delta
            {
                global_delta =
                    row_delta;


                best_a_index =
                    a_index;


                best_b_index =
                    row_best_b;


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
        }


        // ====================================================
        // Progress
        // ====================================================

        let completed =
            row_end;


        if batch_number == 0
            || (batch_number + 1) % 100 == 0
            || completed == total_rows
        {
            let elapsed =
                start
                    .elapsed()
                    .as_secs_f64();


            let rate =
                completed as f64
                / elapsed.max(
                    f64::EPSILON
                );


            let percent =
                100.0
                * completed as f64
                / total_rows as f64;


            println!(
                "Rows: {}/{} ({:.2}%) | delta {} | {:.2} rows/s",
                completed,
                total_rows,
                percent,
                global_delta,
                rate,
            );
        }
    }


    // ========================================================
    // Final result
    // ========================================================

    let elapsed =
        start.elapsed();


    let rows_per_second =
        total_rows as f64
        / elapsed
            .as_secs_f64()
            .max(
                f64::EPSILON
            );


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
        "========== METAL VERIFICATION COMPLETE =========="
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
        total_rows,
        total_rows
    );


    println!(
        "Batches: {}",
        number_of_batches
    );


    println!(
        "Elapsed: {:.3} seconds",
        elapsed.as_secs_f64()
    );


    println!(
        "Average rate: {:.2} DDT rows/s",
        rows_per_second
    );


    println!(
        "================================================="
    );


    Ok(
        MetalFlatVerifierResult {
            delta:
                global_delta,

            best_input_difference,

            best_output_difference,

            rows_checked:
                total_rows,

            batches:
                number_of_batches,

            elapsed,

            rows_per_second,
        }
    )
}
