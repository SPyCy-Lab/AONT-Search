use std::{ffi::{c_char, c_int, CStr}, time::{Duration, Instant}};
use crate::field::{FieldMatrix, FiniteFieldOps};
use crate::types::Symbol;

#[derive(Debug)]
pub struct CudaFlatVerifierResult {
    pub delta: usize,
    pub best_input_difference: Vec<Symbol>,
    pub best_output_difference: Vec<Symbol>,
    pub rows_checked: usize,
    pub batches: usize,
    pub elapsed: Duration,
    pub rows_per_second: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaRowResult { maximum: u32, index: u32, sum: u64 }

#[repr(C)] struct CudaContext { _private: [u8; 0] }
unsafe extern "C" {
    fn cuda_ddt_create(q:u32,s:u32,n:u32,max_batch:u32,
        vectors:*const u32,outputs:*const u32,add:*const u32,sub:*const u32,
        places:*const u32) -> *mut CudaContext;
    fn cuda_ddt_run_reduced(ctx:*mut CudaContext, a_indices:*const u32,
        batch_size:u32, results_out:*mut CudaRowResult) -> c_int;
    fn cuda_ddt_destroy(ctx:*mut CudaContext);
    fn cuda_ddt_last_error() -> *const c_char;
}
fn cuda_error() -> String {
    unsafe { CStr::from_ptr(cuda_ddt_last_error()).to_string_lossy().into_owned() }
}
struct DeviceContext(*mut CudaContext);
impl Drop for DeviceContext {
    fn drop(&mut self) { unsafe { cuda_ddt_destroy(self.0) } }
}

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


/// Exact batched differential-uniformity verification on one NVIDIA CUDA GPU.
pub fn cuda_flat_verify(
    matrix: &FieldMatrix,
    input_perms: &[Vec<Symbol>],
    output_perms: &[Vec<Symbol>],
    field: &dyn FiniteFieldOps,
    batch_size: usize,
) -> Result<CudaFlatVerifierResult, String> {
    let start = Instant::now();
    if batch_size == 0 { return Err("batch_size must be positive".into()); }
    let s = matrix.len();
    if s == 0 || matrix.iter().any(|row| row.len() != s) {
        return Err("matrix must be nonempty and square".into());
    }
    let q = field.order();
    let n = q.checked_pow(s as u32).ok_or("q^s overflow")?;
    if n > u32::MAX as usize { return Err("q^s exceeds u32 index range".into()); }
    let q32 = u32::try_from(q).map_err(|e|e.to_string())?;
    let s32 = u32::try_from(s).map_err(|e|e.to_string())?;
    let n32 = u32::try_from(n).map_err(|e|e.to_string())?;
    let batch32 = u32::try_from(batch_size).map_err(|e|e.to_string())?;
    if input_perms.len() != s || output_perms.len() != s ||
       input_perms.iter().chain(output_perms.iter()).any(|p|p.len()!=q) {
        return Err("invalid coordinate permutation dimensions".into());
    }
    let place_values = build_place_values(q,s);
    let vector_table = build_vector_table(q,s,n);
    let add_table = build_add_table(field,q);
    let sub_table = build_sub_table(field,q);
    let output_cache = build_output_cache(matrix,input_perms,output_perms,field,q,s,n);
    let representatives = build_difference_representatives(field,&vector_table,s,n,&place_values);
    let places: Vec<u32> = place_values.iter().map(|&v|u32::try_from(v).expect("place overflow")).collect();
    let total_rows = representatives.len();
    let mut row_results = vec![CudaRowResult::default(); batch_size];
    let ctx = unsafe { cuda_ddt_create(q32,s32,n32,batch32,
        vector_table.as_ptr(),output_cache.as_ptr(),add_table.as_ptr(),
        sub_table.as_ptr(),places.as_ptr()) };
    if ctx.is_null() { return Err(cuda_error()); }
    let ctx = DeviceContext(ctx);
    let batches = total_rows.div_ceil(batch_size);
    let mut global_delta = 0usize;
    let mut best_a = 0usize;
    let mut best_b = 0usize;
    println!("CUDA exact verifier: q={q}, s={s}, N={n}, rows={total_rows}, batch={batch_size}");
    for (batch_number, chunk) in representatives.chunks(batch_size).enumerate() {
        let active = chunk.len();
        let code = unsafe { cuda_ddt_run_reduced(ctx.0,chunk.as_ptr(),active as u32,row_results.as_mut_ptr()) };
        if code != 0 { return Err(cuda_error()); }
        for (row,&a_index) in chunk.iter().enumerate() {
            let result = row_results[row];
            let row_sum = result.sum;
            let row_delta = result.maximum as usize;
            let row_b = result.index as usize;
            if row_sum != n as u64 {
                return Err(format!("histogram sum mismatch at a={a_index}: {row_sum} != {n}"));
            }
            if row_delta > global_delta {
                global_delta=row_delta; best_a=a_index as usize; best_b=row_b;
                println!("NEW DELTA = {global_delta}, a={:?}, b={:?}",
                    decode_index(best_a,q,s),decode_index(best_b,q,s));
            }
        }
        let completed = ((batch_number+1)*batch_size).min(total_rows);
        if batch_number==0 || (batch_number+1)%100==0 || completed==total_rows {
            let rate=completed as f64/start.elapsed().as_secs_f64().max(f64::EPSILON);
            println!("Rows: {completed}/{total_rows} | delta {global_delta} | {rate:.2} rows/s");
        }
    }
    let elapsed = start.elapsed();
    let rows_per_second = total_rows as f64/elapsed.as_secs_f64().max(f64::EPSILON);
    Ok(CudaFlatVerifierResult {
        delta:global_delta,
        best_input_difference:decode_index(best_a,q,s),
        best_output_difference:decode_index(best_b,q,s),
        rows_checked:total_rows,batches,elapsed,rows_per_second,
    })
}

