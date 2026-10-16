#include <metal_stdlib>
using namespace metal;


// ============================================================
// Parameters
// ============================================================

struct Parameters {
    uint q;
    uint s;
    uint n;

    // Number of a rows in this batch.
    uint batch_size;
};


// ============================================================
// Batched exact DDT kernel
//
// Grid dimensions:
//
//     x = x_index
//     y = row within current batch
//
// So thread:
//
//     (x_index, row)
//
// computes:
//
//     b = f(x + a_row) - f(x)
//
// and atomically increments:
//
//     histograms[row][b]
//
// Finite-field arithmetic is performed ONLY through the
// CPU-generated lookup tables.
//
// Therefore the same kernel works for:
//
//     GF(8)
//     GF(11)
//     GF(31)
//     GF(43)
//     etc.
// ============================================================

kernel void ddt_batch(
    device const uint *vector_table       [[buffer(0)]],
    device const uint *output_cache       [[buffer(1)]],

    device const uint *add_table          [[buffer(2)]],
    device const uint *sub_table          [[buffer(3)]],

    device const uint *place_values       [[buffer(4)]],

    // a_indices[row] gives the encoded input difference
    // for this row of the batch.
    device const uint *a_indices          [[buffer(5)]],

    // Flat layout:
    //
    // histograms[row * n + b]
    device atomic_uint *histograms        [[buffer(6)]],

    constant Parameters &params           [[buffer(7)]],

    uint2 gid                             [[thread_position_in_grid]]
) {
    const uint x_index =
        gid.x;

    const uint row =
        gid.y;


    // ========================================================
    // Bounds
    // ========================================================

    if (x_index >= params.n) {
        return;
    }

    if (row >= params.batch_size) {
        return;
    }


    const uint q =
        params.q;

    const uint s =
        params.s;

    const uint n =
        params.n;


    // ========================================================
    // Input difference for this row
    // ========================================================

    const uint a_index =
        a_indices[row];


    const uint x_offset =
        x_index * s;

    const uint a_offset =
        a_index * s;


    // ========================================================
    // Compute index(x + a)
    // ========================================================

    uint xa_index =
        0;


    for (uint i = 0; i < s; ++i) {
        const uint x_i =
            vector_table[
                x_offset + i
            ];

        const uint a_i =
            vector_table[
                a_offset + i
            ];


        // ----------------------------------------------------
        // FIELD ADDITION
        //
        // The lookup table is what allows this kernel to work
        // correctly over extension fields such as GF(8).
        // ----------------------------------------------------

        const uint xa_i =
            add_table[
                a_i * q + x_i
            ];


        xa_index +=
            xa_i
            * place_values[i];
    }


    // ========================================================
    // Compute:
    //
    //     b = f(x+a) - f(x)
    // ========================================================

    const uint xa_offset =
        xa_index * s;


    uint b_index =
        0;


    for (uint j = 0; j < s; ++j) {
        const uint fx =
            output_cache[
                x_offset + j
            ];

        const uint fxa =
            output_cache[
                xa_offset + j
            ];


        // ----------------------------------------------------
        // FIELD SUBTRACTION
        // ----------------------------------------------------

        const uint difference =
            sub_table[
                fxa * q + fx
            ];


        b_index +=
            difference
            * place_values[j];
    }


    // ========================================================
    // Histogram update
    // ========================================================

    const uint histogram_index =
        row * n + b_index;


    atomic_fetch_add_explicit(
        &histograms[histogram_index],
        1u,
        memory_order_relaxed
    );
}

