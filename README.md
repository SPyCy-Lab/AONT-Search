# AONT Search

A Rust research codebase for enumerating **All-Or-Nothing Transforms (AONTs)** and investigating their **differential uniformity**. It includes branch-and-bound enumeration for small parameters, a randomized search based on coordinate-wise permutations of linear AONTs, exact differential-uniformity verifiers, and statistical experiments.

The main executable is configured by editing `src/main.rs` and separate binaries are used to run the statistical experiments.

## Features

- **Branch-and-bound enumeration:** enumerate AONTs for small `(t,s,q)` parameters, with optional differential-uniformity analysis and column-wise permutation equivalence checks.
- **Randomized search:** start with a linear AONT from a Cauchy matrix, independently permute input and output coordinates, and screen candidates using a subset of differential-distribution-table (DDT) rows.
- **Exact verification:** compute the full differential uniformity of a selected candidate using the serial CPU verifier or, on macOS, the Metal verifier.
- **Statistical experiments:** generate independent random permutation sets, compute exact differential uniformity for each, and report mean, sample standard deviation, median, minimum, and maximum.
- **Reproducible examples:** `permutations_with_low_delta.txt` contains saved input/output permutations and reported differential uniformities.

## Requirements and installation

- Rust toolchain supporting **edition 2024**, with Cargo.
- Dependencies declared in `Cargo.toml`: `ark-ff`, `rand`, `rayon`, and `metal`.
- Apple Metal is used by the macOS GPU verifier and `delta_statistics` binary. The CUDA verifier and `delta_statistics_cuda` binary target Linux with NVIDIA CUDA; building/running them additionally requires an appropriate CUDA toolchain and working CUDA integration. **The provided `Cargo.toml` does not by itself document or configure the CUDA build/linking steps.**

From the repository root:

```bash
cargo build --release --bin aont-search
```

Platform-specific modules and dependencies may require adjustments when building outside macOS; do not assume the same build works unmodified on Linux.

## 1. Branch-and-bound enumeration

Open `src/main.rs` and find the section **BRANCH AND BOUND SEARCH TO ENUMERATE AONTS**.

### Choose a field

The included example defines `GF(3)` with Arkworks:

```rust
#[derive(MontConfig)]
#[modulus = "3"]
#[generator = "2"]
pub struct F3Config;
pub type F3 = Fp64<MontBackend<F3Config, 1>>;
let field = ArkPrimeField::<F3>::new(F3::from(2u64));
let q = 3usize;
```

To use a different prime field, update the modulus, generator, field type, and `q` consistently. The source also gives examples for `GF(4)` (using `Gf4`) and `GF(5)`.

### Set parameters and enable enumeration

```rust
let t = 2usize;
let s = 3usize;

let run_bnb = true;
let run_bnb_du_analysis = false;
let run_bnb_equivalence = false;
```

The two optional checks run only after enumeration, and only if the relevant mappings were found.

### Configure the search

```rust
let mut config = SearchConfig::new(t, s);
config.fix_zero = true;
config.first_nonlinear_to_keep = 5;
config.progress_report_interval = 1_000_000;
config.estimator_samples = 5_000;
```

| Setting | Purpose |
|---|---|
| `fix_zero` | Restrict enumeration to mappings satisfying `f(0)=0`; results are normalized representatives, not the unnormalized total. |
| `first_nonlinear_to_keep` | Number of nonlinear examples retained in memory; output files are handled separately. |
| `progress_report_interval` | Number of visited branches between progress reports. |
| `estimator_samples` | Number of randomized paths used for search-tree estimation. |

Run the main executable:

```bash
cargo run --release --bin aont-search
```

The enumeration writes `linear.txt` and `nonlinear.txt` according to the search configuration. Optional analysis may produce `nonlinear_differential_uniformity.txt`. Output paths and existing files should be checked before rerunning experiments.

## 2. Randomized search for low differential uniformity

In the second part of `src/main.rs`, find the section **RANDOMIZED SEARCH AND FULL DIFFERENTIAL UNIFORMITY VERIFICATION**. This part has its **own** field and dimension settings, independent of the BnB section.

The example uses `GF(11)` with `s=4`. It constructs a Cauchy matrix from two disjoint symbol sets:

```rust
let a: Vec<Symbol> = (0..s).collect();
let b: Vec<Symbol> = (s..2 * s).collect();
let matrix = cauchy_matrix(&a, &b, &field);
```

This particular construction requires **`q >= 2*s`**.

Enable the search and configure it:

```rust
let run_random_search = true;
let run_exact_verification = false;

let random_config = SampledRandomSearchConfig {
    max_trials: 100_000,
    sampled_rows: 10_000,
    seed: 123_456,
    output_path: format!("sampled_s{s}_q{q}.txt"),
    progress_interval: 1_000,
};
```

The search independently selects coordinate-wise input and output permutations and evaluates candidates using sampled DDT rows. **The sampled maximum is only a lower bound on exact differential uniformity**; it is not an exact result. Saved promising candidates should be checked using the full verifier.

Run the same main executable:

```bash
cargo run --release --bin aont-search
```

## 3. Exact differential-uniformity verification

In the same section of `src/main.rs`, set:

```rust
let run_random_search = false;
let run_exact_verification = true;
let use_metal = false; // true for the macOS Metal verifier
```

Set the field size and `s` to match the candidate. Copy the `input_perms` and `output_perms` vectors from `permutations_with_low_delta.txt` into the corresponding arrays in `src/main.rs`.

The current example is for `q=11, s=4` and records a candidate with **Δ = 23**. For a different parameter set, **replace both permutation arrays**: there must be `s` input and `s` output permutations, each containing the `q` symbols exactly once.

- With `use_metal = false`, `serial_flat_verify` runs on the CPU; `cpu_progress_interval` controls reporting.
- With `use_metal = true` on macOS, `metal_flat_verify` runs on the GPU; `metal_batch_size` controls the batch size.

Run:

```bash
cargo run --release --bin aont-search
```

This is an **exact** verification of the specified candidate, not a proof that its differential uniformity is minimal among all AONTs with the same parameters.

## 4. Statistical experiments

The statistical binaries independently generate `s` input and `s` output permutations for each trial, using Rust's `shuffle` functionality. They calculate the **exact** differential uniformity for every candidate (rather than screening candidates by sampled rows). By default both binaries use `TRIALS = 100`.

Each experiment records every trial's differential uniformity, a maximizing input/output difference, the input/output permutations, and elapsed time. Final statistics include the mean, standard deviation, median, minimum, and maximum.

### Apple Metal (macOS)

File: `src/bin/delta_statistics.rs`.

```bash
cargo run --release --bin delta_statistics -- 11 4
```

The first argument is `q` and the second is `s`. The file currently dispatches the prime fields `7, 11, 13, 17, 23, 31, 43, 257`. The batch size is configured in the source with `BATCH_SIZE = 8`.

### NVIDIA CUDA (Linux)

File: `src/bin/delta_statistics_cuda.rs`.

```bash
cargo run --release --bin delta_statistics_cuda -- 47 4
```

This command assumes that the repository's CUDA code has been correctly built and linked on a compatible Linux/NVIDIA system; the supplied Cargo manifest alone is not sufficient to verify that setup. This binary currently dispatches `7, 11, 13, 17, 23, 31, 43, 47, 257`, and uses `BATCH_SIZE = 16`.

Both programs require `q >= 2*s` for their Cauchy construction. To change the number of trials, edit `TRIALS` near the top of the appropriate source file.

**Output location:** each binary creates a file in the *current working directory*:

```text
delta_statistics_s4_q11.txt
delta_statistics_cuda_s4_q47.txt
```

The existing `Delta_statistics/` directory contains saved experiment records. The programs do **not** automatically write into that directory. Output files with the same name are overwritten when a new experiment starts, so back up existing results before rerunning.

## 5. Results and saved permutations

- `linear.txt`, `nonlinear.txt`: enumeration output.
- `nonlinear_differential_uniformity.txt`: differential-uniformity analysis of enumerated nonlinear mappings.
- `permutations_with_low_delta.txt`: recorded low-Δ candidate permutations, grouped by `(s,q)`; use them with the exact verifier to reproduce individual values.
- `Delta_statistics/`: saved results from the independent random-permutation experiments. The filename identifies the field size and number of coordinates; the `cuda` prefix distinguishes CUDA runs.

The saved permutations allow verification of the reported value for that particular construction.

## 6. Source-code guide

| File | Role |
|---|---|
| `src/main.rs` | Editable entry point for enumeration, sampled randomized search, and exact verification |
| `src/bnb.rs`, `src/search.rs`, `src/config.rs` | Enumeration and search configuration |
| `src/field.rs` | Finite-field operations and matrix constructions |
| `src/sampled_random_search.rs` | Parallel randomized candidate screening |
| `src/serial_flat_verifier.rs` | Serial exact differential-uniformity verifier |
| `src/metal_flat_verifier.rs`, `.metal` | Apple Metal verifier and shader |
| `src/cuda_flat_verifier.rs`, `.cu` | CUDA verifier and kernel |
| `src/bin/delta_statistics.rs` | Metal statistical experiment executable |
| `src/bin/delta_statistics_cuda.rs` | CUDA statistical experiment executable |
| `src/differential_uniformity.rs` | Differential-uniformity analysis for enumerated mappings |
| `src/equivalence.rs`, `src/linearity.rs` | Equivalence and linearity analysis |
| `src/mapping_io.rs`, `src/types.rs` | Mapping I/O and shared types |
| `src/hamming.rs`, `src/time_estimator.rs` | Hamming-distance utilities and search-time estimation |
| `src/verify_search_result.rs` | Utilities for constructing/checking search-result mappings |
| `src/lib.rs` | Library module exports |

## 7. Reproducing an example

To verify the saved `(s=4, q=11)` example:

1. In the randomized-search/verification section of `src/main.rs`, keep the `GF(11)` field and set `s=4`.
2. Set `run_random_search = false` and `run_exact_verification = true`.
3. Use the supplied `input_perms` and `output_perms` (or copy them from `permutations_with_low_delta.txt`).
4. Set `use_metal = false` for serial verification, or `true` for macOS Metal verification.
5. Run `cargo run --release --bin aont-search` and inspect the reported differential uniformity; the recorded candidate has `Δ = 23`.

For statistical reproduction, use the appropriate binary and inspect its per-trial output. The binaries currently use `thread_rng()`, so rerunning an experiment generates new random permutations and will not reproduce the *same sequence of trials*.

## Citation

If you use this code in academic work, please cite the associated paper when available.
