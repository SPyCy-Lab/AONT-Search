use aont_search::bnb::enumerate_aonts;
use aont_search::config::SearchConfig;
use aont_search::differential_uniformity::analyze_default_nonlinear_file;
use aont_search::equivalence::check_nonlinear_are_columnwise_perms_of_linear;
use aont_search::field::{cauchy_matrix, ArkPrimeField,};
use aont_search::sampled_random_search::{
    parallel_sampled_random_search, SampledRandomSearchConfig,
};
use aont_search::types::{Symbol};
use aont_search::serial_flat_verifier::serial_flat_verify;
// Apple Metal only: comment out this import and its block on other platforms.
#[cfg(target_os = "macos")]
use aont_search::metal_flat_verifier::metal_flat_verify;

fn main() -> std::io::Result<()> {
    use ark_ff::{Fp64, MontBackend, MontConfig};

    // ============================================================
    // BRANCH AND BOUND SEARCH TO ENUMERATE AONTS
    // ============================================================

    // ============================================================
    // 1. CHOOSE THE FIELD
    // ============================================================
    // Example: GF(3). For another prime field, change the modulus,
    // generator, type name and q below. See examples at the end.
    #[derive(MontConfig)]
    #[modulus = "3"]
    #[generator = "2"]
    pub struct F3Config;
    pub type F3 = Fp64<MontBackend<F3Config, 1>>;
    let field = ArkPrimeField::<F3>::new(F3::from(2u64));
    let q = 3usize;

    // ============================================================
    // 2. CHOOSE THE AONT PARAMETERS
    // ============================================================
    // BnB enumerates (t,s,q)-AONTs. 
    let t = 2usize;
    let s = 3usize;

    // ============================================================
    // 3. CHOOSE WHICH EXPERIMENTS TO RUN
    // ============================================================
    // These are independent switches. Turn on only what you need.
    let run_bnb = true;

    // computes the DU of all AONTs found.
    let run_bnb_du_analysis = false; // Requires run_bnb=true.

    // Check whether every nonlinear AONT can be obtained
    // from a linear AONT by independent symbol permutations
    // in each input/output column.
    let run_bnb_equivalence = false; // Requires run_bnb=true.
    

    // ============================================================
    // 4. BRANCH-AND-BOUND SETTINGS
    // ============================================================
    let mut config = SearchConfig::new(t, s);
    config.fix_zero = false; // Enumerates normalized representatives.
    config.first_nonlinear_to_keep = 1;
    config.progress_report_interval = 1_000_000;
    config.estimator_samples = 5_000;

    
    // ============================================================
    // 5. RUN BRANCH-AND-BOUND ENUMERATION
    // ============================================================
    if run_bnb {
        println!("\n=== BnB enumeration: (t,s,q)=({t},{s},{q}) ===");
        let stats = enumerate_aonts(&field, &config)?;
        println!("Total: {}", stats.total);
        println!("Linear: {}", stats.linear_count);
        println!("Nonlinear: {}", stats.nonlinear_count);
        println!("Branches visited: {}", stats.branches_visited);

        if run_bnb_du_analysis && stats.nonlinear_count > 0 {
            println!("\n=== Differential uniformity of enumerated nonlinear AONTs ===");
            analyze_default_nonlinear_file(&field)?;
        }
        if run_bnb_equivalence && stats.nonlinear_count > 0 && stats.linear_count > 0 {
            println!("\n=== Column-wise permutation equivalence ===");
            let report = check_nonlinear_are_columnwise_perms_of_linear(
                &config.linear_output,
                &config.nonlinear_output,
                q,
                s,
            )?;
            println!("Covered {}/{} nonlinear mappings.",
                report.nonlinear_covered, report.nonlinear_mappings);
        }
    }

    // ============================================================
    // RANDOMIZED SEARCH AND FULL DIFFERENTIAL UNIFORMITY VERIFICATION
    // ============================================================

    // ============================================================
    // 1. CHOOSE THE FIELD
    // ============================================================
    // Example: GF(11). For another prime field, change the modulus,
    // generator, type name and q below. See examples at the end.
    #[derive(MontConfig)]
    #[modulus = "11"]
    #[generator = "2"]
    pub struct F11Config;
    pub type F11 = Fp64<MontBackend<F11Config, 1>>;
    let field = ArkPrimeField::<F11>::new(F11::from(2u64));
    let q = 11usize;
    let s = 4usize;

    // ============================================================
    // 2. CHOOSE WHICH EXPERIMENTS TO RUN
    // ============================================================
    // These are independent switches. Turn on only what you need.
    let run_random_search = false;
    let run_exact_verification = true;

    // ============================================================
    // 3. RANDOMIZED SEARCH SETTINGS
    // ============================================================
    // The sampled search screens candidates using only some DDT rows.
    // A sampled result is a lower bound, NOT exact differential uniformity.
    let random_config = SampledRandomSearchConfig {
        max_trials: 100_000,
        sampled_rows: 10_000,
        seed: 123_456,
        output_path: format!("sampled_s{s}_q{q}.txt"),
        progress_interval: 1_000,
    };

    // ============================================================
    // 4. EXACT VERIFICATION SETTINGS
    // ============================================================
    // Replace these identity permutations with the saved input/output
    // permutations of a candidate to verify its exact Delta.
    // The permutations bleow give a Delta of 23 for s=4 and q = 11.
    // you can find more verified permutation in the file "permutations_with_low_delta.txt"
    let input_perms = vec![
        vec![8, 1, 6, 4, 3, 9, 2, 0, 7, 10, 5],
        vec![7, 10, 3, 6, 5, 9, 8, 4, 2, 0, 1],
        vec![1, 0, 10, 7, 6, 3, 4, 8, 5, 9, 2],
        vec![4, 0, 2, 10, 3, 9, 6, 7, 1, 8, 5],
    ];

    let output_perms = vec![
        vec![9, 0, 10, 2, 3, 1, 6, 7, 8, 4, 5],
        vec![4, 7, 6, 10, 5, 0, 3, 9, 1, 2, 8],
        vec![9, 6, 3, 2, 5, 8, 4, 10, 1, 7, 0],
        vec![4, 6, 8, 1, 2, 5, 3, 7, 0, 9, 10],
    ];
    let cpu_progress_interval = 1_000;
    let metal_batch_size = 8;
    let use_metal = false; // Set true on macOS for GPU verification.


    // ============================================================
    // 5. CONSTRUCT A LINEAR AONT FOR SEARCH / VERIFICATION
    // ============================================================
    // A Cauchy matrix requires 2s distinct field elements (q >= 2s).
    if run_random_search || run_exact_verification {
        assert!(q >= 2 * s, "Cauchy construction requires q >= 2s");
        let a: Vec<Symbol> = (0..s).collect();
        let b: Vec<Symbol> = (s..2 * s).collect();
        let matrix = cauchy_matrix(&a, &b, &field);

        // ========================================================
        // 6. RUN RANDOMIZED SEARCH
        // ========================================================
        if run_random_search {
            println!("\n=== Randomized search: q={q}, s={s} ===");
            let stats = parallel_sampled_random_search(&matrix, &field, &random_config);
            println!("{stats:#?}");
        }

        // ========================================================
        // 7. RUN FULL EXACT DIFFERENTIAL-UNIFORMITY VERIFICATION
        // ========================================================
        if run_exact_verification {
            println!("\n=== Exact differential uniformity: q={q}, s={s} ===");
            if use_metal {
                #[cfg(target_os = "macos")]
                {
                    let result = metal_flat_verify(
                        &matrix, &input_perms, &output_perms,
                        &field, metal_batch_size,
                    );
                    println!("{result:#?}");
                }
                #[cfg(not(target_os = "macos"))]
                eprintln!("Metal is only available on macOS; set use_metal=false.");
            } else {
                let result = serial_flat_verify(
                    &matrix, &input_perms, &output_perms,
                    &field, cpu_progress_interval,
                );
                println!("{result:#?}");
            }
        }
    }

    Ok(())
}

/*
OTHER FIELD EXAMPLES (replace section 1):

GF(4):
    use aont_search::field::Gf4;
    let field = Gf4::default();
    let q = 4usize;

GF(5):
    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "2"]
    pub struct F5Config;
    pub type F5 = Fp64<MontBackend<F5Config, 1>>;
    let field = ArkPrimeField::<F5>::new(F5::from(2u64));
    let q = 5usize;

NOTE: The Cauchy example requires q >= 2s. For smaller q, use BnB
or supply a different known linear AONT matrix.
*/
