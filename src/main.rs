use aont_search::bnb::enumerate_aonts;
use aont_search::config::SearchConfig;
use aont_search::differential_uniformity::analyze_default_nonlinear_file;
use aont_search::equivalence::check_nonlinear_are_columnwise_perms_of_linear;
use aont_search::field::ArkPrimeField;


fn main() -> std::io::Result<()> {

    //
    // ============================================================
    // 1. CHOOSE THE FIELD
    // ============================================================
    //

    // Arkworks imports.
    use ark_ff::{Fp64, MontBackend, MontConfig};

    //
    // ============================================================
    // FIELD DEFINITION
    // ============================================================
    //
    // This example uses GF(3).
    //
    // To use another PRIME field GF(p), change:
    //     #[modulus = "..."]
    //     #[generator = "..."]
    //
    // See the example near the bottom for GF(4) and GF(5).
    //

    #[derive(MontConfig)]
    #[modulus = "3"]
    #[generator = "2"]
    pub struct F3Config;
    
    pub type F3 = Fp64<MontBackend<F3Config, 1>>;
    
    let field = ArkPrimeField::<F3>::new(F3::from(2u64));
    
    // Field size.
    //
    // IMPORTANT:
    // Keep this equal to the actual field used above.
    let q = 3;


    //
    // ============================================================
    // 2. CHOOSE THE AONT PARAMETERS
    // ============================================================
    //
    // Search for (t, s, q)-AONTs.
    //
    // Example below:
    //
    //      (2, 3, 3)-AONT
    //

    let t = 2;
    let s = 3;


    //
    // ============================================================
    // 3. CHOOSE WHICH FEATURES TO RUN
    // ============================================================
    //

    // Run the main branch-and-bound enumeration.
    let run_search = true;

    // After the search, compute differential uniformity
    // for every nonlinear AONT found.
    let run_differential_uniformity = true;

    // Check whether every nonlinear AONT can be obtained
    // from a linear AONT by independent symbol permutations
    // in each input/output column.
    let run_column_equivalence_check = true;


    //
    // ============================================================
    // 4. SEARCH CONFIGURATION
    // ============================================================
    //

    let mut config = SearchConfig::new(t, s);

    // Fix:
    //
    //      f(0) = 0
    //
    // This removes the output-translation symmetry and can
    // dramatically reduce the search size.
    config.fix_zero = false;

    // Keep this many nonlinear AONTs in memory so examples
    // can be printed after the search.
    //
    // ALL nonlinear mappings are still written to nonlinear.txt.
    config.first_nonlinear_to_keep = 5;

    // Print a progress report every N visited BnB nodes.
    //
    // Increase this for large/fast searches if terminal output
    // becomes too frequent.
    config.progress_report_interval = 1_000_000;

    // Number of randomized paths used by the search-tree estimator.
    //
    // Larger values generally give a more stable estimate,
    // but take longer before the actual BnB begins.
    config.estimator_samples = 5_000;


    //
    // ============================================================
    // 5. RUN THE SEARCH
    // ============================================================
    //

    if run_search {
        println!();
        println!("==========================================");
        println!("Starting AONT search");
        println!("==========================================");
        println!("t = {}", t);
        println!("s = {}", s);
        println!("q = {}", q);
        println!("fix_zero = {}", config.fix_zero);
        println!("==========================================");
        println!();

        let stats = enumerate_aonts(
            &field,
            &config,
        )?;

        println!();
        println!("==========================================");
        println!("Search finished");
        println!("==========================================");
        println!("Total AONTs:       {}", stats.total);
        println!("Linear AONTs:      {}", stats.linear_count);
        println!("Nonlinear AONTs:   {}", stats.nonlinear_count);
        println!("Branches visited:  {}", stats.branches_visited);
        println!("==========================================");


        //
        // ========================================================
        // 6. OPTIONAL: DIFFERENTIAL UNIFORMITY
        // ========================================================
        //

        if run_differential_uniformity {
            if stats.nonlinear_count == 0 {
                println!();
                println!(
                    "Skipping differential-uniformity analysis: \
                     no nonlinear AONTs were found."
                );
            } else {
                println!();
                println!("==========================================");
                println!("Differential-uniformity analysis");
                println!("==========================================");

                analyze_default_nonlinear_file(&field)?;
            }
        }


        //
        // ========================================================
        // 7. OPTIONAL: COLUMN-WISE EQUIVALENCE TEST
        // ========================================================
        //

        if run_column_equivalence_check {
            if stats.nonlinear_count == 0 {
                println!();
                println!(
                    "Skipping column-equivalence check: \
                     no nonlinear AONTs were found."
                );
            } else if stats.linear_count == 0 {
                println!();
                println!(
                    "Skipping column-equivalence check: \
                     no linear AONTs were found."
                );
            } else {
                println!();
                println!("==========================================");
                println!("Column-wise permutation equivalence");
                println!("==========================================");

                let report =
                    check_nonlinear_are_columnwise_perms_of_linear(
                        &config.linear_output,
                        &config.nonlinear_output,
                        q,
                        s,
                    )?;

                println!();
                println!(
                    "Covered {}/{} nonlinear mappings.",
                    report.nonlinear_covered,
                    report.nonlinear_mappings,
                );
            }
        }
    }


    Ok(())
}

/*
GF(4) Example
==============

use aont_search::field::Gf4;
let field = Gf4::default();
let q = 4;

GF(3) Example
==============

#[derive(MontConfig)]
#[modulus = "3"]
#[generator = "2"]
pub struct F3Config;

pub type F3 = Fp64<MontBackend<F3Config, 1>>;

let field =
    ArkPrimeField::<F3>::new(F3::from(2u64));

let q = 3;

GF(5) Example
==============

#[derive(MontConfig)]
#[modulus = "5"]
#[generator = "2"]
pub struct F5Config;

pub type F5 = Fp64<MontBackend<F5Config, 1>>;

let field = ArkPrimeField::<F5>::new(F5::from(2u64));

let q = 5;
*/
