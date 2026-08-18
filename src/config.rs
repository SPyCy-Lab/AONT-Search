pub struct SearchConfig {
    pub t: usize,
    pub s: usize,
    pub first_nonlinear_to_keep: usize,
    pub fix_zero: bool,
    pub linear_output: String,
    pub nonlinear_output: String,
    pub progress_report_interval: u128,
    pub estimator_samples: usize,
    pub estimator_seed: u64,
    pub run_old_heuristic: bool,
}

impl SearchConfig {
    pub fn new(t: usize, s: usize) -> Self {
        Self {
            t,
            s,
            first_nonlinear_to_keep: 5,
            fix_zero: true,
            linear_output: "linear.txt".to_string(),
            nonlinear_output: "nonlinear.txt".to_string(),
            progress_report_interval: 10_000_000,
            estimator_samples: 5_000,
            estimator_seed: 12_345,
            run_old_heuristic: false,
        }
    }
}
