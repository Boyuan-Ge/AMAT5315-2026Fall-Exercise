use ising::{
    analysis::{GroupAnalysis, RunAnalysis},
    artifacts::{Algorithm, RunMetadata},
    plot::{onsager_magnetization, plot_saved_run, plot_tau_comparison},
    protocol::{SweepConfig, run_sweep},
};

#[test]
fn onsager_curve_has_the_exact_phase_behavior() {
    assert!((onsager_magnetization(1.8) - 0.956857).abs() < 1e-5);
    assert_eq!(onsager_magnetization(2.26919), 0.0);
    assert_eq!(onsager_magnetization(3.0), 0.0);
}

#[test]
fn comparison_plot_draws_two_algorithms_at_the_shared_size() {
    let make_analysis = |algorithm, tau_values: [f64; 2]| RunAnalysis {
        metadata: RunMetadata {
            sizes: vec![64],
            t_grid: vec![2.0, 2.3],
            eq_sweeps: 2,
            meas_sweeps: 5,
            meas_sweeps_critical: 5,
            sample_every: 1,
            seed: 42,
            algorithm,
        },
        groups: [2.0, 2.3]
            .into_iter()
            .zip(tau_values)
            .map(|(t, tau_int)| GroupAnalysis {
                l: 64,
                t,
                count: 5,
                mean_abs_m: 0.5,
                susceptibility: 1.0,
                naive_error: 0.01,
                blocked_error: 0.02,
                error_ratio: 2.0,
                tau_int,
            })
            .collect(),
        peaks: Vec::new(),
        critical_temperature: None,
    };
    let metropolis = make_analysis(Algorithm::Metropolis, [2.0, 100.0]);
    let wolff = make_analysis(Algorithm::Wolff, [0.6, 0.8]);
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("tau-compare.png");
    plot_tau_comparison(&metropolis, &wolff, 64, &output).unwrap();
    assert!(std::fs::metadata(output).unwrap().len() > 1_000);
}

#[test]
fn plotting_a_saved_tiny_analysis_creates_three_nonempty_pngs() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = SweepConfig::tiny_test(dir.path().to_path_buf());
    config.t_end = 1.9;
    run_sweep(&config).unwrap();

    let paths = plot_saved_run(dir.path(), 5).unwrap();
    assert_eq!(paths.len(), 3);
    assert!(
        paths
            .iter()
            .all(|path| std::fs::metadata(path).unwrap().len() > 1_000)
    );
}
