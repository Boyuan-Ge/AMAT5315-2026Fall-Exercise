use ising::{
    plot::{onsager_magnetization, plot_saved_run},
    protocol::{SweepConfig, run_sweep},
};

#[test]
fn onsager_curve_has_the_exact_phase_behavior() {
    assert!((onsager_magnetization(1.8) - 0.956857).abs() < 1e-5);
    assert_eq!(onsager_magnetization(2.26919), 0.0);
    assert_eq!(onsager_magnetization(3.0), 0.0);
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
