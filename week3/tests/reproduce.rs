use std::process::Command;

#[test]
fn make_reproduce_runs_the_real_pipeline_with_tunable_small_inputs() {
    let output_dir = tempfile::tempdir().unwrap();
    let artifact_path = output_dir.path().join("artifacts");
    let status = Command::new("make")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("reproduce")
        .arg(format!("ARTIFACTS={}", artifact_path.display()))
        .arg("BLOCKS=5")
        .arg("SNAPSHOT_ARGS=--l 4 --t-start 1.5 --t-end 1.6 --t-step 0.05 --eq-sweeps 2 --record-sweeps 4 --frame-every 2 --seed 2026")
        .arg("SWEEP_ARGS=--sizes 8 --t-start 1.5 --t-end 1.6 --t-step 0.1 --critical-start 9 --critical-end 10 --eq-sweeps 3 --meas-sweeps 5 --meas-sweeps-critical 5 --seed 7")
        .status()
        .unwrap();
    assert!(status.success());

    let snapshots = std::fs::read_to_string(artifact_path.join("spins.jsonl")).unwrap();
    let series = std::fs::read_to_string(artifact_path.join("series.jsonl")).unwrap();
    assert_eq!(snapshots.lines().count(), 6);
    assert_eq!(series.lines().count(), 10);
    for name in ["magnetization.png", "susceptibility.png", "tau.png"] {
        assert!(std::fs::metadata(artifact_path.join(name)).unwrap().len() > 1_000);
    }
}
