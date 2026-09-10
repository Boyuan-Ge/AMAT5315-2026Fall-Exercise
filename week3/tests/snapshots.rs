use std::collections::BTreeSet;

use ising::protocol::{SnapshotConfig, run_snapshots};

#[test]
fn a_small_ramp_writes_the_viewer_contract_in_protocol_order() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("spins.jsonl");
    run_snapshots(&SnapshotConfig {
        l: 4,
        t_start: 1.5,
        t_end: 1.6,
        t_step: 0.05,
        eq_sweeps: 2,
        record_sweeps: 4,
        frame_every: 2,
        seed: 2026,
        output: output.clone(),
    })
    .unwrap();

    let contents = std::fs::read_to_string(output).unwrap();
    let lines: Vec<_> = contents
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0]["L"], 4);
    assert_eq!(lines[0]["T"], 1.5);
    assert_eq!(lines[0]["spins"].as_str().unwrap().len(), 16);

    let actual_keys: BTreeSet<_> = lines[0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let expected_keys: BTreeSet<_> = ["L", "T", "m", "spins", "sweep"].into_iter().collect();
    assert_eq!(actual_keys, expected_keys);

    assert!(
        lines
            .windows(2)
            .all(|window| window[0]["sweep"].as_u64() < window[1]["sweep"].as_u64())
    );
}
