use std::collections::BTreeSet;

use ising::protocol::{SweepConfig, contract_temperature_grid, run_sweep};

#[test]
fn metropolis_contract_grid_has_27_ordered_unique_temperatures() {
    let grid = contract_temperature_grid();
    assert_eq!(grid.len(), 27);
    assert_eq!(grid.first().copied(), Some(1.5));
    assert_eq!(grid.last().copied(), Some(3.5));
    assert_eq!(
        grid.iter()
            .filter(|&&temperature| (2.0..=2.6).contains(&temperature))
            .count(),
        13
    );
    assert!(grid.windows(2).all(|window| window[0] < window[1]));
}

#[test]
fn tiny_sweep_writes_exact_metadata_and_series_fields() {
    let dir = tempfile::tempdir().unwrap();
    run_sweep(&SweepConfig::tiny_test(dir.path().to_path_buf())).unwrap();

    let meta: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(meta["algorithm"], "metropolis");
    assert_eq!(meta["sample_every"], 1);
    assert_eq!(meta.as_object().unwrap().len(), 8);

    let series = std::fs::read_to_string(dir.path().join("series.jsonl")).unwrap();
    assert_eq!(series.lines().count(), 10);
    let row: serde_json::Value = serde_json::from_str(series.lines().next().unwrap()).unwrap();
    let actual_keys: BTreeSet<_> = row
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let expected_keys: BTreeSet<_> = ["E", "L", "M", "T", "sweep"].into_iter().collect();
    assert_eq!(actual_keys, expected_keys);
    assert_eq!(row["L"], 8);
    assert_eq!(row["T"], 1.5);
    assert_eq!(row["sweep"], 1);

    let first_line = series.lines().next().unwrap();
    assert!(first_line.contains("\"M\":"));
    assert!(first_line.contains("\"E\":"));
}
