use std::collections::BTreeSet;

use ising::contract::{ContractConfig, ContractMetadata, Update, run_contract};

fn tiny(update: Update, out: std::path::PathBuf) -> ContractConfig {
    ContractConfig {
        update,
        l: 4,
        t_from: 1.5,
        t_to: 1.6,
        t_step: 0.1,
        discard: 2,
        measure: 4,
        seed: 2026,
        every: 2,
        out,
    }
}

#[test]
fn metropolis_contract_writes_exact_metadata_rows_and_integer_frames() {
    let dir = tempfile::tempdir().unwrap();
    run_contract(&tiny(Update::Metropolis, dir.path().to_path_buf())).unwrap();

    let metadata: ContractMetadata =
        serde_json::from_reader(std::fs::File::open(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(metadata.l, 4);
    assert_eq!(metadata.update, Update::Metropolis);
    assert_eq!(metadata.t_grid, vec![1.5, 1.6]);
    assert_eq!(metadata.time_unit, "sweep");
    assert_eq!(metadata.sample_every, 1);

    let rows = std::fs::read_to_string(dir.path().join("series.jsonl")).unwrap();
    assert_eq!(rows.lines().count(), 8);
    let first: serde_json::Value = serde_json::from_str(rows.lines().next().unwrap()).unwrap();
    let keys: BTreeSet<_> = first
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["E", "L", "M", "T", "sweep"].into_iter().collect());
    assert_eq!(first["sweep"], 1);

    let frames = std::fs::read_to_string(dir.path().join("spins.jsonl")).unwrap();
    assert_eq!(frames.lines().count(), 4);
    let first: serde_json::Value = serde_json::from_str(frames.lines().next().unwrap()).unwrap();
    assert_eq!(first["spins"].as_array().unwrap().len(), 16);
    assert!(first["spins"].as_array().unwrap().iter().all(|spin| {
        let value = spin.as_i64().unwrap();
        value == -1 || value == 1
    }));
    assert_eq!(first["sweep"], 4);
}

#[test]
fn wolff_contract_records_one_cluster_flip_per_row() {
    let dir = tempfile::tempdir().unwrap();
    run_contract(&tiny(Update::Wolff, dir.path().to_path_buf())).unwrap();

    let metadata: ContractMetadata =
        serde_json::from_reader(std::fs::File::open(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(metadata.time_unit, "cluster_flip");

    let rows = std::fs::read_to_string(dir.path().join("series.jsonl")).unwrap();
    assert_eq!(rows.lines().count(), 8);
    for line in rows.lines() {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(row["cluster_size"].as_u64().unwrap() >= 1);
        assert!(row["cluster_size"].as_u64().unwrap() <= 16);
    }
}

#[test]
fn one_seed_reproduces_every_contract_file() {
    let root = tempfile::tempdir().unwrap();
    let a = root.path().join("a");
    let b = root.path().join("b");
    run_contract(&tiny(Update::Metropolis, a.clone())).unwrap();
    run_contract(&tiny(Update::Metropolis, b.clone())).unwrap();
    for name in ["run.json", "series.jsonl", "spins.jsonl"] {
        assert_eq!(
            std::fs::read(a.join(name)).unwrap(),
            std::fs::read(b.join(name)).unwrap()
        );
    }
}
