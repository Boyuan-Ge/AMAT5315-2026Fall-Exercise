use ising::metropolis::AcceptanceTable;
use ising::protocol::{RelaxConfig, render_relax, run_relax};

fn small(seed: u64) -> RelaxConfig {
    RelaxConfig {
        l: 12,
        t: 2.3,
        sweeps: 20,
        measure: 30,
        seed,
    }
}

#[test]
fn acceptance_table_matches_the_five_hand_calculated_values() {
    let table = AcceptanceTable::new(2.3).unwrap();
    assert_eq!(table.probability(-8), 1.0);
    assert_eq!(table.probability(-4), 1.0);
    assert_eq!(table.probability(0), 1.0);
    assert!((table.probability(4) - 0.1757).abs() < 1e-4);
    assert!((table.probability(8) - 0.0309).abs() < 1e-4);
}

#[test]
fn one_seed_reproduces_the_complete_printed_run() {
    let a = render_relax(&run_relax(&small(2026)).unwrap());
    let b = render_relax(&run_relax(&small(2026)).unwrap());
    assert_eq!(a, b);
}

#[test]
fn a_different_seed_changes_the_printed_run() {
    let a = render_relax(&run_relax(&small(2026)).unwrap());
    let b = render_relax(&run_relax(&small(2027)).unwrap());
    assert_ne!(a, b);
}
