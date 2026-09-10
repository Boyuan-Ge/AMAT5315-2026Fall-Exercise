use ising::lattice::Lattice;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn periodic_corner_has_four_wrapped_neighbours() {
    let lat = Lattice::from_spins(3, vec![1, -1, 1, -1, 1, 1, 1, 1, -1]).unwrap();
    assert_eq!(lat.neighbour_sum(0), 0);
}

#[test]
fn total_energy_counts_each_periodic_bond_once() {
    let all_up = Lattice::all_up(4).unwrap();
    assert_eq!(all_up.total_energy(), -32);

    let checkerboard = Lattice::from_spins(
        4,
        vec![1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    )
    .unwrap();
    assert_eq!(checkerboard.total_energy(), 32);
}

#[test]
fn invalid_spin_data_is_rejected() {
    assert!(Lattice::from_spins(3, vec![1; 8]).is_err());
    assert!(Lattice::from_spins(2, vec![1, 1, 0, -1]).is_err());
}

#[test]
fn local_delta_matches_recomputed_total_energy_on_random_lattices() {
    let mut rng = StdRng::seed_from_u64(5315);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..40 {
        let lat = Lattice::random(9, &mut rng).unwrap();
        for site in 0..lat.len() {
            let before = lat.total_energy();
            let reported = lat.delta_energy(site);
            let mut flipped = lat.clone();
            flipped.flip(site);
            assert_eq!(i64::from(reported), flipped.total_energy() - before);
            seen.insert(reported);
        }
    }
    assert_eq!(seen, [-8, -4, 0, 4, 8].into_iter().collect());
}
