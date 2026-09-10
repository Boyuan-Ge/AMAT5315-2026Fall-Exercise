use ising::lattice::Lattice;

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
        vec![
            1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1,
        ],
    )
    .unwrap();
    assert_eq!(checkerboard.total_energy(), 32);
}

#[test]
fn invalid_spin_data_is_rejected() {
    assert!(Lattice::from_spins(3, vec![1; 8]).is_err());
    assert!(Lattice::from_spins(2, vec![1, 1, 0, -1]).is_err());
}
