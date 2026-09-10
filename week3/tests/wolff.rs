use ising::{
    lattice::Lattice,
    wolff::{wolff_cluster_flip, wolff_sweep},
};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn zero_temperature_limit_flips_the_entire_all_up_lattice() {
    let mut lat = Lattice::all_up(4).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let touched = wolff_cluster_flip(&mut lat, 0.01, &mut rng).unwrap();
    assert_eq!(touched, 16);
    assert_eq!(lat.magnetization(), -1.0);
}

#[test]
fn one_wolff_sweep_touches_at_least_one_lattice_volume() {
    let mut lat = Lattice::all_up(8).unwrap();
    let mut rng = StdRng::seed_from_u64(2026);
    let stats = wolff_sweep(&mut lat, 2.3, &mut rng).unwrap();
    assert!(stats.touched >= 64);
    assert!(stats.clusters >= 1);
}

#[test]
fn wolff_update_repeats_for_one_seed() {
    let run = |seed| {
        let mut lat = Lattice::all_up(12).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..20 {
            wolff_sweep(&mut lat, 2.3, &mut rng).unwrap();
        }
        lat.spin_bits()
    };
    assert_eq!(run(2026), run(2026));
    assert_ne!(run(2026), run(2027));
}
