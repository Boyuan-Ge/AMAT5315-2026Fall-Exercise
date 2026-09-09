mod force;
mod initialise;
mod integrator;
mod potential;
mod system;

pub use force::{ForceMethod, evaluate_forces, minimum_image};
pub use initialise::{
    kinetic_temperature, rescale_temperature, seeded_velocities, triangular_lattice,
};
pub use integrator::{EnergyTrace, Euler, Integrator, VelocityVerlet, simulate_steps};
pub use potential::{lj_energy, lj_force, shifted_energy};
pub use system::{Boundary, PairModel, System};

/// Return the greeting printed by the `md` executable.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::{
        Boundary, Euler, ForceMethod, PairModel, System, VelocityVerlet, evaluate_forces, greeting,
        kinetic_temperature, lj_energy, lj_force, minimum_image, seeded_velocities, shifted_energy,
        simulate_steps, triangular_lattice,
    };

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn lennard_jones_well_has_unit_depth() {
        let r_min = 2.0_f64.powf(1.0 / 6.0);
        assert!((lj_energy(r_min) + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn lennard_jones_force_matches_energy_derivative() {
        let h = 1.0e-5;
        for r in [0.95, 1.05, 2.0_f64.powf(1.0 / 6.0), 1.3, 2.0] {
            let numerical_force = -(lj_energy(r + h) - lj_energy(r - h)) / (2.0 * h);
            let force = lj_force(r);
            let tolerance = 1.0e-6 * force.abs().max(1.0);
            assert!(
                (force - numerical_force).abs() < tolerance,
                "force mismatch at r={r}: analytic={force}, numerical={numerical_force}"
            );
        }
    }

    #[test]
    fn verlet_conserves_dimer_energy_while_euler_drifts() {
        let euler = simulate_steps(System::dimer(), &Euler, 0.01, 500);
        let verlet = simulate_steps(System::dimer(), &VelocityVerlet, 0.01, 500);

        assert!(verlet.max_relative_error < 1.0e-3);
        assert!(euler.final_relative_error.abs() > 0.5);
    }

    #[test]
    fn minimum_image_crosses_nearest_boundary() {
        assert!((minimum_image(9.8, 10.0) + 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn shifted_energy_is_continuous_just_inside_cutoff() {
        assert!(shifted_energy(2.5 - 1.0e-8, 2.5).abs() < 1.0e-8);
        assert_eq!(shifted_energy(2.5, 2.5), 0.0);
    }

    #[test]
    fn internal_forces_sum_to_zero() {
        let mut system = System {
            pos: vec![[0.2, 0.3], [1.4, 0.4], [4.9, 0.2], [2.4, 3.0]],
            vel: vec![[0.0, 0.0]; 4],
            force: vec![[0.0, 0.0]; 4],
            mass: 1.0,
            boundary: Boundary::Periodic {
                box_size: [5.0, 5.0],
            },
            pair_model: PairModel::ShiftedCutoff { rc: 2.5 },
        };
        evaluate_forces(&mut system, ForceMethod::Naive);
        let sum = system
            .force
            .iter()
            .fold([0.0, 0.0], |s, f| [s[0] + f[0], s[1] + f[1]]);
        assert!(sum[0].abs() < 1.0e-10 && sum[1].abs() < 1.0e-10);
    }

    #[test]
    fn contract_lattice_has_expected_box() {
        let system = triangular_lattice(100, 0.8).unwrap();
        assert!((system.box_size()[0] - 12.014_057_070_673_772).abs() < 1.0e-10);
        assert!((system.box_size()[1] - 10.404_478_625_719_541).abs() < 1.0e-10);
    }

    #[test]
    fn prepared_velocities_are_repeatable_centered_and_at_target_temperature() {
        let first = seeded_velocities(100, 0.5, 2026).unwrap();
        let second = seeded_velocities(100, 0.5, 2026).unwrap();
        assert_eq!(first, second);

        let mean = first.iter().fold([0.0, 0.0], |sum, velocity| {
            [sum[0] + velocity[0], sum[1] + velocity[1]]
        });
        assert!((mean[0] / 100.0).abs() < 1.0e-12);
        assert!((mean[1] / 100.0).abs() < 1.0e-12);
        assert!((kinetic_temperature(&first) - 0.5).abs() < 1.0e-12);
    }
}
