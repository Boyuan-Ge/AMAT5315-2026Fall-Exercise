mod analysis;
mod cli;
mod force;
mod initialise;
mod integrator;
mod potential;
mod render;
mod system;
mod trajectory;

pub use analysis::{CheckReport, check_trajectory, radial_distribution, speed_bin};
pub use cli::{RunOptions, run_cli, run_to_dir};
pub use force::{ForceMethod, cell_candidate_pairs, evaluate_forces, minimum_image};
pub use initialise::{
    kinetic_temperature, ramp_temperature, rescale_temperature, seeded_velocities,
    triangular_lattice,
};
pub use integrator::{EnergyTrace, Euler, Integrator, VelocityVerlet, simulate_steps};
pub use potential::{lj_energy, lj_force, shifted_energy};
pub use render::render_video;
pub use system::{Boundary, PairModel, System};
pub use trajectory::{Frame, RunMetadata, TrajectoryWriter, read_trajectory};

/// Return the greeting printed by the `md` executable.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::{
        Boundary, Euler, ForceMethod, Frame, PairModel, RunMetadata, System, TrajectoryWriter,
        VelocityVerlet, cell_candidate_pairs, check_trajectory, evaluate_forces, greeting,
        kinetic_temperature, lj_energy, lj_force, minimum_image, radial_distribution,
        ramp_temperature, seeded_velocities, shifted_energy, simulate_steps, speed_bin,
        triangular_lattice,
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

    #[test]
    fn maxwell_boltzmann_bins_use_equal_probability_edges() {
        assert_eq!(speed_bin(0.0, 0.5), 0);
        assert_eq!(speed_bin(0.6, 0.5), 7);
        assert_eq!(speed_bin(2.0, 0.5), 23);
    }

    #[test]
    fn checker_rejects_a_corrupted_stored_energy() {
        let temporary = tempfile::tempdir().unwrap();
        let metadata = RunMetadata {
            n: 2,
            rho: 0.08,
            box_size: [5.0, 5.0],
            dt: 0.01,
            temperature: 0.5,
            eq_steps: 0,
            steps: 1,
            sample_every: 1,
            seed: 2026,
            integrator: "velocity-verlet".to_string(),
            force: "naive".to_string(),
            ramp_to: None,
        };
        let mut writer = TrajectoryWriter::create(temporary.path(), &metadata).unwrap();
        writer
            .write_frame(&Frame {
                step: 1,
                t: 0.01,
                pos: vec![[0.0, 0.0], [1.2, 0.0]],
                vel: vec![[-0.5, 0.0], [0.5, 0.0]],
                e_pot: 999.0,
                e_kin: 0.25,
            })
            .unwrap();
        writer.finish().unwrap();

        let error = check_trajectory(temporary.path()).unwrap_err();
        assert!(error.to_string().contains("stored energy"));
    }

    #[test]
    fn radial_distribution_counts_a_known_neighbor_shell() {
        let frames = vec![Frame {
            step: 1,
            t: 0.01,
            pos: vec![[1.0, 1.0], [2.1, 1.0]],
            vel: vec![[0.0, 0.0]; 2],
            e_pot: 0.0,
            e_kin: 0.0,
        }];
        let distribution = radial_distribution(&frames, [10.0, 10.0], 0.02, 10);
        assert_eq!(distribution.len(), 10);
        assert!((distribution[2].0 - 1.25).abs() < 1.0e-12);
        assert!((distribution[2].1 - 12.732_395_447_351_626).abs() < 1.0e-10);
    }

    fn assert_force_methods_match(system: System) {
        let mut naive = system.clone();
        let mut cells = system;
        let naive_energy = evaluate_forces(&mut naive, ForceMethod::Naive);
        let cells_energy = evaluate_forces(&mut cells, ForceMethod::Cells);
        let energy_tolerance = 1.0e-10 * naive_energy.abs().max(1.0);
        assert!((naive_energy - cells_energy).abs() < energy_tolerance);
        for (naive_force, cells_force) in naive.force.iter().zip(cells.force.iter()) {
            for axis in 0..2 {
                let tolerance = 1.0e-10 * naive_force[axis].abs().max(1.0);
                assert!((naive_force[axis] - cells_force[axis]).abs() < tolerance);
            }
        }
    }

    #[test]
    fn cell_list_matches_naive_on_a_perturbed_lattice() {
        let mut system = triangular_lattice(100, 0.8).unwrap();
        for (index, position) in system.pos.iter_mut().take(12).enumerate() {
            position[0] += 0.013 * (index as f64 + 1.0);
            position[1] -= 0.007 * (index as f64 + 1.0);
        }
        system.wrap_positions();
        assert_force_methods_match(system);
    }

    #[test]
    fn cell_list_matches_naive_across_a_periodic_boundary_and_at_cutoff() {
        for positions in [vec![[0.1, 1.0], [9.9, 1.0]], vec![[1.0, 1.0], [3.5, 1.0]]] {
            assert_force_methods_match(System {
                force: vec![[0.0, 0.0]; positions.len()],
                vel: vec![[0.0, 0.0]; positions.len()],
                pos: positions,
                mass: 1.0,
                boundary: Boundary::Periodic {
                    box_size: [10.0, 10.0],
                },
                pair_model: PairModel::ShiftedCutoff { rc: 2.5 },
            });
        }
    }

    #[test]
    fn two_cell_box_generates_each_particle_pair_once() {
        let system = System {
            pos: vec![[0.1, 0.1], [2.0, 0.3], [3.0, 3.1], [5.0, 5.0]],
            vel: vec![[0.0, 0.0]; 4],
            force: vec![[0.0, 0.0]; 4],
            mass: 1.0,
            boundary: Boundary::Periodic {
                box_size: [5.2, 5.2],
            },
            pair_model: PairModel::ShiftedCutoff { rc: 2.5 },
        };
        let pairs = cell_candidate_pairs(&system).unwrap();
        let unique: std::collections::HashSet<_> = pairs.iter().copied().collect();
        assert_eq!(pairs.len(), unique.len());
        assert_eq!(pairs.len(), 6);
        assert_force_methods_match(system);
    }

    #[test]
    fn heating_schedule_is_linear_and_reaches_both_endpoints() {
        assert!((ramp_temperature(0.2, 1.2, 0, 100) - 0.2).abs() < 1.0e-12);
        assert!((ramp_temperature(0.2, 1.2, 50, 100) - 0.7).abs() < 1.0e-12);
        assert!((ramp_temperature(0.2, 1.2, 100, 100) - 1.2).abs() < 1.0e-12);
    }
}
