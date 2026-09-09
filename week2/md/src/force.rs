use crate::potential::{lj_energy, lj_force, shifted_energy, shifted_force};
use crate::{Boundary, PairModel, System};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceMethod {
    Naive,
}

pub fn minimum_image(displacement: f64, box_length: f64) -> f64 {
    displacement - box_length * (displacement / box_length).round()
}

fn pair_displacement(system: &System, i: usize, j: usize) -> [f64; 2] {
    let mut displacement = [
        system.pos[i][0] - system.pos[j][0],
        system.pos[i][1] - system.pos[j][1],
    ];
    if let Boundary::Periodic { box_size } = system.boundary {
        displacement[0] = minimum_image(displacement[0], box_size[0]);
        displacement[1] = minimum_image(displacement[1], box_size[1]);
    }
    displacement
}

fn pair_values(model: PairModel, r: f64) -> (f64, f64) {
    match model {
        PairModel::Plain => (lj_energy(r), lj_force(r)),
        PairModel::ShiftedCutoff { rc } => (shifted_energy(r, rc), shifted_force(r, rc)),
    }
}

pub fn evaluate_forces(system: &mut System, method: ForceMethod) -> f64 {
    match method {
        ForceMethod::Naive => evaluate_naive(system),
    }
}

fn evaluate_naive(system: &mut System) -> f64 {
    system.force.fill([0.0, 0.0]);
    let mut potential = 0.0;

    for i in 0..system.pos.len() {
        for j in (i + 1)..system.pos.len() {
            let displacement = pair_displacement(system, i, j);
            let r = displacement[0].hypot(displacement[1]);
            let (pair_energy, radial_force) = pair_values(system.pair_model, r);
            if radial_force == 0.0 && pair_energy == 0.0 {
                continue;
            }
            let force = [
                radial_force * displacement[0] / r,
                radial_force * displacement[1] / r,
            ];
            system.force[i][0] += force[0];
            system.force[i][1] += force[1];
            system.force[j][0] -= force[0];
            system.force[j][1] -= force[1];
            potential += pair_energy;
        }
    }
    potential
}

pub fn potential_energy(system: &System) -> f64 {
    let mut potential = 0.0;
    for i in 0..system.pos.len() {
        for j in (i + 1)..system.pos.len() {
            let displacement = pair_displacement(system, i, j);
            let r = displacement[0].hypot(displacement[1]);
            potential += pair_values(system.pair_model, r).0;
        }
    }
    potential
}
