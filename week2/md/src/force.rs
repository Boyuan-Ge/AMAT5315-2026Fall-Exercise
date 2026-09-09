use crate::potential::{lj_energy, lj_force, shifted_energy, shifted_force};
use crate::{Boundary, PairModel, System};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceMethod {
    Naive,
    Cells,
}

impl ForceMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Cells => "cells",
        }
    }
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
        ForceMethod::Cells => evaluate_cells(system),
    }
}

fn accumulate_pair(system: &mut System, i: usize, j: usize) -> f64 {
    let displacement = pair_displacement(system, i, j);
    let r = displacement[0].hypot(displacement[1]);
    let (pair_energy, radial_force) = pair_values(system.pair_model, r);
    if radial_force == 0.0 && pair_energy == 0.0 {
        return 0.0;
    }
    let force = [
        radial_force * displacement[0] / r,
        radial_force * displacement[1] / r,
    ];
    system.force[i][0] += force[0];
    system.force[i][1] += force[1];
    system.force[j][0] -= force[0];
    system.force[j][1] -= force[1];
    pair_energy
}

fn evaluate_naive(system: &mut System) -> f64 {
    system.force.fill([0.0, 0.0]);
    let mut potential = 0.0;

    for i in 0..system.pos.len() {
        for j in (i + 1)..system.pos.len() {
            potential += accumulate_pair(system, i, j);
        }
    }
    potential
}

pub fn cell_candidate_pairs(system: &System) -> Result<Vec<(usize, usize)>, String> {
    let box_size = match system.boundary {
        Boundary::Periodic { box_size } => box_size,
        Boundary::Open => return Err("cell lists require periodic boundaries".to_string()),
    };
    let rc = match system.pair_model {
        PairModel::ShiftedCutoff { rc } => rc,
        PairModel::Plain => return Err("cell lists require a finite cutoff".to_string()),
    };
    let nx = ((box_size[0] / rc).floor() as usize).max(1);
    let ny = ((box_size[1] / rc).floor() as usize).max(1);
    let cell_width = box_size[0] / nx as f64;
    let cell_height = box_size[1] / ny as f64;
    let mut cells = vec![Vec::new(); nx * ny];
    for (particle, position) in system.pos.iter().enumerate() {
        let x = ((position[0] / cell_width).floor() as usize).min(nx - 1);
        let y = ((position[1] / cell_height).floor() as usize).min(ny - 1);
        cells[y * nx + x].push(particle);
    }

    let mut pairs = Vec::new();
    for cell_index in 0..cells.len() {
        let x = cell_index % nx;
        let y = cell_index / nx;
        let mut neighbours = Vec::with_capacity(9);
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                let neighbour_x = (x as isize + dx).rem_euclid(nx as isize) as usize;
                let neighbour_y = (y as isize + dy).rem_euclid(ny as isize) as usize;
                neighbours.push(neighbour_y * nx + neighbour_x);
            }
        }
        neighbours.sort_unstable();
        neighbours.dedup();
        for neighbour in neighbours {
            if neighbour < cell_index {
                continue;
            }
            if neighbour == cell_index {
                for first in 0..cells[cell_index].len() {
                    for second in (first + 1)..cells[cell_index].len() {
                        pairs.push((cells[cell_index][first], cells[cell_index][second]));
                    }
                }
            } else {
                for &i in &cells[cell_index] {
                    for &j in &cells[neighbour] {
                        pairs.push((i.min(j), i.max(j)));
                    }
                }
            }
        }
    }
    Ok(pairs)
}

fn evaluate_cells(system: &mut System) -> f64 {
    let pairs = cell_candidate_pairs(system).expect("validated periodic cell-list system");
    system.force.fill([0.0, 0.0]);
    pairs
        .into_iter()
        .map(|(i, j)| accumulate_pair(system, i, j))
        .sum()
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
