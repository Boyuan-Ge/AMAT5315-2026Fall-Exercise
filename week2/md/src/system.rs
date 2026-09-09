use crate::force::{ForceMethod, evaluate_forces, potential_energy};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    Open,
    Periodic { box_size: [f64; 2] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairModel {
    Plain,
    ShiftedCutoff { rc: f64 },
}

#[derive(Clone, Debug)]
pub struct System {
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub force: Vec<[f64; 2]>,
    pub mass: f64,
    pub boundary: Boundary,
    pub pair_model: PairModel,
}

impl System {
    pub fn dimer() -> Self {
        let mut system = Self {
            pos: vec![[0.0, 0.0], [1.2, 0.0]],
            vel: vec![[0.0, 0.0], [0.0, 0.0]],
            force: vec![[0.0, 0.0], [0.0, 0.0]],
            mass: 1.0,
            boundary: Boundary::Open,
            pair_model: PairModel::Plain,
        };
        system.refresh_forces();
        system
    }

    pub fn refresh_forces(&mut self) -> f64 {
        evaluate_forces(self, ForceMethod::Naive)
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.vel
            .iter()
            .map(|v| 0.5 * self.mass * (v[0] * v[0] + v[1] * v[1]))
            .sum()
    }

    pub fn potential_energy(&self) -> f64 {
        potential_energy(self)
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    pub fn box_size(&self) -> [f64; 2] {
        match self.boundary {
            Boundary::Periodic { box_size } => box_size,
            Boundary::Open => panic!("an open system has no periodic box"),
        }
    }

    pub fn wrap_positions(&mut self) {
        if let Boundary::Periodic { box_size } = self.boundary {
            for position in &mut self.pos {
                position[0] = position[0].rem_euclid(box_size[0]);
                position[1] = position[1].rem_euclid(box_size[1]);
            }
        }
    }
}
