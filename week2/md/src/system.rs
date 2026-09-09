use crate::{lj_energy, lj_force};

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
        self.force.fill([0.0, 0.0]);
        let mut potential = 0.0;

        for i in 0..self.pos.len() {
            for j in (i + 1)..self.pos.len() {
                let dx = self.pos[i][0] - self.pos[j][0];
                let dy = self.pos[i][1] - self.pos[j][1];
                let r = dx.hypot(dy);
                let radial_force = lj_force(r);
                let fx = radial_force * dx / r;
                let fy = radial_force * dy / r;
                self.force[i][0] += fx;
                self.force[i][1] += fy;
                self.force[j][0] -= fx;
                self.force[j][1] -= fy;
                potential += lj_energy(r);
            }
        }

        potential
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.vel
            .iter()
            .map(|v| 0.5 * self.mass * (v[0] * v[0] + v[1] * v[1]))
            .sum()
    }

    pub fn potential_energy(&self) -> f64 {
        let mut potential = 0.0;
        for i in 0..self.pos.len() {
            for j in (i + 1)..self.pos.len() {
                let dx = self.pos[i][0] - self.pos[j][0];
                let dy = self.pos[i][1] - self.pos[j][1];
                potential += lj_energy(dx.hypot(dy));
            }
        }
        potential
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }
}
