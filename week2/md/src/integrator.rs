use crate::System;

pub trait Integrator {
    fn name(&self) -> &'static str;
    fn step(&self, system: &mut System, dt: f64) -> f64;
}

pub struct Euler;

impl Integrator for Euler {
    fn name(&self) -> &'static str {
        "forward-euler"
    }

    fn step(&self, system: &mut System, dt: f64) -> f64 {
        for ((pos, vel), force) in system
            .pos
            .iter_mut()
            .zip(system.vel.iter_mut())
            .zip(system.force.iter())
        {
            pos[0] += vel[0] * dt;
            pos[1] += vel[1] * dt;
            vel[0] += force[0] / system.mass * dt;
            vel[1] += force[1] / system.mass * dt;
        }
        system.wrap_positions();
        let potential = system.refresh_forces();
        system.kinetic_energy() + potential
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn name(&self) -> &'static str {
        "velocity-verlet"
    }

    fn step(&self, system: &mut System, dt: f64) -> f64 {
        let half_dt = 0.5 * dt;
        for ((pos, vel), force) in system
            .pos
            .iter_mut()
            .zip(system.vel.iter_mut())
            .zip(system.force.iter())
        {
            vel[0] += force[0] / system.mass * half_dt;
            vel[1] += force[1] / system.mass * half_dt;
            pos[0] += vel[0] * dt;
            pos[1] += vel[1] * dt;
        }

        system.wrap_positions();
        let potential = system.refresh_forces();
        for (vel, force) in system.vel.iter_mut().zip(system.force.iter()) {
            vel[0] += force[0] / system.mass * half_dt;
            vel[1] += force[1] / system.mass * half_dt;
        }
        system.kinetic_energy() + potential
    }
}

#[derive(Debug)]
pub struct EnergyTrace {
    pub relative_errors: Vec<f64>,
    pub max_relative_error: f64,
    pub final_relative_error: f64,
}

pub fn simulate_steps(
    mut system: System,
    integrator: &dyn Integrator,
    dt: f64,
    steps: usize,
) -> EnergyTrace {
    let initial_energy = system.total_energy();
    let mut relative_errors = Vec::with_capacity(steps);

    for _ in 0..steps {
        let energy = integrator.step(&mut system, dt);
        relative_errors.push((energy - initial_energy) / initial_energy.abs());
    }

    let max_relative_error = relative_errors
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let final_relative_error = relative_errors.last().copied().unwrap_or(0.0);

    EnergyTrace {
        relative_errors,
        max_relative_error,
        final_relative_error,
    }
}
