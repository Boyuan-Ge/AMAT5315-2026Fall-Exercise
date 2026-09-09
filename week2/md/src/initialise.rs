use crate::{Boundary, PairModel, System};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

pub fn triangular_lattice(n: usize, rho: f64) -> Result<System, String> {
    if n == 0 || !rho.is_finite() || rho <= 0.0 {
        return Err("n and rho must be positive".to_string());
    }
    let side = (n as f64).sqrt() as usize;
    if side * side != n || side % 2 != 0 {
        return Err("n must be an even square (for example 100, 400, or 1600)".to_string());
    }

    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 3.0_f64.sqrt() * a / 2.0;
    let box_size = [side as f64 * a, side as f64 * h];
    let mut pos = Vec::with_capacity(n);
    for row in 0..side {
        for column in 0..side {
            pos.push([(column as f64 + 0.5 * (row % 2) as f64) * a, row as f64 * h]);
        }
    }

    let mut system = System {
        pos,
        vel: vec![[0.0, 0.0]; n],
        force: vec![[0.0, 0.0]; n],
        mass: 1.0,
        boundary: Boundary::Periodic { box_size },
        pair_model: PairModel::ShiftedCutoff { rc: 2.5 },
    };
    system.refresh_forces();
    Ok(system)
}

pub fn kinetic_temperature(velocities: &[[f64; 2]]) -> f64 {
    let degrees_of_freedom = 2 * velocities.len() - 2;
    let twice_kinetic: f64 = velocities
        .iter()
        .map(|velocity| velocity[0] * velocity[0] + velocity[1] * velocity[1])
        .sum();
    twice_kinetic / degrees_of_freedom as f64
}

pub fn rescale_temperature(velocities: &mut [[f64; 2]], target: f64) -> Result<(), String> {
    if velocities.len() < 2 || !target.is_finite() || target <= 0.0 {
        return Err(
            "temperature requires at least two particles and a positive target".to_string(),
        );
    }
    let current = kinetic_temperature(velocities);
    if !current.is_finite() || current <= 0.0 {
        return Err("cannot rescale zero or non-finite kinetic temperature".to_string());
    }
    let scale = (target / current).sqrt();
    for velocity in velocities {
        velocity[0] *= scale;
        velocity[1] *= scale;
    }
    Ok(())
}

pub fn seeded_velocities(n: usize, temperature: f64, seed: u64) -> Result<Vec<[f64; 2]>, String> {
    if n < 2 || !temperature.is_finite() || temperature <= 0.0 {
        return Err(
            "velocity preparation requires at least two particles and positive temperature"
                .to_string(),
        );
    }
    let normal = Normal::new(0.0, temperature.sqrt()).map_err(|error| error.to_string())?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut velocities: Vec<[f64; 2]> = (0..n)
        .map(|_| [normal.sample(&mut rng), normal.sample(&mut rng)])
        .collect();
    let mean = velocities.iter().fold([0.0, 0.0], |sum, velocity| {
        [sum[0] + velocity[0], sum[1] + velocity[1]]
    });
    for velocity in &mut velocities {
        velocity[0] -= mean[0] / n as f64;
        velocity[1] -= mean[1] / n as f64;
    }
    rescale_temperature(&mut velocities, temperature)?;
    Ok(velocities)
}

pub fn ramp_temperature(start: f64, end: f64, step: usize, steps: usize) -> f64 {
    if steps == 0 {
        return end;
    }
    start + (end - start) * step.min(steps) as f64 / steps as f64
}
