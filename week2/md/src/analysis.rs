use crate::{Boundary, Frame, PairModel, System, minimum_image, read_trajectory};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct CheckReport {
    pub secular_drift: f64,
    pub speed_temperature: f64,
    pub chi2_per_dof: f64,
}

impl CheckReport {
    pub fn passed(&self) -> bool {
        self.secular_drift < 2.0e-3
            && (self.speed_temperature - 0.5).abs() < 0.05
            && self.chi2_per_dof < 2.0
    }
}

pub fn speed_bin(speed: f64, temperature: f64) -> usize {
    let cumulative = 1.0 - (-speed * speed / (2.0 * temperature)).exp();
    ((24.0 * cumulative).floor() as usize).min(23)
}

pub fn radial_distribution(
    frames: &[Frame],
    box_size: [f64; 2],
    rho: f64,
    bins: usize,
) -> Vec<(f64, f64)> {
    if frames.is_empty() || bins == 0 {
        return Vec::new();
    }
    let max_radius = 0.5 * box_size[0].min(box_size[1]);
    let width = max_radius / bins as f64;
    let particles = frames[0].pos.len();
    let mut counts = vec![0usize; bins];
    for frame in frames {
        for i in 0..frame.pos.len() {
            for j in (i + 1)..frame.pos.len() {
                let dx = minimum_image(frame.pos[i][0] - frame.pos[j][0], box_size[0]);
                let dy = minimum_image(frame.pos[i][1] - frame.pos[j][1], box_size[1]);
                let radius = dx.hypot(dy);
                if radius < max_radius {
                    counts[(radius / width) as usize] += 2;
                }
            }
        }
    }
    (0..bins)
        .map(|index| {
            let inner = index as f64 * width;
            let outer = inner + width;
            let measured = counts[index] as f64 / (particles * frames.len()) as f64;
            let expected = rho * std::f64::consts::PI * (outer * outer - inner * inner);
            ((inner + outer) / 2.0, measured / expected)
        })
        .collect()
}

pub fn check_trajectory(directory: &Path) -> Result<CheckReport, Box<dyn std::error::Error>> {
    let (metadata, frames) = read_trajectory(directory)?;
    if frames.is_empty() {
        return Err("trajectory contains no frames".into());
    }

    let mut energies = Vec::with_capacity(frames.len());
    let mut squared_speeds = Vec::with_capacity(frames.len() * metadata.n);
    for frame in &frames {
        if frame.pos.len() != metadata.n || frame.vel.len() != metadata.n {
            return Err(format!("frame {} has the wrong particle count", frame.step).into());
        }
        let system = System {
            pos: frame.pos.clone(),
            vel: frame.vel.clone(),
            force: vec![[0.0, 0.0]; metadata.n],
            mass: 1.0,
            boundary: Boundary::Periodic {
                box_size: metadata.box_size,
            },
            pair_model: PairModel::ShiftedCutoff { rc: 2.5 },
        };
        let potential = system.potential_energy();
        let kinetic = system.kinetic_energy();
        let potential_tolerance = 1.0e-9 * potential.abs().max(1.0);
        let kinetic_tolerance = 1.0e-9 * kinetic.abs().max(1.0);
        if (frame.e_pot - potential).abs() > potential_tolerance
            || (frame.e_kin - kinetic).abs() > kinetic_tolerance
        {
            return Err(format!("stored energy mismatch in frame {}", frame.step).into());
        }
        energies.push(potential + kinetic);
        squared_speeds.extend(
            frame
                .vel
                .iter()
                .map(|velocity| velocity[0] * velocity[0] + velocity[1] * velocity[1]),
        );
    }

    let k = (frames.len() / 10).max(1);
    let first_mean = energies[..k].iter().sum::<f64>() / k as f64;
    let last_mean = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let secular_drift = (last_mean - first_mean).abs() / energies[0].abs();

    let speed_temperature =
        squared_speeds.iter().sum::<f64>() / (2.0 * squared_speeds.len() as f64);
    let mut observed = [0usize; 24];
    for squared_speed in squared_speeds {
        observed[speed_bin(squared_speed.sqrt(), speed_temperature)] += 1;
    }
    let expected = observed.iter().sum::<usize>() as f64 / 24.0;
    let chi2_per_dof = observed
        .iter()
        .map(|count| {
            let difference = *count as f64 - expected;
            difference * difference / expected
        })
        .sum::<f64>()
        / 22.0;

    Ok(CheckReport {
        secular_drift,
        speed_temperature,
        chi2_per_dof,
    })
}
