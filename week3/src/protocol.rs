use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    artifacts::Frame,
    lattice::Lattice,
    metropolis::{AcceptanceTable, metropolis_sweep},
};

#[derive(Clone, Debug)]
pub struct RelaxConfig {
    pub l: usize,
    pub t: f64,
    pub sweeps: usize,
    pub measure: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct RelaxResult {
    pub config: RelaxConfig,
    pub mean_abs_m: f64,
    pub acceptance: f64,
    pub final_lattice: Lattice,
}

#[derive(Clone, Debug)]
pub struct SnapshotConfig {
    pub l: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub t_step: f64,
    pub eq_sweeps: usize,
    pub record_sweeps: usize,
    pub frame_every: usize,
    pub seed: u64,
    pub output: PathBuf,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            l: 64,
            t_start: 1.5,
            t_end: 3.5,
            t_step: 0.05,
            eq_sweeps: 2_000,
            record_sweeps: 200,
            frame_every: 20,
            seed: 2026,
            output: PathBuf::from("artifacts/spins.jsonl"),
        }
    }
}

pub fn run_relax(config: &RelaxConfig) -> Result<RelaxResult> {
    validate_relax(config)?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l)?;
    let table = AcceptanceTable::new(config.t)?;
    let mut accepted = 0_usize;

    for _ in 0..config.sweeps {
        accepted += metropolis_sweep(&mut lattice, &table, &mut rng);
    }

    let mut abs_m_sum = 0.0;
    for _ in 0..config.measure {
        accepted += metropolis_sweep(&mut lattice, &table, &mut rng);
        abs_m_sum += lattice.magnetization().abs();
    }

    let sweep_count = config.sweeps + config.measure;
    let proposal_count = sweep_count
        .checked_mul(lattice.len())
        .ok_or_else(|| anyhow::anyhow!("proposal count overflow"))?;

    Ok(RelaxResult {
        config: config.clone(),
        mean_abs_m: abs_m_sum / config.measure as f64,
        acceptance: accepted as f64 / proposal_count as f64,
        final_lattice: lattice,
    })
}

pub fn render_relax(result: &RelaxResult) -> String {
    format!(
        "L={} T={:.3} sweeps={} measure={} mean_abs_m={:.4} accept={:.4}\n{}\n",
        result.config.l,
        result.config.t,
        result.config.sweeps,
        result.config.measure,
        result.mean_abs_m,
        result.acceptance,
        result.final_lattice.character_picture()
    )
}

pub fn temperature_grid(start: f64, end: f64, step: f64) -> Result<Vec<f64>> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        bail!("temperature grid values must be finite");
    }
    if start <= 0.0 || end <= 0.0 {
        bail!("temperature grid values must be positive");
    }
    if end < start {
        bail!("temperature end must not be below its start");
    }
    if step <= 0.0 {
        bail!("temperature step must be greater than zero");
    }

    let interval_count = (end - start) / step;
    let rounded_intervals = interval_count.round();
    if (interval_count - rounded_intervals).abs() > 1e-9 {
        bail!("temperature range must contain a whole number of steps");
    }
    let intervals = rounded_intervals as usize;
    Ok((0..=intervals)
        .map(|index| round_temperature(start + index as f64 * step))
        .collect())
}

pub fn run_snapshots(config: &SnapshotConfig) -> Result<()> {
    if config.l < 2 {
        bail!("--l must be at least 2");
    }
    if config.frame_every == 0 {
        bail!("--frame-every must be greater than zero");
    }
    if config.record_sweeps == 0 {
        bail!("--record-sweeps must be greater than zero");
    }
    if !config.record_sweeps.is_multiple_of(config.frame_every) {
        bail!("--record-sweeps must be divisible by --frame-every");
    }

    let temperatures = temperature_grid(config.t_start, config.t_end, config.t_step)?;
    if let Some(parent) = config.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file = File::create(&config.output)
        .with_context(|| format!("failed to create {}", config.output.display()))?;
    let mut writer = BufWriter::new(file);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l)?;
    let mut cumulative_sweep = 0_u64;

    for &temperature in &temperatures {
        let table = AcceptanceTable::new(temperature)?;
        for _ in 0..config.eq_sweeps {
            metropolis_sweep(&mut lattice, &table, &mut rng);
            cumulative_sweep += 1;
        }
        for recorded in 1..=config.record_sweeps {
            metropolis_sweep(&mut lattice, &table, &mut rng);
            cumulative_sweep += 1;
            if recorded.is_multiple_of(config.frame_every) {
                let frame = Frame {
                    l: config.l,
                    t: temperature,
                    sweep: cumulative_sweep,
                    m: lattice.magnetization(),
                    spins: lattice.spin_bits(),
                };
                serde_json::to_writer(&mut writer, &frame).with_context(|| {
                    format!("failed to write a frame to {}", config.output.display())
                })?;
                writer.write_all(b"\n")?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn validate_relax(config: &RelaxConfig) -> Result<()> {
    if config.l < 2 {
        bail!("--l must be at least 2");
    }
    if !config.t.is_finite() || config.t <= 0.0 {
        bail!("--t must be a positive finite number");
    }
    if config.measure == 0 {
        bail!("--measure must be greater than zero");
    }
    Ok(())
}

fn round_temperature(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}
