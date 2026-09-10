use anyhow::{Result, bail};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
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
