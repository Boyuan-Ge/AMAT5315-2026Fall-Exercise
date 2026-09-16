use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    lattice::Lattice,
    metropolis::{AcceptanceTable, metropolis_sweep},
    protocol::temperature_grid,
    wolff::WolffUpdater,
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Update {
    Metropolis,
    Wolff,
}

#[derive(Clone, Debug)]
pub struct ContractConfig {
    pub update: Update,
    pub l: usize,
    pub t_from: f64,
    pub t_to: f64,
    pub t_step: f64,
    pub discard: usize,
    pub measure: usize,
    pub seed: u64,
    pub every: usize,
    pub out: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContractMetadata {
    #[serde(rename = "L")]
    pub l: usize,
    pub update: Update,
    pub t_grid: Vec<f64>,
    pub discard: usize,
    pub measure: usize,
    pub seed: u64,
    pub sample_every: usize,
    pub time_unit: String,
}

#[derive(Debug, Serialize)]
struct ContractFrame<'a> {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "T")]
    t: f64,
    sweep: u64,
    m: f64,
    spins: &'a [i8],
}

pub fn run_contract(config: &ContractConfig) -> Result<()> {
    validate(config)?;
    let temperatures = temperature_grid(config.t_from, config.t_to, config.t_step)?;
    fs::create_dir_all(&config.out)
        .with_context(|| format!("failed to create {}", config.out.display()))?;

    write_metadata(config, &temperatures)?;
    let series_file = File::create(config.out.join("series.jsonl"))
        .with_context(|| format!("failed to create {}/series.jsonl", config.out.display()))?;
    let mut series = BufWriter::new(series_file);
    let mut frames = if config.every > 0 {
        Some(BufWriter::new(
            File::create(config.out.join("spins.jsonl")).with_context(|| {
                format!("failed to create {}/spins.jsonl", config.out.display())
            })?,
        ))
    } else {
        None
    };

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l)?;
    let mut wolff = WolffUpdater::new(lattice.len());
    let mut cumulative_step = 0_u64;

    match config.update {
        Update::Metropolis => println!("T\tmean_abs_M\tacceptance_rate"),
        Update::Wolff => println!("T\tmean_abs_M\tmean_cluster_size"),
    }

    for &temperature in &temperatures {
        let table = match config.update {
            Update::Metropolis => Some(AcceptanceTable::new(temperature)?),
            Update::Wolff => None,
        };
        let mut accepted = 0_usize;
        let mut cluster_size_sum = 0_usize;

        for _ in 0..config.discard {
            cumulative_step += 1;
            match config.update {
                Update::Metropolis => {
                    accepted += metropolis_sweep(
                        &mut lattice,
                        table.as_ref().expect("Metropolis table exists"),
                        &mut rng,
                    );
                }
                Update::Wolff => {
                    cluster_size_sum += wolff.cluster_flip(&mut lattice, temperature, &mut rng)?;
                }
            }
        }

        let mut abs_m_sum = 0.0;
        for step in 1..=config.measure {
            cumulative_step += 1;
            let cluster_size = match config.update {
                Update::Metropolis => {
                    accepted += metropolis_sweep(
                        &mut lattice,
                        table.as_ref().expect("Metropolis table exists"),
                        &mut rng,
                    );
                    None
                }
                Update::Wolff => {
                    let size = wolff.cluster_flip(&mut lattice, temperature, &mut rng)?;
                    cluster_size_sum += size;
                    Some(size)
                }
            };

            let magnetization = lattice.magnetization();
            abs_m_sum += magnetization.abs();
            write_series_row(
                &mut series,
                config.l,
                temperature,
                step,
                magnetization,
                lattice.energy_per_site(),
                cluster_size,
            )?;

            if config.every > 0 && step.is_multiple_of(config.every) {
                let frame = ContractFrame {
                    l: config.l,
                    t: temperature,
                    sweep: cumulative_step,
                    m: magnetization,
                    spins: lattice.spins(),
                };
                let frame_writer = frames.as_mut().expect("frame writer exists");
                serde_json::to_writer(&mut *frame_writer, &frame)?;
                frame_writer.write_all(b"\n")?;
            }
        }

        let mean_abs_m = abs_m_sum / config.measure as f64;
        match config.update {
            Update::Metropolis => {
                let proposals = (config.discard + config.measure)
                    .checked_mul(lattice.len())
                    .ok_or_else(|| anyhow::anyhow!("proposal count overflow"))?;
                println!(
                    "{temperature:.6}\t{mean_abs_m:.6}\t{:.6}",
                    accepted as f64 / proposals as f64
                );
            }
            Update::Wolff => println!(
                "{temperature:.6}\t{mean_abs_m:.6}\t{:.6}",
                cluster_size_sum as f64 / (config.discard + config.measure) as f64
            ),
        }
    }

    series.flush()?;
    if let Some(writer) = frames.as_mut() {
        writer.flush()?;
    }
    Ok(())
}

fn write_metadata(config: &ContractConfig, temperatures: &[f64]) -> Result<()> {
    let metadata = ContractMetadata {
        l: config.l,
        update: config.update,
        t_grid: temperatures.to_vec(),
        discard: config.discard,
        measure: config.measure,
        seed: config.seed,
        sample_every: 1,
        time_unit: match config.update {
            Update::Metropolis => "sweep",
            Update::Wolff => "cluster_flip",
        }
        .to_string(),
    };
    let path = config.out.join("run.json");
    serde_json::to_writer_pretty(
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?,
        &metadata,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn write_series_row(
    writer: &mut impl Write,
    l: usize,
    temperature: f64,
    sweep: usize,
    magnetization: f64,
    energy: f64,
    cluster_size: Option<usize>,
) -> Result<()> {
    write!(
        writer,
        "{{\"L\":{l},\"T\":{temperature},\"sweep\":{sweep},\"M\":{magnetization:.6},\"E\":{energy:.6}"
    )?;
    if let Some(size) = cluster_size {
        write!(writer, ",\"cluster_size\":{size}")?;
    }
    writeln!(writer, "}}")?;
    Ok(())
}

fn validate(config: &ContractConfig) -> Result<()> {
    if config.l < 2 {
        bail!("--l must be at least 2");
    }
    if config.measure == 0 {
        bail!("--measure must be greater than zero");
    }
    if config.every > config.measure {
        bail!("--every must not exceed --measure");
    }
    temperature_grid(config.t_from, config.t_to, config.t_step)?;
    Ok(())
}
