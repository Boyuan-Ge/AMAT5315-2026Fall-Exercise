use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "ising", about = "Two-dimensional Ising Monte Carlo simulator")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Relax and measure one lattice at one temperature.
    Relax {
        #[arg(long)]
        l: usize,
        #[arg(long)]
        t: f64,
        #[arg(long)]
        sweeps: usize,
        #[arg(long)]
        measure: usize,
        #[arg(long)]
        seed: u64,
    },
    /// Record a heating ramp as JSON Lines frames for the public viewer.
    Snapshots {
        #[arg(long, default_value_t = 64)]
        l: usize,
        #[arg(long, default_value_t = 1.5)]
        t_start: f64,
        #[arg(long, default_value_t = 3.5)]
        t_end: f64,
        #[arg(long, default_value_t = 0.05)]
        t_step: f64,
        #[arg(long, default_value_t = 2_000)]
        eq_sweeps: usize,
        #[arg(long, default_value_t = 200)]
        record_sweeps: usize,
        #[arg(long, default_value_t = 20)]
        frame_every: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value = "artifacts/spins.jsonl")]
        output: PathBuf,
    },
    /// Sweep lattice sizes and temperatures into the course artifact format.
    Sweep {
        #[arg(long, value_delimiter = ',', default_value = "32,64")]
        sizes: Vec<usize>,
        #[arg(long, default_value_t = 1.5)]
        t_start: f64,
        #[arg(long, default_value_t = 3.5)]
        t_end: f64,
        #[arg(long, default_value_t = 0.1)]
        t_step: f64,
        #[arg(long, default_value_t = 2.0)]
        critical_start: f64,
        #[arg(long, default_value_t = 2.6)]
        critical_end: f64,
        #[arg(long, default_value_t = 0.05)]
        critical_step: f64,
        #[arg(long, default_value_t = 2_000)]
        eq_sweeps: usize,
        #[arg(long, default_value_t = 5_000)]
        meas_sweeps: usize,
        #[arg(long, default_value_t = 100_000)]
        meas_sweeps_critical: usize,
        #[arg(long, default_value_t = 1)]
        sample_every: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "artifacts")]
        output: PathBuf,
    },
    /// Analyze a saved run with correlation-aware error bars.
    Analyze {
        folder: PathBuf,
        #[arg(long, default_value_t = 50)]
        blocks: usize,
    },
}
