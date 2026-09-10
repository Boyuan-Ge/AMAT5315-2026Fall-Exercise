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
        #[arg(long)]
        wolff: bool,
        #[arg(long, value_delimiter = ',')]
        sizes: Option<Vec<usize>>,
        #[arg(long)]
        t_start: Option<f64>,
        #[arg(long)]
        t_end: Option<f64>,
        #[arg(long)]
        t_step: Option<f64>,
        #[arg(long)]
        critical_start: Option<f64>,
        #[arg(long)]
        critical_end: Option<f64>,
        #[arg(long)]
        critical_step: Option<f64>,
        #[arg(long)]
        eq_sweeps: Option<usize>,
        #[arg(long)]
        meas_sweeps: Option<usize>,
        #[arg(long)]
        meas_sweeps_critical: Option<usize>,
        #[arg(long)]
        sample_every: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<u64>>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Analyze a saved run with correlation-aware error bars.
    Analyze {
        folder: PathBuf,
        #[arg(long, default_value_t = 50)]
        blocks: usize,
    },
    /// Draw magnetization, susceptibility, and autocorrelation charts.
    Plot {
        folder: PathBuf,
        #[arg(long, default_value_t = 50)]
        blocks: usize,
    },
    /// Compare Metropolis and Wolff autocorrelation times on one chart.
    CompareTau {
        metropolis: PathBuf,
        wolff: PathBuf,
        #[arg(long, default_value_t = 64)]
        l: usize,
        #[arg(long, default_value = "tau-compare.png")]
        output: PathBuf,
        #[arg(long, default_value_t = 50)]
        blocks: usize,
    },
}
