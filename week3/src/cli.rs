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
}
