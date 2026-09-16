use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::contract::{ContractConfig, Update};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum UpdateArg {
    Metropolis,
    Wolff,
}

#[derive(Debug, Parser)]
#[command(
    name = "ising",
    about = "Sample the two-dimensional Ising model along one temperature ramp"
)]
pub struct Cli {
    #[arg(long, value_enum)]
    pub update: UpdateArg,
    #[arg(long)]
    pub l: usize,
    #[arg(long = "t-from")]
    pub t_from: f64,
    #[arg(long = "t-to")]
    pub t_to: f64,
    #[arg(long = "t-step")]
    pub t_step: f64,
    #[arg(long)]
    pub discard: usize,
    #[arg(long)]
    pub measure: usize,
    #[arg(long)]
    pub seed: u64,
    #[arg(long, default_value_t = 0)]
    pub every: usize,
    #[arg(long)]
    pub out: PathBuf,
}

impl From<Cli> for ContractConfig {
    fn from(cli: Cli) -> Self {
        Self {
            update: match cli.update {
                UpdateArg::Metropolis => Update::Metropolis,
                UpdateArg::Wolff => Update::Wolff,
            },
            l: cli.l,
            t_from: cli.t_from,
            t_to: cli.t_to,
            t_step: cli.t_step,
            discard: cli.discard,
            measure: cli.measure,
            seed: cli.seed,
            every: cli.every,
            out: cli.out,
        }
    }
}
