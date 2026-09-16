use anyhow::Result;
use clap::Parser;
use ising::{cli::Cli, contract::run_contract};

fn main() -> Result<()> {
    run_contract(&Cli::parse().into())
}
