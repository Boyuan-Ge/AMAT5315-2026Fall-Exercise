use anyhow::Result;
use clap::Parser;
use ising::{
    cli::{Cli, Command},
    protocol::{RelaxConfig, render_relax, run_relax},
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Relax {
            l,
            t,
            sweeps,
            measure,
            seed,
        } => {
            let result = run_relax(&RelaxConfig {
                l,
                t,
                sweeps,
                measure,
                seed,
            })?;
            print!("{}", render_relax(&result));
        }
    }
    Ok(())
}
