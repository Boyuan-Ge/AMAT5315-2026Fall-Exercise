use anyhow::Result;
use clap::Parser;
use ising::{
    cli::{Cli, Command},
    protocol::{RelaxConfig, SnapshotConfig, render_relax, run_relax, run_snapshots},
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
        Command::Snapshots {
            l,
            t_start,
            t_end,
            t_step,
            eq_sweeps,
            record_sweeps,
            frame_every,
            seed,
            output,
        } => run_snapshots(&SnapshotConfig {
            l,
            t_start,
            t_end,
            t_step,
            eq_sweeps,
            record_sweeps,
            frame_every,
            seed,
            output,
        })?,
    }
    Ok(())
}
