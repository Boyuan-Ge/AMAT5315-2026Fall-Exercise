use anyhow::Result;
use clap::Parser;
use ising::{
    analysis::{analyze_folder, render_analysis},
    cli::{Cli, Command},
    protocol::{
        RelaxConfig, SnapshotConfig, SweepConfig, render_relax, run_relax, run_snapshots, run_sweep,
    },
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
        Command::Sweep {
            sizes,
            t_start,
            t_end,
            t_step,
            critical_start,
            critical_end,
            critical_step,
            eq_sweeps,
            meas_sweeps,
            meas_sweeps_critical,
            sample_every,
            seed,
            output,
        } => run_sweep(&SweepConfig {
            sizes,
            t_start,
            t_end,
            t_step,
            critical_start,
            critical_end,
            critical_step,
            eq_sweeps,
            meas_sweeps,
            meas_sweeps_critical,
            sample_every,
            seed,
            output,
            algorithm: ising::artifacts::Algorithm::Metropolis,
        })?,
        Command::Analyze { folder, blocks } => {
            let analysis = analyze_folder(&folder, blocks)?;
            print!("{}", render_analysis(&analysis));
        }
    }
    Ok(())
}
