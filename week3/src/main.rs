use anyhow::Result;
use clap::Parser;
use ising::{
    analysis::{analyze_folder, render_analysis},
    cli::{Cli, Command},
    plot::{plot_saved_run, plot_tau_comparison},
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
            wolff,
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
            seeds,
            output,
        } => {
            let mut config = if wolff {
                SweepConfig::wolff_default()
            } else {
                SweepConfig::metropolis_default()
            };
            if let Some(value) = sizes {
                config.sizes = value;
            }
            if let Some(value) = t_start {
                config.t_start = value;
            }
            if let Some(value) = t_end {
                config.t_end = value;
            }
            if let Some(value) = t_step {
                config.t_step = value;
            }
            if let Some(value) = critical_start {
                config.critical_start = value;
            }
            if let Some(value) = critical_end {
                config.critical_end = value;
            }
            if let Some(value) = critical_step {
                config.critical_step = value;
            }
            if let Some(value) = eq_sweeps {
                config.eq_sweeps = value;
            }
            if let Some(value) = meas_sweeps {
                config.meas_sweeps = value;
            }
            if let Some(value) = meas_sweeps_critical {
                config.meas_sweeps_critical = value;
            }
            if let Some(value) = sample_every {
                config.sample_every = value;
            }
            if let Some(value) = seed {
                config.seed = value;
            }
            if let Some(value) = seeds {
                config.seeds = Some(value);
            }
            if let Some(value) = output {
                config.output = value;
            }
            run_sweep(&config)?;
        }
        Command::Analyze { folder, blocks } => {
            let analysis = analyze_folder(&folder, blocks)?;
            print!("{}", render_analysis(&analysis));
        }
        Command::Plot { folder, blocks } => {
            for path in plot_saved_run(&folder, blocks)? {
                println!("wrote {}", path.display());
            }
        }
        Command::CompareTau {
            metropolis,
            wolff,
            l,
            output,
            blocks,
        } => {
            let metropolis = analyze_folder(&metropolis, blocks)?;
            let wolff = analyze_folder(&wolff, blocks)?;
            plot_tau_comparison(&metropolis, &wolff, l, &output)?;
            println!("wrote {}", output.display());
        }
    }
    Ok(())
}
