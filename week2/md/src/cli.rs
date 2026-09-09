use crate::{
    ForceMethod, Frame, Integrator, RunMetadata, TrajectoryWriter, VelocityVerlet,
    rescale_temperature, seeded_velocities, triangular_lattice,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "md", version, about = "Two-dimensional Lennard-Jones dynamics")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Run(RunArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ForceChoice {
    Naive,
}

#[derive(Args)]
struct RunArgs {
    #[arg(long, default_value_t = 100)]
    n: usize,
    #[arg(long, default_value_t = 0.8)]
    rho: f64,
    #[arg(long, default_value_t = 0.5)]
    temperature: f64,
    #[arg(long, default_value_t = 0.01)]
    dt: f64,
    #[arg(long, default_value_t = 2_000)]
    eq_steps: usize,
    #[arg(long, default_value_t = 10_000)]
    steps: usize,
    #[arg(long, default_value_t = 50)]
    sample_every: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = ForceChoice::Naive)]
    force: ForceChoice,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub force: ForceMethod,
    pub ramp_to: Option<f64>,
}

impl RunOptions {
    fn validate(&self) -> Result<(), String> {
        if self.sample_every == 0 {
            return Err("--sample-every must be greater than zero".to_string());
        }
        if self.steps == 0 {
            return Err("--steps must be greater than zero".to_string());
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err("--dt must be positive".to_string());
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err("--temperature must be positive".to_string());
        }
        Ok(())
    }
}

pub fn run_to_dir(
    options: &RunOptions,
    directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    options.validate()?;
    let mut system = triangular_lattice(options.n, options.rho)?;
    system.vel = seeded_velocities(options.n, options.temperature, options.seed)?;
    let integrator = VelocityVerlet;
    system.refresh_forces();

    for step in 1..=options.eq_steps {
        integrator.step(&mut system, options.dt);
        if step % 50 == 0 {
            rescale_temperature(&mut system.vel, options.temperature)?;
        }
    }

    let metadata = RunMetadata {
        n: options.n,
        rho: options.rho,
        box_size: system.box_size(),
        dt: options.dt,
        temperature: options.temperature,
        eq_steps: options.eq_steps,
        steps: options.steps,
        sample_every: options.sample_every,
        seed: options.seed,
        integrator: integrator.name().to_string(),
        force: "naive".to_string(),
        ramp_to: options.ramp_to,
    };
    let mut writer = TrajectoryWriter::create(directory, &metadata)?;

    for step in 1..=options.steps {
        integrator.step(&mut system, options.dt);
        if step % options.sample_every == 0 {
            writer.write_frame(&Frame {
                step,
                t: step as f64 * options.dt,
                pos: system.pos.clone(),
                vel: system.vel.clone(),
                e_pot: system.potential_energy(),
                e_kin: system.kinetic_energy(),
            })?;
        }
    }
    writer.finish()?;
    Ok(())
}

pub fn run_cli() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        None => println!("{}", crate::greeting()),
        Some(Command::Run(args)) => {
            let options = RunOptions {
                n: args.n,
                rho: args.rho,
                temperature: args.temperature,
                dt: args.dt,
                eq_steps: args.eq_steps,
                steps: args.steps,
                sample_every: args.sample_every,
                seed: args.seed,
                force: ForceMethod::Naive,
                ramp_to: None,
            };
            run_to_dir(&options, &args.out)?;
            println!(
                "wrote {} saved frames to {}",
                args.steps / args.sample_every,
                args.out.display()
            );
        }
    }
    Ok(())
}
