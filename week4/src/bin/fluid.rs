use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use week4_fluid::{Fluid, integrator};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    method: String,
    #[arg(long)]
    nu: f64,
    #[arg(long)]
    dt: f64,
    #[arg(long = "t-end")]
    t_end: f64,
    #[arg(long)]
    every: f64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize)]
struct Input {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[i32; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Serialize)]
struct Run<'a> {
    case: &'a str,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[i32; 2]>,
    method: &'a str,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

#[derive(Serialize)]
struct Frame {
    t: f64,
    step: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    omega: Vec<f64>,
}

fn rounded(values: Vec<f64>) -> Vec<f64> {
    values
        .into_iter()
        .map(|v| (v * 1e6).round() / 1e6)
        .collect()
}

fn save_frame(
    file: &mut BufWriter<File>,
    fluid: &Fluid,
    w: &[num_complex::Complex64],
    step: usize,
    dt: f64,
) -> Result<()> {
    let (u, v) = fluid.velocity(w);
    let omega = fluid.fft.inverse_real(w);
    serde_json::to_writer(
        &mut *file,
        &Frame {
            t: (step as f64 * dt * 1e9).round() / 1e9,
            step,
            u: rounded(u),
            v: rounded(v),
            omega: rounded(omega),
        },
    )?;
    writeln!(file)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.nu < 0.0 || cli.dt <= 0.0 || cli.t_end < 0.0 || cli.every <= 0.0 {
        bail!("nu >= 0, dt > 0, t-end >= 0, every > 0 are required");
    }
    let method = integrator(&cli.method).context("method must be euler, rk2, or rk4")?;
    let input: Input =
        serde_json::from_reader(io::stdin().lock()).context("read field JSON from stdin")?;
    if input.n < 8 || !input.n.is_power_of_two() {
        bail!("n must be a power of two and at least 8");
    }
    if input.u.len() != input.n * input.n || input.v.len() != input.n * input.n {
        bail!("u/v length must be n*n");
    }
    fs::create_dir_all(&cli.out)?;
    let run = Run {
        case: &input.case,
        n: input.n,
        seed: input.seed,
        k_band: input.k_band,
        method: &cli.method,
        nu: cli.nu,
        dt: cli.dt,
        t_end: cli.t_end,
        snapshot_every: cli.every,
    };
    serde_json::to_writer_pretty(File::create(cli.out.join("run.json"))?, &run)?;
    let mut frames = BufWriter::new(File::create(cli.out.join("fields.jsonl"))?);
    let fluid = Fluid::new(input.n, cli.nu);
    let mut w = fluid.vorticity_from_velocity(&input.u, &input.v);
    let stride = ((cli.every / cli.dt).round() as usize).max(1);
    let steps = (cli.t_end / cli.dt - 1e-10).ceil().max(0.0) as usize;
    println!("t\tE\tZ");
    for step in 0..=steps {
        let t = step as f64 * cli.dt;
        let (e, z) = fluid.energy_enstrophy(&w);
        if !e.is_finite() || !z.is_finite() {
            println!("{t:.6}\tnon-finite\tnon-finite");
            io::stdout().flush()?;
            std::process::exit(1);
        }
        if step % stride == 0 || step == steps {
            println!("{t:.6}\t{e:.6}\t{z:.6}");
            save_frame(&mut frames, &fluid, &w, step, cli.dt)?;
        }
        if step < steps {
            w = method.step(&w, cli.dt, &|state| fluid.rate(state));
            fluid.mask(&mut w);
        }
    }
    Ok(())
}
