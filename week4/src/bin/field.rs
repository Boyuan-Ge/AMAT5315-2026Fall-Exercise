use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use num_complex::Complex64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::f64::consts::PI;
use week4_fluid::{Fluid, index};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    TaylorGreen {
        #[arg(long)]
        n: usize,
        #[arg(long)]
        nu: Option<f64>,
        #[arg(long, default_value_t = 0.0)]
        t: f64,
    },
    Random {
        #[arg(long)]
        n: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long = "k-min")]
        k_min: i32,
        #[arg(long = "k-max")]
        k_max: i32,
    },
}

#[derive(Serialize)]
struct Field {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[i32; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn check_n(n: usize) -> Result<()> {
    if n < 8 || !n.is_power_of_two() {
        bail!("n must be a power of two and at least 8");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let field = match cli.command {
        Command::TaylorGreen { n, nu, t } => {
            check_n(n)?;
            if t < 0.0 {
                bail!("t must be nonnegative");
            }
            if t > 0.0 && nu.is_none() {
                bail!("--nu is required for t > 0");
            }
            let decay = (-2.0 * nu.unwrap_or(0.0) * t).exp();
            let mut u = Vec::with_capacity(n * n);
            let mut v = Vec::with_capacity(n * n);
            for y in 0..n {
                for x in 0..n {
                    let xx = 2.0 * PI * x as f64 / n as f64;
                    let yy = 2.0 * PI * y as f64 / n as f64;
                    u.push(xx.cos() * yy.sin() * decay);
                    v.push(-xx.sin() * yy.cos() * decay);
                }
            }
            Field {
                case: "taylor-green".into(),
                n,
                seed: None,
                k_band: None,
                u,
                v,
            }
        }
        Command::Random {
            n,
            seed,
            k_min,
            k_max,
        } => {
            check_n(n)?;
            if k_min < 1 || k_max < k_min || k_max >= (n / 3) as i32 {
                bail!("invalid band; require 1 <= k-min <= k-max < floor(n/3)");
            }
            let fluid = Fluid::new(n, 0.0);
            let mut omega = vec![Complex64::new(0.0, 0.0); n * n];
            let mut rng = StdRng::seed_from_u64(seed);
            for ky in -k_max..=k_max {
                for kx in -k_max..=k_max {
                    let k2 = kx * kx + ky * ky;
                    if k2 < k_min * k_min || k2 > k_max * k_max || ky < 0 || (ky == 0 && kx <= 0) {
                        continue;
                    }
                    let theta = rng.random::<f64>() * 2.0 * PI;
                    let value = Complex64::from_polar(1.0, theta);
                    omega[index(kx, ky, n)] = value;
                    omega[index(-kx, -ky, n)] = value.conj();
                }
            }
            let (e, _) = fluid.energy_enstrophy(&omega);
            let scale = (0.5 / e).sqrt();
            for value in &mut omega {
                *value *= scale;
            }
            let (u, v) = fluid.velocity(&omega);
            Field {
                case: "random".into(),
                n,
                seed: Some(seed),
                k_band: Some([k_min, k_max]),
                u,
                v,
            }
        }
    };
    serde_json::to_writer(std::io::stdout().lock(), &field).context("write field JSON")?;
    println!();
    Ok(())
}
