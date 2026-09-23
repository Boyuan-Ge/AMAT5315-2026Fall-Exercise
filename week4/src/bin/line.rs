use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use num_complex::Complex64;
use rustfft::FftPlanner;
use serde::Serialize;
use std::f64::consts::PI;
use week4_fluid::{integrator, line_rate};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run {
        #[arg(long)]
        n: usize,
        #[arg(long)]
        c: f64,
        #[arg(long)]
        nu: f64,
        #[arg(long)]
        dt: f64,
        #[arg(long = "t-end")]
        t_end: f64,
        #[arg(long)]
        sigma: f64,
        #[arg(long)]
        method: String,
        #[arg(long, default_value = "fourier")]
        derivative: String,
        #[arg(long, default_value_t = false)]
        frames: bool,
    },
    Growth {
        #[arg(long)]
        re: f64,
        #[arg(long)]
        im: f64,
        #[arg(long, default_value = "rk4")]
        method: String,
    },
    GrowthMap {
        #[arg(long, default_value_t = 241)]
        count: usize,
    },
}

#[derive(Serialize)]
struct Output {
    n: usize,
    dt: f64,
    t_end: f64,
    x: Vec<f64>,
    u: Vec<f64>,
    exact: Vec<f64>,
    max_error: f64,
    times: Vec<f64>,
    profiles: Vec<Vec<f64>>,
}

fn pulse(x: f64, t: f64, c: f64, nu: f64, sigma: f64) -> f64 {
    let width = (sigma * sigma + 2.0 * nu * t).sqrt();
    (-3..=3)
        .map(|m| {
            let d = x - PI / 2.0 - c * t + 2.0 * PI * m as f64;
            sigma / width * (-d * d / (2.0 * width * width)).exp()
        })
        .sum()
}

fn inverse(modes: &[Complex64], fft: &std::sync::Arc<dyn rustfft::Fft<f64>>) -> Vec<f64> {
    let mut v = modes.to_vec();
    fft.process(&mut v);
    let scale = 1.0 / v.len() as f64;
    v.iter().map(|z| z.re * scale).collect()
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::GrowthMap { count } => {
            let stepper = integrator("rk4").unwrap();
            let mut values = Vec::with_capacity(count * count);
            for row in 0..count {
                for col in 0..count {
                    let z = Complex64::new(
                        -4.0 + 5.0 * col as f64 / (count - 1) as f64,
                        -3.5 + 7.0 * row as f64 / (count - 1) as f64,
                    );
                    values.push(
                        stepper.step(&[Complex64::new(1.0, 0.0)], 1.0, &|y| vec![z * y[0]])[0]
                            .norm(),
                    );
                }
            }
            serde_json::to_writer(std::io::stdout().lock(), &values)?;
            println!();
        }
        Command::Growth { re, im, method } => {
            let stepper = integrator(&method).ok_or_else(|| anyhow::anyhow!("unknown method"))?;
            let z = Complex64::new(re, im);
            let result = stepper.step(&[Complex64::new(1.0, 0.0)], 1.0, &|y| vec![z * y[0]]);
            println!("{}", result[0].norm());
        }
        Command::Run {
            n,
            c,
            nu,
            dt,
            t_end,
            sigma,
            method,
            derivative,
            frames,
        } => {
            if n < 8 || !n.is_power_of_two() || dt <= 0.0 || sigma <= 0.0 || nu < 0.0 || t_end < 0.0
            {
                bail!("invalid grid or parameters");
            }
            let stepper = integrator(&method).ok_or_else(|| anyhow::anyhow!("unknown method"))?;
            let centred = match derivative.as_str() {
                "fourier" => false,
                "centred" => true,
                _ => bail!("derivative must be fourier or centred"),
            };
            let x: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
            let mut y: Vec<Complex64> = x
                .iter()
                .map(|&xx| Complex64::new(pulse(xx, 0.0, c, nu, sigma), 0.0))
                .collect();
            let mut planner = FftPlanner::<f64>::new();
            planner.plan_fft_forward(n).process(&mut y);
            let inv = planner.plan_fft_inverse(n);
            let steps = (t_end / dt).round() as usize;
            let mut times = Vec::new();
            let mut profiles = Vec::new();
            if frames {
                times.push(0.0);
                profiles.push(inverse(&y, &inv));
            }
            for step in 1..=steps {
                y = stepper.step(&y, dt, &|z| line_rate(z, n, c, nu, centred));
                if frames {
                    times.push(step as f64 * dt);
                    profiles.push(inverse(&y, &inv));
                }
            }
            let final_t = steps as f64 * dt;
            let u = inverse(&y, &inv);
            let exact: Vec<f64> = x
                .iter()
                .map(|&xx| pulse(xx, final_t, c, nu, sigma))
                .collect();
            let max_error = u
                .iter()
                .zip(&exact)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            serde_json::to_writer(
                std::io::stdout().lock(),
                &Output {
                    n,
                    dt,
                    t_end: final_t,
                    x,
                    u,
                    exact,
                    max_error,
                    times,
                    profiles,
                },
            )?;
            println!();
        }
    }
    Ok(())
}
