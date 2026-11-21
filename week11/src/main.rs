mod gates;
mod network;
use anyhow::{ensure, Context, Result};
use clap::Parser;
use network::{Tensor, Weight};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, value_parser=["gadgets","plan","count","sample"])]
    mode: String,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    n: Option<usize>,
    #[arg(long)]
    l: Option<usize>,
    #[arg(long)]
    p_af: Option<f64>,
    #[arg(long)]
    beta: Option<f64>,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, default_value="treesa",value_parser=["greedy","treesa"])]
    order: String,
    #[arg(long)]
    samples: Option<usize>,
}

type Term = (Vec<usize>, i64);
fn magnet(a: &Args, rng: &mut StdRng) -> Result<(usize, Vec<Term>)> {
    if let Some(path) = &a.input {
        let v: Value = serde_json::from_slice(&fs::read(path)?)?;
        let target = &v["target"];
        ensure!(target["type"] == "SpinGlass", "not a SpinGlass bundle");
        let data = &target["data"];
        let n = data["graph"]["num_vertices"]
            .as_u64()
            .context("num_vertices")? as usize;
        let edges: Vec<[usize; 2]> = serde_json::from_value(data["graph"]["edges"].clone())?;
        let j: Vec<i64> = serde_json::from_value(data["couplings"].clone())?;
        let h: Vec<i64> = serde_json::from_value(data["fields"].clone())?;
        ensure!(edges.len() == j.len() && h.len() == n, "malformed magnet");
        let mut terms: Vec<_> = edges
            .into_iter()
            .zip(j)
            .map(|(e, w)| (e.to_vec(), w))
            .collect();
        let present: std::collections::BTreeSet<_> =
            terms.iter().flat_map(|t| t.0.iter().copied()).collect();
        terms.extend(
            h.into_iter()
                .enumerate()
                .filter(|(i, h)| *h != 0 || !present.contains(i))
                .map(|(i, h)| (vec![i], h)),
        );
        Ok((n, terms))
    } else {
        let l = a.l.context("--l required")?;
        let p = a.p_af.context("--p-af required")?;
        ensure!(
            (2..=64).contains(&l) && (0. ..=1.).contains(&p),
            "invalid lattice settings"
        );
        let mut terms = vec![];
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                for j in [
                    ((x + 1 < l).then_some(i + 1)),
                    ((y + 1 < l).then_some(i + l)),
                ]
                .into_iter()
                .flatten()
                {
                    terms.push((vec![i, j], if rng.gen::<f64>() < p { 1 } else { -1 }));
                }
            }
        }
        Ok((l * l, terms))
    }
}
fn run(a: &Args) -> Result<Value> {
    if a.mode == "gadgets" {
        let (gates, circuit) = gates::enumerate();
        let count = circuit.iter().filter(|r| r["energy"] == -5).count();
        return Ok(json!({"gates":gates,"circuit":circuit,"circuit_ground_states":count}));
    }
    ensure!(
        [a.n.is_some(), a.l.is_some(), a.input.is_some()]
            .into_iter()
            .filter(|x| *x)
            .count()
            == 1,
        "specify exactly one of --n, --l or --input"
    );
    let mut rng = StdRng::seed_from_u64(a.seed);
    let mut terms = vec![];
    let (mut factors, n) = if let Some(n) = a.n {
        ensure!((4..=10).contains(&n), "supported queen sizes 4–10");
        network::queens(n)
    } else {
        let beta = if a.mode == "plan" {
            0.
        } else {
            a.beta.context("--beta required")?
        };
        ensure!(beta.is_finite() && beta >= 0., "invalid beta");
        let (n, m) = magnet(a, &mut rng)?;
        terms = m;
        let factors = terms
            .iter()
            .map(|(labels, c)| {
                let entries = (0..1 << labels.len())
                    .map(|x| {
                        let e = (0..labels.len())
                            .map(|i| 2 * ((x >> i) & 1) as i64 - 1)
                            .product::<i64>()
                            * c;
                        (x, Weight::from_energy(e, beta))
                    })
                    .collect();
                Tensor {
                    labels: labels.clone(),
                    entries,
                }
            })
            .collect();
        (factors, n)
    };
    let (steps, tc, sc) = network::plan(&factors, &a.order)?;
    let mut result = json!({"variables":n,"tc":tc,"sc":sc,"planner":"omeco 0.2.6 built-in seeds","tree":steps.iter().map(|s|json!({"left":s.a,"right":s.b,"union":s.union,"retain":s.keep})).collect::<Vec<_>>()});
    if a.mode == "plan" {
        return Ok(result);
    }
    let sampling = a.mode == "sample";
    let trace = network::contract(&mut factors, &steps, sampling)?;
    let root = factors
        .last()
        .unwrap()
        .entries
        .get(&0)
        .context("network has no solutions")?;
    if a.n.is_some() {
        result["board_count"] = json!(root.count as u64);
    } else {
        result["log_z"] = json!(root.logz);
        result["e_min"] = json!(root.energy);
        result["g_0"] = json!(root.count.to_string());
        result["terms"] = json!(terms);
    }
    if sampling {
        let samples = a.samples.context("--samples required")?;
        ensure!(samples > 0, "empty sample request");
        let mut draws = Vec::with_capacity(samples);
        for _ in 0..samples {
            let x = network::draw(&factors, &steps, &trace, &mut rng, n);
            if let Some(n) = a.n {
                let board: Vec<_> = (0..n)
                    .map(|i| (0..n).find(|j| x[i * n + j] != 0).unwrap())
                    .collect();
                draws.push(json!(board));
            } else {
                let e = terms
                    .iter()
                    .map(|(ix, c)| ix.iter().map(|&i| 2 * x[i] as i64 - 1).product::<i64>() * c)
                    .sum::<i64>();
                draws.push(json!({"bits":x,"energy":e}));
            }
        }
        result["draws"] = json!(draws);
    }
    Ok(result)
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !a.out.exists(),
        "output directory exists; refusing to overwrite evidence"
    );
    let start = Instant::now();
    let mut result = run(&a)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.;
    result["schema_version"] = json!(1);
    result["mode"] = json!(a.mode);
    result["seed"] = json!(a.seed);
    result["parameters"] = json!({"n":a.n,"l":a.l,"p_af":a.p_af,"beta":a.beta,"input":a.input,"order":a.order,"samples":a.samples});
    result["runtime_ms"] = json!(elapsed);
    fs::create_dir_all(&a.out)?;
    fs::write(a.out.join("run.json"), serde_json::to_vec_pretty(&result)?)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for key in [
        "circuit_ground_states",
        "variables",
        "tc",
        "sc",
        "board_count",
        "log_z",
        "e_min",
        "g_0",
    ] {
        if let Some(value) = result.get(key) {
            println!("{}\t{key}\t{value}\t{elapsed}", a.mode);
        }
    }
    Ok(())
}
