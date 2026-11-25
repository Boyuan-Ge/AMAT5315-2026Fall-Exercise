mod bnb;
mod learned;
use anyhow::{Result, ensure};
use clap::Parser;
use highs::{HighsModelStatus, RowProblem, Sense};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long,value_parser=["gap","bnb","eval"])]
    mode: String,
    #[arg(long,default_value="most-frac",value_parser=["most-frac","strong"])]
    policy: String,
    #[arg(long)]
    export_labels: bool,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    spins: usize,
    #[arg(long)]
    edges: usize,
    #[arg(long)]
    count: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
struct Glass {
    bonds: Vec<(usize, usize, i32)>,
    fields: Vec<i32>,
}
struct Relaxation {
    bound: f64,
    x: Vec<f64>,
    z: Vec<f64>,
    dual: Vec<f64>,
}
impl Glass {
    fn random(n: usize, m: usize, seed: u64, id: usize) -> Result<Self> {
        ensure!(
            n >= 3 && m >= 2 * n && m <= n * (n - 1) / 2,
            "edge count must allow scaffold and fit the simple graph"
        );
        let mut rng =
            SplitMix(SplitMix(seed ^ (id as u64).wrapping_mul(0xD1B54A32D192ED03)).next());
        let mut edges = BTreeSet::new();
        let mut degree = vec![0; n];
        let add = |u: usize, v: usize, d: &mut [usize], e: &mut BTreeSet<(usize, usize)>| {
            if u != v && e.insert((u.min(v), u.max(v))) {
                d[u] += 1;
                d[v] += 1;
            }
        };
        for i in 0..n {
            while degree[i] < 2 {
                add(i, rng.index(n), &mut degree, &mut edges);
            }
        }
        while edges.len() < m {
            add(rng.index(n), rng.index(n), &mut degree, &mut edges);
        }
        Ok(Self {
            bonds: edges
                .into_iter()
                .map(|(u, v)| (u, v, if rng.next() & 1 == 0 { 1 } else { -1 }))
                .collect(),
            fields: (0..n).map(|_| rng.index(3) as i32 - 1).collect(),
        })
    }
    fn energy(&self, x: &[i32]) -> i32 {
        self.bonds
            .iter()
            .map(|&(u, v, j)| j * (1 - 2 * x[u]) * (1 - 2 * x[v]))
            .sum::<i32>()
            + self
                .fields
                .iter()
                .zip(x)
                .map(|(h, x)| h * (1 - 2 * x))
                .sum::<i32>()
    }
    fn solve(&self, integer: bool, fix: &[Option<i32>]) -> Result<(f64, Vec<f64>)> {
        let r = self.relaxation(integer, fix, false)?;
        Ok((r.bound, r.x))
    }
    fn relaxation(&self, integer: bool, fix: &[Option<i32>], features: bool) -> Result<Relaxation> {
        let mut p = RowProblem::default();
        let x: Vec<_> = self
            .fields
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                let (lo, hi) = fix[i].map_or((0., 1.), |k| (k as f64, k as f64));
                if integer {
                    p.add_integer_column(-2. * h as f64, lo..=hi)
                } else {
                    p.add_column(-2. * h as f64, lo..=hi)
                }
            })
            .collect();
        let z: Vec<_> = self
            .bonds
            .iter()
            .map(|&(_, _, j)| p.add_column(-2. * j as f64, 0.0..=1.0))
            .collect();
        for (&(u, v, _), &z) in self.bonds.iter().zip(&z) {
            p.add_row(..=0., [(z, 1.), (x[u], -1.), (x[v], -1.)]);
            p.add_row(..=2., [(z, 1.), (x[u], 1.), (x[v], 1.)]);
            p.add_row(..=0., [(z, -1.), (x[u], 1.), (x[v], -1.)]);
            p.add_row(..=0., [(z, -1.), (x[u], -1.), (x[v], 1.)]);
        }
        let mut model = p.optimise(Sense::Minimise);
        model.set_option("presolve", "off");
        model.set_option("threads", 1);
        model.set_option("output_flag", false);
        let solved = model.solve();
        ensure!(
            solved.status() == HighsModelStatus::Optimal,
            "HiGHS status {:?}",
            solved.status()
        );
        let sol = solved.get_solution();
        let val = sol.columns();
        let f = self
            .fields
            .iter()
            .zip(val)
            .map(|(&h, &v)| h as f64 * (1. - 2. * v))
            .sum::<f64>()
            + self
                .bonds
                .iter()
                .zip(&val[x.len()..])
                .map(|(&(_, _, j), &v)| j as f64 * (1. - 2. * v))
                .sum::<f64>();
        Ok(Relaxation {
            bound: f,
            x: val[..x.len()].to_vec(),
            z: if features {
                val[x.len()..].to_vec()
            } else {
                vec![]
            },
            dual: if features {
                sol.dual_rows().to_vec()
            } else {
                vec![]
            },
        })
    }
    fn gap(&self, id: usize) -> Result<Value> {
        let fix = vec![None; self.fields.len()];
        let (lp, relaxed) = self.solve(false, &fix)?;
        let (mip, integer) = self.solve(true, &fix)?;
        let bits = round(&integer);
        let rounded = round(&relaxed);
        let optimum = self.energy(&bits);
        let e = self.energy(&rounded);
        ensure!(
            (optimum as f64 - mip).abs() < 1e-5 && lp <= mip + 1e-6 && e >= optimum,
            "invalid bounds or energy"
        );
        Ok(
            json!({"id":id,"n":self.fields.len(),"bonds":self.bonds,"fields":self.fields,
          "lp_bound":lp,"integer_optimum":optimum,"gap":mip-lp,"lp_spins":relaxed,"integer_spins":bits,
          "fractional_count":relaxed.iter().filter(|&&a|a>1e-6&&a<1.-1e-6).count(),
          "rounded_spins":rounded,"rounded_energy":e,"rounding_excess":e-optimum}),
        )
    }
}
fn round(x: &[f64]) -> Vec<i32> {
    x.iter().map(|&v| i32::from(v >= 0.5)).collect()
}
fn square() -> Glass {
    Glass {
        bonds: vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, -1)],
        fields: vec![0; 4],
    }
}
fn diet() -> Result<Value> {
    let mut p = RowProblem::default();
    let x = p.add_column(6., 0.0..);
    let y = p.add_column(3.5, 0.0..);
    p.add_row(8.0.., [(y, 7.)]);
    p.add_row(15.0.., [(x, 4.), (y, 2.)]);
    p.add_row(3.0.., [(x, 2.), (y, 1.)]);
    let mut m = p.optimise(Sense::Minimise);
    m.set_option("threads", 1);
    m.set_option("output_flag", false);
    let ans = m.solve();
    ensure!(
        ans.status() == HighsModelStatus::Optimal,
        "diet not optimal"
    );
    let s = ans.get_solution();
    let [x, y] = s.columns() else { unreachable!() };
    Ok(json!({"x1":x,"x2":y,"cost":6.*x+3.5*y}))
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !a.out.exists() && a.count > 0,
        "output must be new and count positive"
    );
    let start = Instant::now();
    if a.mode == "bnb" {
        return bnb::benchmark(&a, start);
    }
    if a.mode == "eval" {
        return bnb::evaluate(&a, start);
    }
    ensure!(
        !a.export_labels && a.model.is_none(),
        "gap does not use labels or a model"
    );
    let rows = (0..a.count)
        .map(|id| Glass::random(a.spins, a.edges, a.seed, id)?.gap(id))
        .collect::<Result<Vec<_>>>()?;
    let square = square().gap(0)?;
    let diet = diet()?;
    let average =
        |k: &str| rows.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / rows.len() as f64;
    let report = json!({"schema":1,"mode":a.mode,"seed":a.seed,"parameters":{"spins":a.spins,"edges":a.edges,"count":a.count,"threads":1,"presolve":false},"square":square,"diet":diet,"summary":{"mean_gap":average("gap"),"mean_rounding_excess":average("rounding_excess")},"instances":rows});
    fs::create_dir_all(&a.out)?;
    fs::write(a.out.join("run.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    println!(
        "gap\tmean_gap\t{}\t{}",
        report["summary"]["mean_gap"],
        start.elapsed().as_secs_f64() * 1000.
    );
    Ok(())
}
#[test]
fn square_and_diet() {
    let r = square().gap(0).unwrap();
    assert_eq!(r["integer_optimum"], -2);
    assert!((r["lp_bound"].as_f64().unwrap() + 4.).abs() < 1e-9);
    assert!((diet().unwrap()["cost"].as_f64().unwrap() - 23.07142857142857).abs() < 1e-8);
}
