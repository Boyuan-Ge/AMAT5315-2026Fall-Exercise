mod bp;
mod exact;
mod model;
mod plan;

use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use model::{Id, Model, Observations};
use plan::{Plan, plan, recount, reopen};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, ValueEnum)]
enum Order {
    Random,
    Greedy,
    Treesa,
}
#[derive(Clone, Debug, ValueEnum)]
enum Evidence {
    Prior,
    #[value(alias = "AD")]
    Ad,
    #[value(alias = "ADX")]
    Adx,
    Promedus,
}
#[derive(Clone, Debug, ValueEnum)]
enum Method {
    Exact,
    Bp,
}

#[derive(Parser)]
#[command(about = "CPU float64 inference and contraction-order analysis for UAI MARKOV networks")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum)]
    evidence: Evidence,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    observations: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    queries: Vec<Id>,
    #[arg(long,value_enum,conflicts_with_all=["plan","analyze"])]
    order: Option<Order>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, conflicts_with = "analyze")]
    plan: Option<PathBuf>,
    #[arg(long)]
    analyze: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "exact")]
    method: Method,
    #[arg(long)]
    damping: Option<f64>,
    #[arg(long)]
    tolerance: Option<f64>,
    #[arg(long)]
    max_sweeps: Option<usize>,
}
fn write(path: &Path, name: &str, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path.join(name), bytes)?;
    Ok(())
}
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0_f64, f64::max)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let m = Model::read(&args.model)?;
    let benchmark = matches!(args.evidence, Evidence::Promedus);
    ensure!(
        benchmark == args.observations.is_some(),
        "--observations is required for promedus only"
    );
    ensure!(
        benchmark || args.queries.is_empty(),
        "ASIA always evaluates all eight node marginals and T,L joint"
    );
    let obs: Observations = match args.evidence {
        Evidence::Prior => BTreeMap::new(),
        Evidence::Ad => [(0, 0), (7, 0)].into(),
        Evidence::Adx => [(0, 0), (7, 0), (6, 0)].into(),
        Evidence::Promedus => Model::evidence_file(args.observations.as_ref().unwrap())?,
    };
    m.check_evidence(&obs)?;
    ensure!(
        benchmark || (m.dims == [2; 8] && m.factors.len() == 8),
        "ASIA mode requires the eight binary tables"
    );
    let bp_mode = matches!(args.method, Method::Bp);
    let bp_options = [
        args.damping.is_some(),
        args.tolerance.is_some(),
        args.max_sweeps.is_some(),
    ];
    ensure!(
        bp_options.iter().all(|&v| v == bp_mode),
        "damping, tolerance and max-sweeps are all required in BP only"
    );
    ensure!(
        !bp_mode || (args.plan.is_none() && args.analyze.is_none() && args.order.is_none()),
        "BP does not take a contraction plan"
    );
    ensure!(
        args.seed.is_some() == matches!(args.order, Some(Order::Random)),
        "seed is required for random and forbidden otherwise"
    );
    let ids = if benchmark {
        args.queries.clone()
    } else {
        (0..8).collect()
    };
    ensure!(
        ids.iter().all(|&id| (id as usize) < m.dims.len()),
        "unknown query ID"
    );
    fs::create_dir_all(&args.out)?;
    if bp_mode {
        let bp = bp::run(
            &m.sliced(&obs),
            &m.dims,
            &obs,
            args.damping.unwrap(),
            args.tolerance.unwrap(),
            args.max_sweeps.unwrap(),
        )?;
        for point in &bp.history {
            println!(
                "BP sweep={} residual={:.12e} cumulative_work={}",
                point.sweep, point.residual, point.cumulative_work
            );
        }
        return write(
            &args.out,
            "run.json",
            &json!({"method":"bp","observations":obs,"query_ids":ids,"bp":bp}),
        );
    }
    let factors = if benchmark {
        m.sliced(&obs)
    } else {
        m.capped(&obs)
    };
    let saved = args.analyze.as_ref().or(args.plan.as_ref());
    let p: Plan = if let Some(path) = saved {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        let method = match args.order.unwrap_or(Order::Greedy) {
            Order::Random => "random",
            Order::Greedy => "greedy",
            Order::Treesa => "treesa",
        };
        plan(&factors, &m.dims, &[], method, args.seed)?
    };
    ensure!(
        p.output.is_empty() && p.inputs.iter().eq(factors.iter().map(|f| &f.labels)),
        "saved scalar plan does not match input scopes"
    );
    let cost = recount(&p, &m.dims)?;
    write(&args.out, "complexity.json", &cost)?;
    if args.analyze.is_some() {
        println!(
            "analysis only: work={}, largest tensor={}",
            cost.work, cost.largest_tensor
        );
        return Ok(());
    }
    write(&args.out, "order.json", &p)?;
    let started = Instant::now();
    let (scalar, scale) = exact::contract(&factors, &m.dims, &p, benchmark)?;
    let scalar_seconds = started.elapsed().as_secs_f64();
    ensure!(scalar[0] > 0., "impossible observations");
    let log_z = scalar[0].ln() + scale;
    let mut marginals = BTreeMap::new();
    let mut query_plans = BTreeMap::new();
    let mut query_costs = BTreeMap::new();
    let mut normalization_error = 0_f64;
    for &id in &ids {
        let posterior = if benchmark && obs.contains_key(&id) {
            (0..m.dims[id as usize])
                .map(|s| f64::from(s == obs[&id]))
                .collect()
        } else {
            let q = reopen(&p, &[id])?;
            let (values, log_scale) = exact::contract(&factors, &m.dims, &q, benchmark)?;
            query_costs.insert(id, recount(&q, &m.dims)?);
            query_plans.insert(id, q);
            values
                .iter()
                .map(|x| x * (log_scale - log_z).exp())
                .collect::<Vec<_>>()
        };
        normalization_error = normalization_error.max((posterior.iter().sum::<f64>() - 1.).abs());
        marginals.insert(id, posterior);
    }
    if benchmark {
        ensure!(
            !ids.is_empty(),
            "benchmark exact inference requires explicit --queries"
        );
        let elapsed = started.elapsed().as_secs_f64();
        ensure!(
            normalization_error < 1e-10,
            "scaled benchmark marginals do not sum to one"
        );
        println!(
            "log10 Z={:.15}; marginals={marginals:?}; work={}",
            log_z / std::f64::consts::LN_10,
            cost.work
        );
        write(&args.out, "query-plans.json", &query_plans)?;
        write(&args.out, "query-complexities.json", &query_costs)?;
        return write(
            &args.out,
            "run.json",
            &json!({"method":"exact","observations":obs,"query_ids":ids,
            "log10_partition":log_z/std::f64::consts::LN_10,"log_partition":log_z,"marginals":marginals,
            "normalization_error":normalization_error,"scalar_contraction_seconds":scalar_seconds,
            "total_contraction_seconds":elapsed,"planning_seconds":p.planning_seconds,
            "retained_scalar_factors":factors.iter().filter(|f|f.labels.is_empty()).count()}),
        );
    }
    let mass = m.enumerate(&obs, &[])?[0];
    let exact_joint = |method: &str| -> Result<Vec<f64>> {
        let (values, denom) = if method == "enum" {
            (m.enumerate(&obs, &[1, 3])?, mass)
        } else {
            (
                exact::contract(&factors, &m.dims, &reopen(&p, &[1, 3])?, false)?.0,
                scalar[0],
            )
        };
        Ok(values.iter().map(|x| x / denom).collect())
    };
    let joint = exact_joint("contract")?;
    let enum_joint = exact_joint("enum")?;
    let mut enum_marginals = BTreeMap::new();
    let mut discrepancy = max_diff(&joint, &enum_joint);
    for &v in &ids {
        let values = m
            .enumerate(&obs, &[v])?
            .iter()
            .map(|x| x / mass)
            .collect::<Vec<_>>();
        discrepancy = discrepancy.max(max_diff(&values, &marginals[&v]));
        enum_marginals.insert(v, values);
    }
    normalization_error = normalization_error.max((joint.iter().sum::<f64>() - 1.).abs());
    let relative = (scalar[0] - mass).abs() / mass;
    let product: Vec<_> = marginals[&1]
        .iter()
        .flat_map(|x| marginals[&3].iter().map(move |y| x * y))
        .collect();
    let dependence: Vec<_> = joint.iter().zip(&product).map(|(x, y)| x - y).collect();
    ensure!(
        discrepancy < 1e-10 && relative < 1e-10 && normalization_error < 1e-10,
        "ASIA exact checks failed"
    );
    let examples: Value = exact::numerical_examples()?;
    println!(
        "evidence={:.15} TB={:.12} lung={:.12} bronchitis={:.12} marginal_max_abs={:.3e} evidence_relative={:.3e}",
        scalar[0], marginals[&1][0], marginals[&3][0], marginals[&4][0], discrepancy, relative
    );
    println!("numerical examples: {examples}");
    write(
        &args.out,
        "run.json",
        &json!({"method":"exact","observations":obs,
        "tenferro":{"evidence_mass":scalar[0],"marginals":marginals,"joint_TL_row_major":joint},
        "enumeration":{"evidence_mass":mass,"marginals":enum_marginals,"joint_TL_row_major":enum_joint},
        "product_TL_row_major":product,"dependence_TL_row_major":dependence,
        "marginal_max_absolute_error":discrepancy,"evidence_relative_error":relative,
        "normalization_error":normalization_error,"examples":examples}),
    )
}
