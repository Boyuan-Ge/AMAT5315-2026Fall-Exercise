use crate::model::{Factor, Id};
use anyhow::{Context, Result, ensure};
use omeco::{EinCode, GreedyMethod, NestedEinsum, ScoreFunction, TreeSA, optimize_code};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Step {
    pub left: usize,
    pub right: usize,
    pub retain: Vec<Id>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub method: String,
    pub settings: Value,
    pub planning_seconds: f64,
    pub inputs: Vec<Vec<Id>>,
    pub output: Vec<Id>,
    pub steps: Vec<Step>,
}
#[derive(Serialize)]
pub struct CostStep {
    pub result: usize,
    pub left: usize,
    pub right: usize,
    pub union: Vec<Id>,
    pub retain: Vec<Id>,
    pub work: u64,
    pub entries: u64,
}
#[derive(Serialize)]
pub struct Complexity {
    pub work: u64,
    pub largest_tensor: u64,
    pub largest_generated: u64,
    pub steps: Vec<CostStep>,
}

fn join(
    live: &mut [Option<Vec<Id>>],
    a: usize,
    b: usize,
    output: &[Id],
) -> Result<(Vec<Id>, Vec<Id>)> {
    ensure!(a != b, "cannot reuse a factor in its own pair");
    let lhs = live
        .get_mut(a)
        .and_then(Option::take)
        .context("left factor is not live")?;
    let rhs = live
        .get_mut(b)
        .and_then(Option::take)
        .context("right factor is not live")?;
    let union: BTreeSet<_> = lhs.into_iter().chain(rhs).collect();
    let retain = if live.iter().all(Option::is_none) {
        output.to_vec()
    } else {
        let outside: BTreeSet<_> = live
            .iter()
            .flatten()
            .flatten()
            .copied()
            .chain(output.iter().copied())
            .collect();
        union.intersection(&outside).copied().collect()
    };
    ensure!(
        retain.iter().all(|v| union.contains(v)),
        "output label missing"
    );
    Ok((union.into_iter().collect(), retain))
}
fn descend(tree: &NestedEinsum<Id>, n: usize, pairs: &mut Vec<(usize, usize)>) -> Result<usize> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => Ok(*tensor_index),
        NestedEinsum::Node { args, .. } => {
            ensure!(args.len() == 2, "expected binary omeco plan");
            let a = descend(&args[0], n, pairs)?;
            let b = descend(&args[1], n, pairs)?;
            let id = n + pairs.len();
            pairs.push((a, b));
            Ok(id)
        }
    }
}
fn from_pairs(inputs: &[Vec<Id>], output: &[Id], pairs: &[(usize, usize)]) -> Result<Vec<Step>> {
    let mut live: Vec<_> = inputs.iter().cloned().map(Some).collect();
    let mut steps = Vec::new();
    for &(left, right) in pairs {
        let (_, retain) = join(&mut live, left, right, output)?;
        live.push(Some(retain.clone()));
        steps.push(Step {
            left,
            right,
            retain,
        });
    }
    Ok(steps)
}
pub fn plan(
    factors: &[Factor],
    dims: &[usize],
    output: &[Id],
    method: &str,
    seed: Option<u64>,
) -> Result<Plan> {
    ensure!(!factors.is_empty(), "empty network");
    let start = Instant::now();
    let inputs: Vec<_> = factors.iter().map(|f| f.labels.clone()).collect();
    let n = inputs.len();
    let mut pairs = Vec::new();
    let settings = if method == "random" {
        let seed = seed.context("random requires --seed")?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut live: Vec<_> = (0..n).collect();
        while live.len() > 1 {
            let a = rng.gen_range(0..live.len());
            let left = live.swap_remove(a);
            let b = rng.gen_range(0..live.len());
            let right = live.swap_remove(b);
            live.push(n + pairs.len());
            pairs.push((left, right));
        }
        json!({"seed":seed,"rng":"rand StdRng; uniform pair"})
    } else {
        let code = EinCode::new(inputs.clone(), output.to_vec());
        let sizes: HashMap<_, _> = dims
            .iter()
            .enumerate()
            .map(|(v, &d)| (v as Id, d))
            .collect();
        let (tree, settings) = match method {
            "greedy" => (
                optimize_code(&code, &sizes, &GreedyMethod::default()),
                json!({"library":"omeco 0.2.6","method":"GreedyMethod defaults"}),
            ),
            "treesa" => {
                let config = TreeSA {
                    score: ScoreFunction::time_optimized(),
                    ..TreeSA::default()
                };
                let info = json!({"library":"omeco 0.2.6","seed_policy":"SmallRng(seed=42+trial index), built in",
                    "seeds":(42..42+config.ntrials).collect::<Vec<_>>(),
                    "betas":config.betas,"trials":config.ntrials,"iters_per_beta":config.niters,
                    "initializer":"greedy","time_weight":1,"space_weight":0,"read_write_weight":0});
                (optimize_code(&code, &sizes, &config), info)
            }
            _ => anyhow::bail!("unknown order {method}"),
        };
        descend(
            &tree.context("omeco could not plan this network")?,
            n,
            &mut pairs,
        )?;
        settings
    };
    let steps = from_pairs(&inputs, output, &pairs)?;
    let result = Plan {
        method: method.into(),
        settings,
        planning_seconds: start.elapsed().as_secs_f64(),
        inputs,
        output: output.to_vec(),
        steps,
    };
    recount(&result, dims)?;
    Ok(result)
}
pub fn reopen(p: &Plan, output: &[Id]) -> Result<Plan> {
    let mut other = p.clone();
    other.output = output.to_vec();
    other.steps = from_pairs(
        &p.inputs,
        output,
        &p.steps
            .iter()
            .map(|s| (s.left, s.right))
            .collect::<Vec<_>>(),
    )?;
    other.planning_seconds = 0.;
    other.settings = json!({"reused_tree":true,"source_method":p.method});
    Ok(other)
}
pub fn recount(plan: &Plan, dims: &[usize]) -> Result<Complexity> {
    let count = |ix: &[Id]| {
        ix.iter()
            .try_fold(1_u64, |a, &v| a.checked_mul(*dims.get(v as usize)? as u64))
            .context("unknown label or complexity overflow")
    };
    ensure!(
        !plan.inputs.is_empty() && plan.steps.len() + 1 == plan.inputs.len(),
        "plan must reduce every input"
    );
    let mut live: Vec<_> = plan.inputs.iter().cloned().map(Some).collect();
    let largest_input = plan
        .inputs
        .iter()
        .map(|x| count(x))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap();
    let (mut work, mut generated) = (0_u64, 0_u64);
    let mut steps = Vec::new();
    for s in &plan.steps {
        let (union, retain) = join(&mut live, s.left, s.right, &plan.output)?;
        ensure!(
            retain == s.retain,
            "incorrect retained labels in saved plan"
        );
        let w = count(&union)?;
        let entries = count(&retain)?;
        work = work.checked_add(w).context("work overflow")?;
        generated = generated.max(entries);
        steps.push(CostStep {
            result: live.len(),
            left: s.left,
            right: s.right,
            union,
            retain: retain.clone(),
            work: w,
            entries,
        });
        live.push(Some(retain));
    }
    Ok(Complexity {
        work,
        largest_tensor: largest_input.max(generated),
        largest_generated: generated,
        steps,
    })
}
