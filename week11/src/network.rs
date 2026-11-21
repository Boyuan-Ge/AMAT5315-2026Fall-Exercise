use anyhow::{ensure, Result};
use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub logz: f64,
    pub energy: i64,
    pub count: u128,
}
impl Weight {
    pub fn from_energy(energy: i64, beta: f64) -> Self {
        Self {
            logz: -beta * energy as f64,
            energy,
            count: 1,
        }
    }
    fn product(self, b: Self) -> Self {
        Self {
            logz: self.logz + b.logz,
            energy: self.energy + b.energy,
            count: self.count * b.count,
        }
    }
    fn add(&mut self, b: Self) {
        let max = self.logz.max(b.logz);
        self.logz = max + ((self.logz - max).exp() + (b.logz - max).exp()).ln();
        if b.energy < self.energy {
            self.energy = b.energy;
            self.count = b.count;
        } else if b.energy == self.energy {
            self.count += b.count;
        }
    }
}
pub struct Tensor {
    pub labels: Vec<usize>,
    pub entries: BTreeMap<u64, Weight>,
}
pub struct Step {
    pub a: usize,
    pub b: usize,
    pub union: Vec<usize>,
    pub keep: Vec<usize>,
}
pub fn plan(factors: &[Tensor], order: &str) -> Result<(Vec<Step>, f64, usize)> {
    let inputs: Vec<_> = factors.iter().map(|t| t.labels.clone()).collect();
    let sizes: HashMap<_, _> = inputs.iter().flatten().map(|&i| (i, 2)).collect();
    let code = EinCode::new(inputs.clone(), vec![]);
    let tree = if order == "greedy" {
        optimize_code(&code, &sizes, &GreedyMethod::default())
    } else {
        optimize_code(&code, &sizes, &TreeSA::default())
    };
    fn visit(
        t: &NestedEinsum<usize>,
        live: &mut Vec<Option<Vec<usize>>>,
        steps: &mut Vec<Step>,
    ) -> Result<usize> {
        match t {
            NestedEinsum::Leaf { tensor_index } => Ok(*tensor_index),
            NestedEinsum::Node { args, .. } => {
                ensure!(args.len() == 2, "binary tree required");
                let a = visit(&args[0], live, steps)?;
                let b = visit(&args[1], live, steps)?;
                let union: BTreeSet<_> = live[a]
                    .take()
                    .unwrap()
                    .into_iter()
                    .chain(live[b].take().unwrap())
                    .collect();
                let outside: BTreeSet<_> = live.iter().flatten().flatten().copied().collect();
                let keep: Vec<_> = union.intersection(&outside).copied().collect();
                let i = live.len();
                live.push(Some(keep.clone()));
                steps.push(Step {
                    a,
                    b,
                    union: union.into_iter().collect(),
                    keep,
                });
                Ok(i)
            }
        }
    }
    let mut live = inputs.into_iter().map(Some).collect();
    let mut steps = vec![];
    visit(
        &tree.ok_or_else(|| anyhow::anyhow!("planner found no tree"))?,
        &mut live,
        &mut steps,
    )?;
    let tc = steps
        .iter()
        .map(|s| 2f64.powi(s.union.len() as i32))
        .sum::<f64>()
        .log2();
    let sc = steps
        .iter()
        .map(|s| s.keep.len())
        .chain(factors.iter().map(|t| t.labels.len()))
        .max()
        .unwrap();
    Ok((steps, tc, sc))
}
fn positions(from: &[usize], to: &[usize]) -> Vec<usize> {
    to.iter()
        .map(|x| from.iter().position(|y| x == y).unwrap())
        .collect()
}
fn project(x: u64, pos: &[usize]) -> u64 {
    pos.iter()
        .enumerate()
        .map(|(j, &i)| ((x >> i) & 1) << j)
        .sum()
}
fn scatter(x: u64, pos: &[usize]) -> u64 {
    pos.iter()
        .enumerate()
        .map(|(j, &i)| ((x >> j) & 1) << i)
        .sum()
}
pub type Branches = BTreeMap<u64, Vec<(u64, u64, Weight)>>;
pub fn contract(t: &mut Vec<Tensor>, steps: &[Step], sample: bool) -> Result<Vec<Branches>> {
    let mut traces = vec![];
    for s in steps {
        ensure!(
            s.union.len() <= 64,
            "contraction union too wide; plan-only mode recommended"
        );
        let (a, b) = (&t[s.a], &t[s.b]);
        let shared: Vec<_> = a
            .labels
            .iter()
            .filter(|x| b.labels.contains(x))
            .copied()
            .collect();
        let ia = positions(&a.labels, &shared);
        let ib = positions(&b.labels, &shared);
        let ua = positions(&s.union, &a.labels);
        let ub = positions(&s.union, &b.labels);
        let keep = positions(&s.union, &s.keep);
        let mut index: HashMap<u64, Vec<(u64, Weight)>> = HashMap::new();
        for (&x, &v) in &b.entries {
            index.entry(project(x, &ib)).or_default().push((x, v));
        }
        let mut entries = BTreeMap::<u64, Weight>::new();
        let mut trace = Branches::new();
        for (&x, &v) in &a.entries {
            if let Some(rows) = index.get(&project(x, &ia)) {
                let u = scatter(x, &ua);
                for &(y, w) in rows {
                    let z = project(u | scatter(y, &ub), &keep);
                    let m = v.product(w);
                    entries.entry(z).and_modify(|v| v.add(m)).or_insert(m);
                    if sample {
                        trace.entry(z).or_default().push((x, y, m));
                    }
                }
            }
        }
        t[s.a].entries.clear();
        t[s.b].entries.clear();
        t.push(Tensor {
            labels: s.keep.clone(),
            entries,
        });
        traces.push(trace);
    }
    Ok(traces)
}
pub fn draw(
    t: &[Tensor],
    steps: &[Step],
    trace: &[Branches],
    rng: &mut impl Rng,
    n: usize,
) -> Vec<u8> {
    let leaves = t.len() - steps.len();
    let mut result = vec![0; n];
    let mut keys = vec![0; t.len()];
    for (i, step) in steps.iter().enumerate().rev() {
        let rows = &trace[i][&keys[leaves + i]];
        let max = rows
            .iter()
            .map(|r| r.2.logz)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut u = rng.gen::<f64>() * rows.iter().map(|r| (r.2.logz - max).exp()).sum::<f64>();
        let mut picked = rows.last().unwrap();
        for r in rows {
            u -= (r.2.logz - max).exp();
            if u < 0. {
                picked = r;
                break;
            }
        }
        keys[step.a] = picked.0;
        keys[step.b] = picked.1;
    }
    for (i, node) in t[..leaves].iter().enumerate() {
        for (j, &label) in node.labels.iter().enumerate() {
            result[label] = ((keys[i] >> j) & 1) as u8;
        }
    }
    result
}

pub fn queens(n: usize) -> (Vec<Tensor>, usize) {
    let mut next = n * n;
    let mut terms = vec![];
    let lines = (0..n)
        .map(|r| (true, (0..n).map(|c| r * n + c).collect::<Vec<_>>()))
        .chain((0..n).map(|c| (true, (0..n).map(|r| r * n + c).collect())))
        .chain((0..2 * n - 1).flat_map(|d| {
            [false, true].map(move |rev| {
                (
                    false,
                    (0..n * n)
                        .filter(|&i| {
                            let (r, c) = (i / n, i % n);
                            r + if rev { n - 1 - c } else { c } == d
                        })
                        .collect(),
                )
            })
        }));
    for (exact, line) in lines {
        let mut previous = None;
        for (i, &x) in line.iter().enumerate() {
            let last = i + 1 == line.len();
            let after = if last {
                None
            } else {
                let label = next;
                next += 1;
                Some(label)
            };
            let labels: Vec<_> = previous.into_iter().chain([x]).chain(after).collect();
            let mut entries = BTreeMap::new();
            for (a, b, c) in [(0, 0, 0), (0, 1, 1), (1, 0, 1)] {
                if (previous.is_none() && a != 0) || (last && exact && c != 1) {
                    continue;
                }
                let key = previous
                    .map(|_| a)
                    .into_iter()
                    .chain([b])
                    .chain(after.map(|_| c))
                    .enumerate()
                    .map(|(j, v)| v << j)
                    .sum();
                entries.insert(key, Weight::from_energy(0, 0.));
            }
            terms.push(Tensor { labels, entries });
            previous = after;
        }
    }
    (terms, next)
}
