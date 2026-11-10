use crate::model::{Factor, Observations, shape, states};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
pub struct Sweep {
    pub sweep: usize,
    pub residual: f64,
    pub cumulative_work: u64,
    pub beliefs: Vec<Vec<f64>>,
}
#[derive(Serialize)]
pub struct Report {
    pub damping: f64,
    pub tolerance: f64,
    pub max_sweeps: usize,
    pub sweeps: usize,
    pub converged: bool,
    pub residual: f64,
    pub cumulative_work: u64,
    pub work_per_proposal: u64,
    pub runtime_seconds: f64,
    pub marginals: Vec<Vec<f64>>,
    pub history: Vec<Sweep>,
}
fn normalize(x: &mut [f64]) -> Result<()> {
    let sum: f64 = x.iter().sum();
    ensure!(
        sum > 0. && sum.is_finite(),
        "zero/nonfinite BP normalization"
    );
    x.iter_mut().for_each(|v| *v /= sum);
    Ok(())
}
pub fn run(
    factors: &[Factor],
    dims: &[usize],
    obs: &Observations,
    eta: f64,
    tolerance: f64,
    limit: usize,
) -> Result<Report> {
    ensure!(
        eta > 0. && eta <= 1. && tolerance > 0. && tolerance.is_finite() && limit > 0,
        "invalid BP settings"
    );
    let start = Instant::now();
    let mut neighbors = vec![Vec::new(); dims.len()];
    let mut edges = Vec::new();
    let mut factor_edges = Vec::new();
    let mut assignments = Vec::new();
    for f in factors {
        if f.labels.is_empty() {
            ensure!(f.values[0] > 0., "zero evidence scalar");
        }
        let mut ids = Vec::new();
        for &label in &f.labels {
            let v = label as usize;
            ids.push(edges.len());
            neighbors[v].push(edges.len());
            edges.push(v);
        }
        factor_edges.push(ids);
        let shape = shape(&f.labels, dims);
        assignments.push(
            (0..f.values.len())
                .map(|i| states(i, &shape))
                .collect::<Vec<_>>(),
        );
    }
    let uniform: Vec<_> = edges
        .iter()
        .map(|&v| vec![1. / dims[v] as f64; dims[v]])
        .collect();
    let mut vf = uniform.clone();
    let mut fv = uniform.clone();
    let mut history = Vec::new();
    let variable_work: u64 = edges
        .iter()
        .map(|&v| (dims[v] * (neighbors[v].len() - 1)) as u64)
        .sum();
    let factor_work: u64 = factors
        .iter()
        .map(|f| (f.values.len() * f.labels.len() * f.labels.len()) as u64)
        .sum();
    let work = variable_work + factor_work;
    for sweep in 0..=limit {
        // Both sets of proposals read the previous message state.
        let mut next_vf = uniform.clone();
        let mut next_fv = uniform.clone();
        for (e, &v) in edges.iter().enumerate() {
            for s in 0..dims[v] {
                next_vf[e][s] = neighbors[v]
                    .iter()
                    .filter(|&&j| j != e)
                    .map(|&j| fv[j][s])
                    .product();
            }
            normalize(&mut next_vf[e])?;
        }
        for (fi, f) in factors.iter().enumerate() {
            let fe = &factor_edges[fi];
            for (a, &e) in fe.iter().enumerate() {
                next_fv[e].fill(0.);
                for (i, &x) in f.values.iter().enumerate() {
                    let state = &assignments[fi][i];
                    let mut contribution = x;
                    for (b, &source) in fe.iter().enumerate() {
                        if a != b {
                            contribution *= vf[source][state[b]];
                        }
                    }
                    next_fv[e][state[a]] += contribution;
                }
                normalize(&mut next_fv[e])?;
            }
        }
        let residual = vf
            .iter()
            .flatten()
            .chain(fv.iter().flatten())
            .zip(next_vf.iter().flatten().chain(next_fv.iter().flatten()))
            .map(|(x, y)| (x - y).abs())
            .fold(0_f64, f64::max);
        let mut beliefs = Vec::new();
        for (v, &d) in dims.iter().enumerate() {
            let mut b: Vec<f64> = if let Some(&state) = obs.get(&(v as u32)) {
                (0..d).map(|s| f64::from(state == s)).collect()
            } else {
                (0..d)
                    .map(|s| neighbors[v].iter().map(|&e| fv[e][s]).product())
                    .collect()
            };
            normalize(&mut b)?;
            beliefs.push(b);
        }
        // The diagnostic proposal is actually evaluated, including at sweep 0 and the returned state.
        let cumulative_work = (sweep as u64 + 1) * work;
        history.push(Sweep {
            sweep,
            residual,
            cumulative_work,
            beliefs: beliefs.clone(),
        });
        if residual <= tolerance || sweep == limit {
            return Ok(Report {
                damping: eta,
                tolerance,
                max_sweeps: limit,
                sweeps: sweep,
                converged: residual <= tolerance,
                residual,
                cumulative_work,
                work_per_proposal: work,
                runtime_seconds: start.elapsed().as_secs_f64(),
                marginals: beliefs,
                history,
            });
        }
        for (old, new) in vf
            .iter_mut()
            .flatten()
            .chain(fv.iter_mut().flatten())
            .zip(next_vf.iter().flatten().chain(next_fv.iter().flatten()))
        {
            *old = (1. - eta) * *old + eta * new;
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    #[test]
    fn nonuniform_tree_matches_independent_enumeration() -> Result<()> {
        let model = Model {
            dims: vec![2; 3],
            factors: vec![
                Factor {
                    labels: vec![0],
                    values: vec![0.83, 0.17],
                },
                Factor {
                    labels: vec![0, 1],
                    values: vec![0.8, 0.2, 0.1, 0.9],
                },
                Factor {
                    labels: vec![1, 2],
                    values: vec![0.3, 0.7, 0.9, 0.1],
                },
                Factor {
                    labels: vec![2],
                    values: vec![0.41, 0.59],
                },
            ],
        };
        let obs = Observations::new();
        let report = run(&model.factors, &model.dims, &obs, 0.5, 1e-12, 1000)?;
        assert!(report.converged);
        for v in 0..3 {
            let p = model.enumerate(&obs, &[v])?;
            let z: f64 = p.iter().sum();
            for (x, y) in p.iter().zip(&report.marginals[v as usize]) {
                assert!((x / z - y).abs() < 1e-10);
            }
        }
        Ok(())
    }
}
