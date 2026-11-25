use crate::{Glass, Relaxation};
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

pub const FEATURE_NAMES: [&str; 12] = [
    "x_val",
    "frac_dist",
    "field_h",
    "abs_field_h",
    "deg_norm",
    "bond_bias",
    "eff_field",
    "abs_eff_field",
    "mean_bond_dual",
    "mean_bond_gap",
    "mean_nbr_frac",
    "gap_ratio",
];
pub type Features = [f64; 12];

pub fn extract(
    g: &Glass,
    r: &Relaxation,
    fixed: &[Option<i32>],
    best: Option<i32>,
) -> Vec<Features> {
    let x: Vec<_> = r.x.iter().map(|v| v.clamp(0., 1.)).collect();
    let mut deg = vec![0; x.len()];
    let mut sums = vec![[0.; 5]; x.len()];
    let spin = |i: usize| 1. - 2. * x[i];
    let frac = |i: usize| 2. * x[i].min(1. - x[i]);
    let dual_scale = r.dual.iter().map(|x| x.abs()).fold(1e-9, f64::max);
    for (e, &(i, j, w)) in g.bonds.iter().enumerate() {
        let edge_dual = r.dual[4 * e..4 * e + 4]
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            / dual_scale;
        let edge_gap = (r.z[e] - 0.5 * (1. - spin(i) * spin(j))).abs();
        for (u, v) in [(i, j), (j, i)] {
            deg[u] += 1;
            sums[u][0] += (w.signum()) as f64;
            sums[u][1] += w as f64 * spin(v);
            sums[u][2] += edge_dual;
            sums[u][3] += edge_gap;
            if fixed[v].is_none() {
                sums[u][4] += frac(v);
            }
        }
    }
    let max_deg = deg.iter().copied().max().unwrap_or(1).max(1) as f64;
    let gap = best.map_or(1., |e| {
        ((e as f64 - r.bound) / (e.abs().max(1) as f64)).clamp(0., 2.)
    });
    (0..x.len())
        .map(|i| {
            let h = g.fields[i] as f64;
            let d = deg[i].max(1) as f64;
            let [bias, neighbors, dual, gaps, neighbor_frac] = sums[i];
            let eff = (h + neighbors) / (deg[i] + 1) as f64;
            [
                x[i],
                frac(i),
                h,
                h.abs(),
                deg[i] as f64 / max_deg,
                bias / d,
                eff,
                eff.abs(),
                dual / d,
                gaps / d,
                neighbor_frac / d,
                gap,
            ]
        })
        .collect()
}

#[derive(Deserialize)]
pub struct Model {
    mean: Features,
    std: Features,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    probes: Vec<Probe>,
}
#[derive(Deserialize)]
struct Probe {
    features: Vec<Features>,
    logits: Vec<f64>,
    probabilities: Vec<f64>,
}
impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let m: Self = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            m.w1.len() == 12
                && m.w1.iter().all(|row| row.len() == 48)
                && m.b1.len() == 48
                && m.w2.len() == 48,
            "expected a 12x48x1 net"
        );
        ensure!(
            m.std.iter().all(|s| s.is_finite() && *s >= 1e-6) && m.probes.len() == 6,
            "invalid scaling or probe shape"
        );
        Ok(m)
    }
    pub fn score(&self, f: &Features) -> f64 {
        let x: Features = std::array::from_fn(|j| (f[j] - self.mean[j]) / self.std[j]);
        self.b2
            + (0..48)
                .map(|i| {
                    let a = self.b1[i] + (0..12).map(|j| x[j] * self.w1[j][i]).sum::<f64>();
                    self.w2[i] * a.tanh()
                })
                .sum::<f64>()
    }
    pub fn verify_probes(&self) -> Result<(f64, f64)> {
        let mut max_prob: f64 = 0.;
        let mut max_logit: f64 = 0.;
        for p in &self.probes {
            let logits: Vec<_> = p.features.iter().map(|f| self.score(f)).collect();
            ensure!(
                !logits.is_empty()
                    && logits.len() == p.probabilities.len()
                    && logits.len() == p.logits.len(),
                "wrong probe shape"
            );
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom = logits.iter().map(|v| (v - top).exp()).sum::<f64>();
            for (i, &l) in logits.iter().enumerate() {
                max_logit = max_logit.max((l - p.logits[i]).abs());
                max_prob = max_prob.max(((l - top).exp() / denom - p.probabilities[i]).abs());
            }
        }
        ensure!(max_prob < 1e-4, "JAX probability mismatch: {max_prob}");
        Ok((max_prob, max_logit))
    }
}
