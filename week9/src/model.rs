use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

pub type Id = u32;
pub type Observations = BTreeMap<Id, usize>;
#[derive(Clone, Debug)]
pub struct Factor {
    pub labels: Vec<Id>,
    pub values: Vec<f64>, // UAI order: final axis varies fastest.
}
pub struct Model {
    pub dims: Vec<usize>,
    pub factors: Vec<Factor>,
}
pub fn states(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut out = vec![0; dims.len()];
    for (state, &d) in out.iter_mut().zip(dims).rev() {
        *state = index % d;
        index /= d;
    }
    out
}
pub fn shape(labels: &[Id], dims: &[usize]) -> Vec<usize> {
    labels.iter().map(|&v| dims[v as usize]).collect()
}
impl Model {
    pub fn read(path: &Path) -> Result<Self> {
        let data = fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let mut tokens = data.split_whitespace();
        ensure!(
            tokens.next() == Some("MARKOV"),
            "expected a MARKOV UAI model"
        );
        let mut next = || -> Result<usize> {
            Ok(tokens
                .next()
                .context("unexpected end of UAI header")?
                .parse()?)
        };
        let n = next()?;
        let dims: Vec<_> = (0..n).map(|_| next()).collect::<Result<_>>()?;
        ensure!(dims.iter().all(|&d| d > 0), "nonpositive cardinality");
        let m = next()?;
        let mut scopes = Vec::new();
        for _ in 0..m {
            let rank = next()?;
            let mut labels = Vec::new();
            for _ in 0..rank {
                let v = next()?;
                ensure!(v < n && !labels.contains(&(v as Id)), "invalid scope label");
                labels.push(v as Id);
            }
            scopes.push(labels);
        }
        let mut factors = Vec::new();
        for labels in scopes {
            let len: usize = tokens.next().context("missing table length")?.parse()?;
            ensure!(
                len == shape(&labels, &dims).iter().product::<usize>(),
                "UAI table length mismatch"
            );
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let x: f64 = tokens.next().context("missing table entry")?.parse()?;
                ensure!(x.is_finite() && x >= 0., "invalid potential");
                values.push(x);
            }
            factors.push(Factor { labels, values });
        }
        ensure!(tokens.next().is_none(), "unexpected trailing UAI tokens");
        Ok(Self { dims, factors })
    }
    pub fn check_evidence(&self, obs: &Observations) -> Result<()> {
        for (&v, &s) in obs {
            ensure!(
                self.dims.get(v as usize).is_some_and(|&d| s < d),
                "invalid observation {v}={s}"
            );
        }
        Ok(())
    }
    pub fn evidence_file(path: &Path) -> Result<Observations> {
        let data = fs::read_to_string(path)?;
        let parts = data
            .split_whitespace()
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            !parts.is_empty() && parts.len() == 1 + 2 * parts[0],
            "expected one UAI evidence case"
        );
        let obs: Observations = parts[1..]
            .chunks_exact(2)
            .map(|x| (x[0] as Id, x[1]))
            .collect();
        ensure!(obs.len() == parts[0], "duplicate observation");
        Ok(obs)
    }
    pub fn capped(&self, obs: &Observations) -> Vec<Factor> {
        let mut out = self.factors.clone();
        for (&v, &s) in obs {
            out.push(Factor {
                labels: vec![v],
                values: (0..self.dims[v as usize])
                    .map(|i| f64::from(i == s))
                    .collect(),
            });
        }
        out
    }
    pub fn sliced(&self, obs: &Observations) -> Vec<Factor> {
        let mut out = Vec::new();
        for f in &self.factors {
            let dims = shape(&f.labels, &self.dims);
            let labels = f
                .labels
                .iter()
                .filter(|v| !obs.contains_key(v))
                .copied()
                .collect();
            let values = f
                .values
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    let a = states(i, &dims);
                    f.labels
                        .iter()
                        .zip(a)
                        .all(|(v, s)| obs.get(v).is_none_or(|&wanted| wanted == s))
                        .then_some(x)
                })
                .collect();
            out.push(Factor { labels, values }); // including rank-zero factors
        }
        let used: BTreeSet<_> = out.iter().flat_map(|f| f.labels.iter().copied()).collect();
        for (v, &d) in self.dims.iter().enumerate() {
            if !obs.contains_key(&(v as Id)) && !used.contains(&(v as Id)) {
                out.push(Factor {
                    labels: vec![v as Id],
                    values: vec![1.; d],
                });
            }
        }
        out
    }
    pub fn enumerate(&self, obs: &Observations, open: &[Id]) -> Result<Vec<f64>> {
        let count = self
            .dims
            .iter()
            .try_fold(1_usize, |a, &b| a.checked_mul(b))
            .context("assignment count overflow")?;
        ensure!(
            count <= 1_000_000,
            "enumeration limited to 1,000,000 assignments"
        );
        let mut result = vec![0.; shape(open, &self.dims).iter().product()];
        for i in 0..count {
            let a = states(i, &self.dims);
            if obs.iter().any(|(&v, &s)| a[v as usize] != s) {
                continue;
            }
            let p: f64 = self
                .factors
                .iter()
                .map(|f| {
                    let index = f
                        .labels
                        .iter()
                        .fold(0, |i, &v| i * self.dims[v as usize] + a[v as usize]);
                    f.values[index]
                })
                .product();
            let index = open
                .iter()
                .fold(0, |i, &v| i * self.dims[v as usize] + a[v as usize]);
            result[index] += p;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uai_asymmetric_scope_and_scalar_slice() -> Result<()> {
        let m = Model::read(Path::new("asia.uai"))?;
        assert_eq!(m.factors[1].labels, [1, 0]);
        assert_eq!(m.factors[1].values, [0.05, 0.01, 0.95, 0.99]);
        let f = m.sliced(&[(0, 0)].into());
        assert!(f[0].labels.is_empty());
        assert_eq!(f[0].values, [0.01]);
        assert_eq!(f[1].values, [0.05, 0.95]);
        Ok(())
    }
}
