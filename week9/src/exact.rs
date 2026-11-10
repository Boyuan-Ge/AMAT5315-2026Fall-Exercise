use crate::{
    model::{Factor, shape, states},
    plan::{Plan, recount},
};
use anyhow::{Result, ensure};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;

// Convert, rather than transpose, between the UAI and CPU storage conventions.
fn col_index(a: &[usize], dims: &[usize]) -> usize {
    a.iter().zip(dims).rev().fold(0, |i, (&s, &d)| i * d + s)
}
fn input(f: &Factor, dims: &[usize]) -> Result<Tensor> {
    let dims = shape(&f.labels, dims);
    let mut data = vec![0.; f.values.len()];
    for (i, &x) in f.values.iter().enumerate() {
        data[col_index(&states(i, &dims), &dims)] = x;
    }
    Ok(Tensor::from_vec_col_major(dims, data)?)
}
fn row_output(t: &Tensor) -> Result<Vec<f64>> {
    let values = t.as_slice::<f64>()?;
    let dims = t.shape();
    Ok((0..values.len())
        .map(|i| values[col_index(&states(i, dims), dims)])
        .collect())
}
fn normalize(t: Tensor, log_scale: f64) -> Result<(Tensor, f64)> {
    let data = t.as_slice::<f64>()?;
    let maximum = data.iter().copied().fold(0_f64, f64::max);
    ensure!(
        maximum.is_finite() && maximum > 0.,
        "zero or nonfinite network weight"
    );
    if maximum == 1. {
        return Ok((t, log_scale));
    }
    let values = data.iter().map(|v| v / maximum).collect();
    Ok((
        Tensor::from_vec_col_major(t.shape().to_vec(), values)?,
        log_scale + maximum.ln(),
    ))
}
pub fn contract(
    factors: &[Factor],
    dims: &[usize],
    p: &Plan,
    scaled: bool,
) -> Result<(Vec<f64>, f64)> {
    ensure!(
        factors.iter().map(|f| &f.labels).eq(p.inputs.iter()),
        "plan input scopes differ from model"
    );
    let cost = recount(p, dims)?;
    ensure!(
        cost.largest_tensor <= 1 << 25,
        "plan exceeds the 2^25-entry per-tensor safety limit"
    );
    let mut engine = CpuEngine::serial();
    let mut live = Vec::new();
    for f in factors {
        let t = input(f, dims)?;
        live.push(Some(if scaled { normalize(t, 0.)? } else { (t, 0.) }));
    }
    let labels: Vec<_> = p
        .inputs
        .iter()
        .chain(p.steps.iter().map(|s| &s.retain))
        .collect();
    for s in &p.steps {
        let (a, la) = live[s.left].take().unwrap();
        let (b, lb) = live[s.right].take().unwrap();
        let sub = EinsumSubscripts::new(&[labels[s.left], labels[s.right]], &s.retain);
        let t = [&a, &b].einsum_subscripts(&sub, &mut engine)?;
        let result = if scaled {
            normalize(t, la + lb)?
        } else {
            (t, 0.)
        };
        live.push(Some(result));
    }
    let (mut t, mut scale) = live.pop().unwrap().unwrap();
    if p.steps.is_empty() && p.inputs[0] != p.output {
        let sub = EinsumSubscripts::new(&[&p.inputs[0]], &p.output);
        t = [&t].einsum_subscripts(&sub, &mut engine)?;
        if scaled {
            (t, scale) = normalize(t, scale)?;
        }
    }
    Ok((row_output(&t)?, scale))
}
pub fn numerical_examples() -> Result<serde_json::Value> {
    use serde_json::json;
    let mut cpu = CpuEngine::serial();
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1., 3., 2., 4.])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![2., 1., 0., 2.])?;
    let ab = [&a, &b].einsum("ij,jk->ik", &mut cpu)?;
    let trace = [&a, &b].einsum("ij,ji->", &mut cpu)?;
    let k = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            0.5_f64.exp(),
            (-0.5_f64).exp(),
            (-0.5_f64).exp(),
            0.5_f64.exp(),
        ],
    )?;
    let z = [&k, &k, &k].einsum("ij,jk,ki->", &mut cpu)?;
    let spin = [-1., 1.];
    let mut enumerated = 0.;
    for s in spin {
        for t in spin {
            for u in spin {
                enumerated += (0.5_f64 * (s * t + t * u + u * s)).exp();
            }
        }
    }
    let z = z.as_slice::<f64>()?[0];
    ensure!((z - enumerated).abs() < 1e-12, "Ising check");
    Ok(
        json!({"AB_row_major":row_output(&ab)?,"trace_AB":trace.as_slice::<f64>()?[0],
        "ising_tenferro":z,"ising_enumeration":enumerated,"ising_absolute_error":(z-enumerated).abs()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::Model,
        plan::{plan, reopen},
    };

    #[test]
    fn asymmetric_axes_and_scalar_scale_are_preserved() -> Result<()> {
        let model = Model {
            dims: vec![2, 3],
            factors: vec![
                Factor {
                    labels: vec![1, 0],
                    values: vec![0.5, 1.4, 1.1, 0.4, 2.3, 0.2],
                },
                Factor {
                    labels: vec![0],
                    values: vec![0.3, 0.7],
                },
                Factor {
                    labels: vec![],
                    values: vec![0.07],
                },
            ],
        };
        let p = plan(&model.factors, &model.dims, &[], "greedy", None)?;
        let q = reopen(&p, &[0, 1])?;
        let expected = model.enumerate(&Default::default(), &[0, 1])?;
        let z: f64 = expected.iter().sum();
        for multiplier in [1., 10.] {
            let mut factors = model.factors.clone();
            factors[2].values[0] *= multiplier;
            let (scalar, scale) = contract(&factors, &model.dims, &p, true)?;
            let log_z = scalar[0].ln() + scale;
            assert!((log_z / std::f64::consts::LN_10 - (z * multiplier).log10()).abs() < 1e-12);
            let (joint, scale) = contract(&factors, &model.dims, &q, true)?;
            for (x, y) in joint.iter().zip(&expected) {
                assert!((x * (scale - log_z).exp() - y / z).abs() < 1e-12);
            }
        }
        Ok(())
    }
}
