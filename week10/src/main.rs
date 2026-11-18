//! CPU f64 implementation of learning-sheet Algorithms 1--5.
use anyhow::{Result, ensure};
use clap::Parser;
use faer::{Mat, Par};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_parser=["svd", "boundary", "trg"])]
    method: String,
    #[arg(long)]
    l: Option<usize>,
    #[arg(long)]
    k: Option<f64>,
    #[arg(long)]
    chi: Option<usize>,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Serialize, Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}
impl Tensor {
    fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![0.; shape.iter().product()],
        }
    }
    fn at(&self, ix: &[usize]) -> f64 {
        self.data[self.offset(ix)]
    }
    fn offset(&self, ix: &[usize]) -> usize {
        ix.iter().zip(&self.shape).fold(0, |a, (i, d)| a * d + i)
    }
    fn set(&mut self, ix: &[usize], v: f64) {
        let o = self.offset(ix);
        self.data[o] = v;
    }
    fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    fn save(&self, dir: &Path, name: &str) -> Result<Value> {
        let file = format!("{name}.f64");
        let mut w = BufWriter::new(fs::File::create(dir.join(&file))?);
        for x in &self.data {
            w.write_all(&x.to_le_bytes())?;
        }
        w.flush()?;
        Ok(
            json!({"file":file,"shape":self.shape,"dtype":"little-endian float64","order":"C, last axis fastest"}),
        )
    }
}
fn matrix(t: &Tensor, m: usize, n: usize) -> Mat<f64> {
    Mat::from_fn(m, n, |i, j| t.data[i * n + j])
}
fn tensor(m: &Mat<f64>, shape: &[usize]) -> Tensor {
    Tensor {
        shape: shape.to_vec(),
        data: (0..m.nrows())
            .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
            .collect(),
    }
}
fn svd(m: &Mat<f64>) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let s = m
        .thin_svd(Par::Seq)
        .map_err(|e| anyhow::anyhow!("SVD {e:?}"))?;
    let vals = (0..m.nrows().min(m.ncols())).map(|i| s.S()[i]).collect();
    Ok((s.U().to_owned(), vals, s.V().transpose().to_owned()))
}
fn split(
    t: &Tensor,
    a: [usize; 2],
    b: [usize; 2],
    cap: usize,
) -> Result<(Tensor, Tensor, Vec<f64>, f64)> {
    let d = &t.shape;
    let m = d[a[0]] * d[a[1]];
    let n = d[b[0]] * d[b[1]];
    let mat = Mat::from_fn(m, n, |i, j| {
        let mut ix = [0; 4];
        ix[a[0]] = i / d[a[1]];
        ix[a[1]] = i % d[a[1]];
        ix[b[0]] = j / d[b[1]];
        ix[b[1]] = j % d[b[1]];
        t.at(&ix)
    });
    let (u, s, vt) = svd(&mat)?;
    let q = cap.min(s.len());
    let x = Mat::from_fn(m, q, |i, j| u[(i, j)] * s[j].sqrt());
    let y = Mat::from_fn(q, n, |i, j| s[i].sqrt() * vt[(i, j)]);
    let loss = s[q..].iter().map(|v| v * v).sum::<f64>() / s.iter().map(|v| v * v).sum::<f64>();
    Ok((
        tensor(&x, &[d[a[0]], d[a[1]], q]),
        tensor(&y, &[q, d[b[0]], d[b[1]]]),
        s,
        loss,
    ))
}
fn site(k: f64) -> Tensor {
    let w = [
        [k.cosh().sqrt(), k.sinh().sqrt()],
        [k.cosh().sqrt(), -k.sinh().sqrt()],
    ];
    let mut a = Tensor::zeros(&[2, 2, 2, 2]);
    for r in 0..2 {
        for u in 0..2 {
            for l in 0..2 {
                for d in 0..2 {
                    a.set(
                        &[r, u, l, d],
                        (0..2).map(|s| w[s][r] * w[s][u] * w[s][l] * w[s][d]).sum(),
                    );
                }
            }
        }
    }
    a
}
fn compress(b: &mut [Tensor], cap: usize) -> Result<Vec<Value>> {
    let n = b.len();
    for i in (1..n).rev() {
        let (a, p, c) = (b[i].shape[0], b[i].shape[1], b[i].shape[2]);
        let (u, s, vt) = svd(&matrix(&b[i], a, p * c))?;
        let q = s.len();
        let us = Mat::from_fn(a, q, |i, j| u[(i, j)] * s[j]);
        let left = &b[i - 1];
        let (la, lp) = (left.shape[0], left.shape[1]);
        let updated = &matrix(left, la * lp, a) * &us;
        b[i - 1] = tensor(&updated, &[la, lp, q]);
        b[i] = tensor(&vt, &[q, p, c]);
    }
    let mut diagnostics = Vec::new();
    for i in 0..n - 1 {
        let (a, p, c) = (b[i].shape[0], b[i].shape[1], b[i].shape[2]);
        let (u, s, vt) = svd(&matrix(&b[i], a * p, c))?;
        let q = cap.min(s.len());
        let kept = Mat::from_fn(a * p, q, |i, j| u[(i, j)]);
        let tail = s[q..].iter().map(|v| v * v).sum::<f64>();
        let all = s.iter().map(|v| v * v).sum::<f64>();
        diagnostics.push(json!({"bond":i,"before":c,"after":q,"discarded_squared":tail,"relative_discarded_squared":if all>0.{tail/all}else{0.}}));
        let sv = Mat::from_fn(q, c, |i, j| s[i] * vt[(i, j)]);
        let next = &b[i + 1];
        let (np, nc) = (next.shape[1], next.shape[2]);
        let updated = &sv * &matrix(next, c, np * nc);
        b[i] = tensor(&kept, &[a, p, q]);
        b[i + 1] = tensor(&updated, &[q, np, nc]);
    }
    Ok(diagnostics)
}
fn close_boundary(b: &[Tensor], edge: &[f64; 2]) -> f64 {
    let mut v = vec![1.];
    for t in b {
        let mut w = vec![0.; t.shape[2]];
        for (a, &va) in v.iter().enumerate() {
            for p in 0..2 {
                for (c, wc) in w.iter_mut().enumerate() {
                    *wc += va * t.at(&[a, p, c]) * edge[p];
                }
            }
        }
        v = w;
    }
    v[0]
}
fn boundary(l: usize, k: f64, cap: usize, dir: Option<&Path>) -> Result<(f64, Value)> {
    let a = site(k);
    let edge = [1. / k.cosh().sqrt(), 0.];
    let row: Vec<Tensor> = (0..l)
        .map(|i| {
            let mut t = Tensor::zeros(&[
                if i == 0 { 1 } else { 2 },
                2,
                2,
                if i + 1 == l { 1 } else { 2 },
            ]);
            for left in 0..t.shape[0] {
                for u in 0..2 {
                    for d in 0..2 {
                        for r in 0..t.shape[3] {
                            let val = if i == 0 {
                                (0..2).map(|x| edge[x] * a.at(&[r, u, x, d])).sum()
                            } else if i + 1 == l {
                                (0..2).map(|x| edge[x] * a.at(&[x, u, left, d])).sum()
                            } else {
                                a.at(&[r, u, left, d])
                            };
                            t.set(&[left, u, d, r], val);
                        }
                    }
                }
            }
            t
        })
        .collect();
    let mut b: Vec<Tensor> = (0..l)
        .map(|_| Tensor {
            shape: vec![1, 2, 1],
            data: edge.to_vec(),
        })
        .collect();
    let mut scales = Vec::new();
    let mut diag = Vec::new();
    let mut logscale = 0.;
    for y in 0..l {
        b = b
            .iter()
            .zip(&row)
            .map(|(b, t)| {
                let (al, be, le, re) = (b.shape[0], b.shape[2], t.shape[0], t.shape[3]);
                let mut c = Tensor::zeros(&[al * le, 2, be * re]);
                for x in 0..al {
                    for z in 0..be {
                        for left in 0..le {
                            for r in 0..re {
                                for d in 0..2 {
                                    c.set(
                                        &[x * le + left, d, z * re + r],
                                        (0..2)
                                            .map(|u| b.at(&[x, u, z]) * t.at(&[left, u, d, r]))
                                            .sum(),
                                    );
                                }
                            }
                        }
                    }
                }
                c
            })
            .collect();
        let trunc = compress(&mut b, cap)?;
        let scale = b[l - 1].norm();
        ensure!(scale > 0. && scale.is_finite(), "invalid boundary scale");
        b[l - 1].data.iter_mut().for_each(|x| *x /= scale);
        logscale += scale.ln();
        scales.push(scale);
        diag.push(json!({"row":y+1,"bonds":trunc}));
    }
    let z = close_boundary(&b, &edge);
    ensure!(z > 0., "nonpositive closure");
    let cores = if let Some(dir) = dir {
        b.iter()
            .enumerate()
            .map(|(i, b)| b.save(dir, &format!("core{i:03}")))
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![]
    };
    Ok((
        logscale + z.ln(),
        json!({"edge":edge,"scales":scales,"log_scale":logscale,"closure":z,"cores":cores,"truncation":diag}),
    ))
}
// Pair contractions leave (w,b,r,u) and (b,w,l,d); final GEMM sums (w,b).
// This is the paper's exact four-factor ring, with no extra approximation.
fn regroup(s1: &Tensor, s2: &Tensor, s3: &Tensor, s4: &Tensor) -> Tensor {
    let d = s1.shape[0];
    let q = s1.shape[2];
    let p = Mat::from_fn(d * q, d, |i, a| s1.at(&[i / q, a, i % q]));
    let r = Mat::from_fn(d, d * q, |a, j| s2.at(&[a, j / q, j % q]));
    let pr = &p * &r;
    let t = Mat::from_fn(d * q, d, |i, g| s3.at(&[i % q, i / q, g]));
    let u = Mat::from_fn(d, d * q, |g, j| s4.at(&[j % q, g, j / q]));
    let tu = &t * &u;
    let left = Mat::from_fn(q * q, d * d, |i, j| {
        pr[(j / d * q + i / q, j % d * q + i % q)]
    });
    let right = Mat::from_fn(d * d, q * q, |i, j| {
        tu[(i % d * q + j / q, i / d * q + j % q)]
    });
    tensor(&(&left * &right), &[q, q, q, q])
}
fn trg(l: usize, k: f64, cap: usize, dir: Option<&Path>) -> Result<(f64, Value)> {
    let mut a = site(k);
    let n = 2 * l.ilog2();
    let mut count = l * l;
    let mut ell = 0.;
    let mut steps = Vec::new();
    for j in 0..n {
        let scale = a.data.iter().map(|x| x.abs()).fold(0., f64::max);
        a.data.iter_mut().for_each(|x| *x /= scale);
        ell += (count as f64) * scale.ln();
        let (s1, s3, sa, la) = split(&a, [2, 1], [0, 3], cap)?;
        let (s2, s4, sb, lb) = split(&a, [3, 2], [1, 0], cap)?;
        let factors = if let Some(dir) = dir {
            [&s1, &s2, &s3, &s4]
                .iter()
                .enumerate()
                .map(|(i, t)| t.save(dir, &format!("step{j:02}-s{}", i + 1)))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        steps.push(json!({"step":j+1,"site_count":count,"scale":scale,"singular_a":sa,"singular_b":sb,"relative_discarded_squared":[la,lb],"factors":factors}));
        a = regroup(&s1, &s2, &s3, &s4);
        count /= 2;
    }
    let mut z = 0.;
    for i in 0..a.shape[0] {
        for j in 0..a.shape[1] {
            z += a.at(&[i, j, i, j]);
        }
    }
    ensure!(z > 0. && z.is_finite(), "invalid TRG closure");
    let final_tensor = if let Some(dir) = dir {
        a.save(dir, "final")?
    } else {
        Value::Null
    };
    Ok((
        ell + z.ln(),
        json!({"steps":steps,"final_tensor":final_tensor,"closure":z,"log_scale":ell}),
    ))
}
fn gaussian(dir: &Path) -> Result<Value> {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let n = 512;
    let mut m = Tensor::zeros(&[n, n]);
    for pair in m.data.chunks_mut(2) {
        let u = 1. - rng.r#gen::<f64>();
        let v = rng.r#gen::<f64>();
        let rad = (-2. * u.ln()).sqrt();
        let angle = std::f64::consts::TAU * v;
        pair[0] = rad * angle.cos();
        pair[1] = rad * angle.sin();
    }
    let (u, s, vt) = svd(&matrix(&m, n, n))?;
    let mut recons = Vec::new();
    for cap in [16, 64, 128, 256, 512] {
        let left = Mat::from_fn(n, cap, |i, j| u[(i, j)] * s[j]);
        let right = Mat::from_fn(cap, n, |i, j| vt[(i, j)]);
        recons.push(json!({"rank":cap,"matrix":tensor(&(&left*&right),&[n,n]).save(dir,&format!("rank{cap}"))?}));
    }
    Ok(
        json!({"seed":2026,"rng":"rand_chacha 0.3.1 ChaCha8Rng::seed_from_u64; rand 0.8.8 f64 uniform; Box-Muller pairs; row-major fill","matrix":m.save(dir,"matrix")?,"singular_values":s,"reconstructions":recons}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.out.exists(),
        "output exists; use a new directory to preserve evidence"
    );
    if args.method != "svd" {
        let l = args.l.ok_or_else(|| anyhow::anyhow!("--l required"))?;
        let k = args.k.ok_or_else(|| anyhow::anyhow!("--k required"))?;
        let chi = args.chi.ok_or_else(|| anyhow::anyhow!("--chi required"))?;
        ensure!(
            l >= 2 && k >= 0. && k.is_finite() && chi >= 1,
            "invalid lattice, K, or chi"
        );
        ensure!(
            args.method != "trg" || l.is_power_of_two(),
            "TRG L must be a power of two"
        );
    }
    fs::create_dir_all(&args.out)?;
    let start = Instant::now();
    let mut result = if args.method == "svd" {
        gaussian(&args.out)?
    } else {
        let (l, k, chi) = (args.l.unwrap(), args.k.unwrap(), args.chi.unwrap());
        let (logz, evidence) = if args.method == "boundary" {
            boundary(l, k, chi, Some(&args.out))?
        } else {
            trg(l, k, chi, Some(&args.out))?
        };
        json!({"method":args.method,"geometry":if args.method=="boundary"{"open square"}else{"periodic square; positive-direction bonds, including L=2 parallel bonds"},"l":l,"k":k,"chi":chi,"model":"H=-sum_edges s_i*s_j; s=+/-1; J=k_B=1; no field","log_z":logz,"g":logz/(l*l) as f64,"evidence":evidence})
    };
    result["solver_seconds"] = json!(start.elapsed().as_secs_f64());
    result["implementation"] = json!(
        "Rust 1.96.0; CPU f64; t4a-faer 0.33.0 sequential thin SVD and matrix multiplication"
    );
    fs::write(
        args.out.join("run.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "{}",
        json!({"out":args.out,"g":result.get("g"),"solver_seconds":result["solver_seconds"]})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn enumerated(l: usize, k: f64, periodic: bool) -> f64 {
        let mut z = 0.;
        for state in 0u64..(1u64 << (l * l)) {
            let spin = |i: usize| if state >> i & 1 == 0 { 1. } else { -1. };
            let mut energy = 0.;
            for y in 0..l {
                for x in 0..l {
                    let i = y * l + x;
                    if periodic || x + 1 < l {
                        energy -= spin(i) * spin(y * l + (x + 1) % l);
                    }
                    if periodic || y + 1 < l {
                        energy -= spin(i) * spin(((y + 1) % l) * l + x);
                    }
                }
            }
            z += (-k * energy).exp();
        }
        z.ln()
    }
    #[test]
    fn open_exact() {
        for k in [0., 0.44, 0.7] {
            assert!(
                (boundary(4, k, 8, None).unwrap().0 - enumerated(4, k, false)).abs() / 16. < 1e-8
            );
        }
    }
    #[test]
    fn torus_exact() {
        assert!((trg(2, 0.44, 16, None).unwrap().0 - enumerated(2, 0.44, true)).abs() / 4. < 1e-8);
        assert!((trg(4, 0.44, 8, None).unwrap().0 - enumerated(4, 0.44, true)).abs() / 16. < 2e-3);
    }
    #[test]
    fn infinite_temperature() {
        for l in [2, 4, 8] {
            for method in [boundary, trg] {
                assert!(
                    (method(l, 0., 8, None).unwrap().0 / (l * l) as f64 - 2f64.ln()).abs() < 1e-12
                );
            }
        }
    }
    #[test]
    fn split_and_gauge() {
        let t = Tensor {
            shape: vec![2, 3, 2, 2],
            data: (0..24).map(|i| (i as f64 * 0.31).sin()).collect(),
        };
        let (x, y, _, _) = split(&t, [2, 1], [0, 3], 20).unwrap();
        let p = &matrix(&x, 6, 4) * &matrix(&y, 4, 4);
        for l in 0..2 {
            for u in 0..3 {
                for r in 0..2 {
                    for d in 0..2 {
                        assert!((p[(l * 3 + u, r * 2 + d)] - t.at(&[r, u, l, d])).abs() < 1e-12);
                    }
                }
            }
        }
    }
    #[test]
    fn regroup_ring() {
        let t = site(0.44);
        let (a, c, _, _) = split(&t, [2, 1], [0, 3], 4).unwrap();
        let (b, d, _, _) = split(&t, [3, 2], [1, 0], 4).unwrap();
        let out = regroup(&a, &b, &c, &d);
        for r in 0..4 {
            for u in 0..4 {
                for l in 0..4 {
                    for dd in 0..4 {
                        let mut v = 0.;
                        for w in 0..2 {
                            for aa in 0..2 {
                                for bb in 0..2 {
                                    for g in 0..2 {
                                        v += a.at(&[w, aa, r])
                                            * b.at(&[aa, bb, u])
                                            * c.at(&[l, bb, g])
                                            * d.at(&[dd, g, w]);
                                    }
                                }
                            }
                        }
                        assert!((v - out.at(&[r, u, l, dd])).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
