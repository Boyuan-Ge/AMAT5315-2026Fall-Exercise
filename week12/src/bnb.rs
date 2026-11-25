use super::{Args, Glass, round};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{fs, time::Instant};

const EPS: f64 = 1e-6;
pub(super) fn search(g: &Glass, policy: &str, trace: bool) -> Result<Value> {
    let start = Instant::now();
    let mut stack = vec![(vec![None; g.fields.len()], 0)];
    let mut nodes = 0;
    let mut solves = 0;
    let mut best = i32::MAX;
    let mut witness = vec![];
    let mut log = vec![];
    while let Some((mut fix, depth)) = stack.pop() {
        nodes += 1;
        solves += 1;
        // The edge LP remains feasible for every spin box: any completion
        // together with its XOR values satisfies all four inequalities.
        let (bound, x) = g.solve(false, &fix)?;
        let mut event = json!({"node":nodes,"depth":depth,"fix":fix,"bound":bound,"spin_values":x});
        let free: Vec<_> = (0..x.len())
            .filter(|&i| fix[i].is_none() && x[i] > EPS && x[i] < 1. - EPS)
            .collect();
        if bound >= best as f64 - EPS {
            event["status"] = json!("pruned");
        } else if free.is_empty() {
            witness = round(&x);
            best = g.energy(&witness);
            ensure!(
                (best as f64 - bound).abs() < EPS,
                "integral LP bound disagrees with direct energy"
            );
            event["status"] = json!("incumbent");
        } else {
            let mut pick = free[0];
            if policy == "strong" {
                let mut high: f64 = -1.;
                for &i in &free {
                    let mut score = 1.;
                    for bit in [0, 1] {
                        fix[i] = Some(bit);
                        solves += 1;
                        let (child, _) = g.solve(false, &fix)?;
                        score *= if child >= best as f64 - EPS {
                            1e6
                        } else {
                            (child - bound).max(EPS)
                        };
                    }
                    fix[i] = None;
                    if score > high + 1e-10 * high.abs() {
                        high = score;
                        pick = i;
                    }
                }
            } else {
                for &i in &free {
                    if (x[i] - 0.5).abs() < (x[pick] - 0.5).abs() - EPS {
                        pick = i;
                    }
                }
            }
            let near = i32::from(x[pick] >= 0.5 - EPS);
            let mut far = fix.clone();
            far[pick] = Some(1 - near);
            fix[pick] = Some(near);
            stack.push((far, depth + 1));
            stack.push((fix, depth + 1));
            event["status"] = json!("branch");
            event["branch"] = json!(pick);
            event["near_bit"] = json!(near);
        }
        if trace {
            log.push(event);
        }
    }
    ensure!(best != i32::MAX, "no incumbent found");
    Ok(
        json!({"energy":best,"witness":witness,"nodes":nodes,"lp_solves":solves,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"trace":log}),
    )
}
fn tetrahedron() -> Glass {
    Glass {
        bonds: (0..4)
            .flat_map(|i| (i + 1..4).map(move |j| (i, j, 1)))
            .collect(),
        fields: vec![1, 0, 0, 0],
    }
}
pub(super) fn benchmark(a: &Args, start: Instant) -> Result<()> {
    let tetra = search(&tetrahedron(), "most-frac", true)?;
    ensure!(
        tetra["nodes"] == 5 && tetra["energy"] == -3,
        "unexpected tetrahedron trace"
    );
    let mut rows = Vec::new();
    for id in 0..a.count {
        let g = Glass::random(a.spins, a.edges, a.seed, id)?;
        let (ans, _) = g.solve(true, &vec![None; g.fields.len()])?;
        let mut row = search(&g, &a.policy, false)?;
        let err = (row["energy"].as_f64().unwrap() - ans).abs();
        ensure!(err < EPS, "incorrect energy for {id}: {err}");
        row["error"] = json!(err);
        row["highs_energy"] = json!(ans);
        row["id"] = json!(id);
        rows.push(row);
    }
    let avg =
        |k: &str| rows.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / rows.len() as f64;
    let out = json!({"schema":1,"mode":"bnb","seed":a.seed,"parameters":{"policy":a.policy,"spins":a.spins,"edges":a.edges,"count":a.count},"tetrahedron":tetra,"summary":{"mean_nodes":avg("nodes"),"mean_lp_solves":avg("lp_solves")},"instances":rows});
    fs::create_dir_all(&a.out)?;
    fs::write(a.out.join("run.json"), serde_json::to_vec_pretty(&out)?)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    println!(
        "bnb\tmean_nodes\t{}\t{}",
        out["summary"]["mean_nodes"],
        start.elapsed().as_secs_f64() * 1000.
    );
    Ok(())
}
#[test]
fn five_node_trace() {
    let v = search(&tetrahedron(), "most-frac", true).unwrap();
    assert_eq!(v["energy"], -3);
    assert_eq!(v["nodes"], 5);
    assert_eq!(v["lp_solves"], 5);
}
