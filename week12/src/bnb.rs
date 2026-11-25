use super::{Args, Glass, round};
use crate::learned::{self, Model};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{fs, time::Instant};

const EPS: f64 = 1e-6;
fn search(
    g: &Glass,
    policy: &str,
    trace: bool,
    model: Option<&Model>,
    export_labels: bool,
) -> Result<Value> {
    let start = Instant::now();
    let mut stack = vec![(vec![None; g.fields.len()], 0)];
    let mut nodes = 0;
    let mut solves = 0;
    let mut best = i32::MAX;
    let mut witness = vec![];
    let mut log = vec![];
    let mut labels = vec![];
    let mut first_splits = vec![];
    let mut root_scores = vec![];
    let mut lp_ms = 0.;
    let mut infer_ms = 0.;
    let mut inference_calls = 0;
    while let Some((mut fix, depth)) = stack.pop() {
        nodes += 1;
        solves += 1;
        // The edge LP remains feasible for every spin box: any completion
        // together with its XOR values satisfies all four inequalities.
        let get_features = policy == "neural"
            || (policy == "root-neural" && depth == 0)
            || (export_labels && labels.len() < 80);
        let clock = Instant::now();
        let lp = g.relaxation(false, &fix, get_features)?;
        lp_ms += clock.elapsed().as_secs_f64() * 1000.;
        let (bound, x) = (lp.bound, &lp.x);
        let mut event = if trace {
            json!({"node":nodes,"depth":depth,"fix":fix,"bound":bound,"spin_values":x})
        } else {
            Value::Null
        };
        let free: Vec<_> = (0..x.len())
            .filter(|&i| fix[i].is_none() && x[i] > EPS && x[i] < 1. - EPS)
            .collect();
        if bound >= best as f64 - EPS {
            if trace {
                event["status"] = json!("pruned");
            }
        } else if free.is_empty() {
            witness = round(x);
            best = g.energy(&witness);
            ensure!(
                (best as f64 - bound).abs() < EPS,
                "integral LP bound disagrees with direct energy"
            );
            if trace {
                event["status"] = json!("incumbent");
            }
        } else {
            let mut pick = free[0];
            if policy == "strong" {
                let mut high: f64 = -1.;
                let collect = export_labels && free.len() >= 2 && labels.len() < 80;
                let features = collect
                    .then(|| learned::extract(g, &lp, &fix, (best != i32::MAX).then_some(best)));
                let mut candidates = vec![];
                for &i in &free {
                    let mut gains = [0.; 2];
                    let mut pruned = 0;
                    for bit in [0, 1] {
                        fix[i] = Some(bit);
                        solves += 1;
                        let clock = Instant::now();
                        let (child, _) = g.solve(false, &fix)?;
                        lp_ms += clock.elapsed().as_secs_f64() * 1000.;
                        if child >= best as f64 - EPS {
                            gains[bit as usize] = 1e6;
                            pruned += 1;
                        } else {
                            gains[bit as usize] = child - bound;
                        }
                    }
                    fix[i] = None;
                    let score = gains[0].max(EPS) * gains[1].max(EPS);
                    if let Some(f) = &features {
                        candidates.push(json!({"spin":i,"child_gains":gains,"score":score,"pruned_children":pruned,"features":f[i]}));
                    }
                    if score > high + 1e-10 * high.abs() {
                        high = score;
                        pick = i;
                    }
                }
                if collect {
                    labels.push(json!({"node":nodes,"depth":depth,"spin_values":x,"fix":fix,"bound":bound,
                    "incumbent":(best!=i32::MAX).then_some(best),"selected_spin":pick,"candidates":candidates}));
                }
            } else if let Some(net) =
                model.filter(|_| policy == "neural" || policy == "root-neural")
            {
                if policy == "neural" || depth == 0 {
                    let clock = Instant::now();
                    let f = learned::extract(g, &lp, &fix, (best != i32::MAX).then_some(best));
                    inference_calls += 1;
                    if policy == "neural" {
                        let mut high = f64::NEG_INFINITY;
                        for &i in &free {
                            let score = net.score(&f[i]);
                            if score > high {
                                high = score;
                                pick = i;
                            }
                        }
                    } else {
                        root_scores = f.iter().map(|v| net.score(v)).collect();
                        let mean = root_scores.iter().sum::<f64>() / root_scores.len() as f64;
                        let std = (root_scores.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                            / root_scores.len() as f64)
                            .sqrt()
                            .max(EPS);
                        root_scores.iter_mut().for_each(|v| *v = (*v - mean) / std);
                    }
                    infer_ms += clock.elapsed().as_secs_f64() * 1000.;
                }
                if policy == "root-neural" {
                    // This cached rule does not construct new features or run the MLP below the root.
                    let score = |i: usize| root_scores[i] + 6. * x[i].min(1. - x[i]);
                    for &i in &free {
                        if score(i) > score(pick) {
                            pick = i;
                        }
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
            if first_splits.len() < 2 {
                first_splits.push((pick, near));
            }
            let mut far = fix.clone();
            far[pick] = Some(1 - near);
            fix[pick] = Some(near);
            stack.push((far, depth + 1));
            stack.push((fix, depth + 1));
            if trace {
                event["status"] = json!("branch");
                event["branch"] = json!(pick);
                event["near_bit"] = json!(near);
            }
        }
        if trace {
            log.push(event);
        }
    }
    ensure!(best != i32::MAX, "no incumbent found");
    Ok(
        json!({"energy":best,"witness":witness,"nodes":nodes,"lp_solves":solves,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,
            "lp_ms":lp_ms,"inference_ms":infer_ms,"inference_calls":inference_calls,"first_splits":first_splits,"trace":log,"labels":labels}),
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
    ensure!(
        a.model.is_none() && (!a.export_labels || a.policy == "strong"),
        "label export requires strong branching"
    );
    let tetra = search(&tetrahedron(), "most-frac", true, None, false)?;
    ensure!(
        tetra["nodes"] == 5 && tetra["energy"] == -3,
        "unexpected tetrahedron trace"
    );
    let mut rows = Vec::new();
    let mut train = vec![];
    let mut val = vec![];
    for id in 0..a.count {
        let g = Glass::random(a.spins, a.edges, a.seed, id)?;
        let (ans, _) = g.solve(true, &vec![None; g.fields.len()])?;
        let mut row = search(&g, &a.policy, false, None, a.export_labels)?;
        let ls = row.as_object_mut().unwrap().remove("labels").unwrap();
        if a.export_labels {
            ensure!(a.count == 48, "label export requires exactly 48 instances");
            let bucket = if id < 36 { &mut train } else { &mut val };
            bucket.push(json!({"id":id,"bonds":g.bonds,"fields":g.fields,"nodes":ls}));
        }
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
    if a.export_labels {
        let dataset = json!({"schema":1,"seed":a.seed,"feature_names":learned::FEATURE_NAMES,"train":train,"validation":val});
        fs::write(
            a.out.join("dataset.json"),
            serde_json::to_vec_pretty(&dataset)?,
        )?;
    }
    println!("mode\tmetric\tvalue\truntime_ms");
    println!(
        "bnb\tmean_nodes\t{}\t{}",
        out["summary"]["mean_nodes"],
        start.elapsed().as_secs_f64() * 1000.
    );
    Ok(())
}
pub(super) fn evaluate(a: &Args, start: Instant) -> Result<()> {
    const RULES: [&str; 4] = ["most-frac", "strong", "neural", "root-neural"];
    ensure!(!a.export_labels, "labels are not exported in eval mode");
    let model = Model::load(a.model.as_ref().context("eval requires --model")?)?;
    let (probe, logit) = model.verify_probes()?;
    let mut rows = vec![];
    for id in 0..a.count {
        let g = Glass::random(a.spins, a.edges, a.seed, id)?;
        let (opt, _) = g.solve(true, &vec![None; g.fields.len()])?;
        let mut policies = serde_json::Map::new();
        // Rotate order to avoid systematically favoring one timing position.
        for k in 0..RULES.len() {
            let rule = RULES[(id + k) % RULES.len()];
            let mut row = search(&g, rule, false, Some(&model), false)?;
            let error = (row["energy"].as_f64().unwrap() - opt).abs();
            ensure!(error < EPS, "{rule} optimum mismatch on {id}");
            ensure!(
                rule == "strong" || row["nodes"] == row["lp_solves"],
                "hidden LP lookahead"
            );
            if rule == "root-neural" {
                ensure!(
                    row["inference_calls"].as_u64().unwrap() <= 1,
                    "root network called below root"
                );
            }
            row["error"] = json!(error);
            row.as_object_mut().unwrap().remove("labels");
            policies.insert(rule.into(), row);
        }
        rows.push(json!({"id":id,"highs_energy":opt,"rules":policies}));
    }
    let mut summary = serde_json::Map::new();
    for rule in RULES {
        let avg = |key: &str| {
            rows.iter()
                .map(|r| r["rules"][rule][key].as_f64().unwrap())
                .sum::<f64>()
                / rows.len() as f64
        };
        let max_error = rows
            .iter()
            .map(|r| r["rules"][rule]["error"].as_f64().unwrap())
            .fold(0., f64::max);
        summary.insert(rule.into(),json!({"max_error":max_error,"mean_nodes":avg("nodes"),"mean_lp_solves":avg("lp_solves"),
            "mean_total_ms":avg("elapsed_ms"),"mean_lp_ms":avg("lp_ms"),"mean_inference_ms":avg("inference_ms")}));
    }
    let saving = rows
        .iter()
        .max_by_key(|r| {
            r["rules"]["most-frac"]["nodes"].as_i64().unwrap()
                - r["rules"]["neural"]["nodes"].as_i64().unwrap()
        })
        .unwrap();
    let out = json!({"schema":1,"mode":"eval","seed":a.seed,"parameters":{"spins":a.spins,"edges":a.edges,"count":a.count},
        "max_probe_probability_error":probe,"max_probe_logit_error":logit,"summary":summary,"largest_neural_saving":saving,"instances":rows});
    fs::create_dir_all(&a.out)?;
    fs::write(a.out.join("run.json"), serde_json::to_vec_pretty(&out)?)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for rule in RULES {
        println!(
            "eval\t{rule}_mean_nodes\t{}\t{}",
            out["summary"][rule]["mean_nodes"],
            start.elapsed().as_secs_f64() * 1000.
        );
    }
    Ok(())
}
#[test]
fn five_node_trace() {
    let v = search(&tetrahedron(), "most-frac", true, None, false).unwrap();
    assert_eq!(v["energy"], -3);
    assert_eq!(v["nodes"], 5);
    assert_eq!(v["lp_solves"], 5);
}
