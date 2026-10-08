//! Week 5 acoustic solver: Enzyme derivatives and Treeverse replay.
use ndarray::{Array2, Array3, Axis};
use ndarray_npy::{read_npy, write_npy};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
    fn enzyme_step(
        prev: *const f64,
        cur: *const f64,
        speed: *const f64,
        damping: *const f64,
        source: *const f64,
        dt: f64,
        dx: f64,
        nx: usize,
        nz: usize,
        out: *mut f64,
    );
    fn enzyme_step_jvp(
        prev: *const f64,
        dprev: *const f64,
        cur: *const f64,
        dcur: *const f64,
        speed: *const f64,
        dspeed: *const f64,
        damping: *const f64,
        source: *const f64,
        dt: f64,
        dx: f64,
        nx: usize,
        nz: usize,
        out: *mut f64,
        dout: *mut f64,
    );
    fn enzyme_step_vjp(
        prev: *const f64,
        dprev: *mut f64,
        cur: *const f64,
        dcur: *mut f64,
        speed: *const f64,
        dspeed: *mut f64,
        damping: *const f64,
        source: *const f64,
        dt: f64,
        dx: f64,
        nx: usize,
        nz: usize,
        out: *mut f64,
        dout: *mut f64,
    );
}

#[derive(Deserialize)]
struct Experiment {
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    steps: usize,
    source_frequency: f64,
    source_peak_time: f64,
    source_amplitude: f64,
    sponge_width: f64,
    sponge_strength: f64,
    shots: Vec<[f64; 2]>,
    receivers: Vec<[usize; 2]>,
    background: Vec<Vec<f64>>,
    perturbation: Vec<Vec<f64>>,
}
impl Experiment {
    fn size(&self) -> usize {
        self.nx * self.nz
    }
    fn speed(&self) -> Vec<f64> {
        self.background.iter().flatten().copied().collect()
    }
    fn perturbation(&self) -> Vec<f64> {
        self.perturbation.iter().flatten().copied().collect()
    }
    fn damping(&self) -> Vec<f64> {
        let mut v = vec![0.0; self.size()];
        for z in 0..self.nz {
            for x in 0..self.nx {
                let distance = x.min(z).min(self.nx - 1 - x).min(self.nz - 1 - z) as f64;
                let q = (1.0 - distance / self.sponge_width).max(0.0);
                v[z * self.nx + x] = self.sponge_strength * q * q;
            }
        }
        v
    }
    fn source(&self, shot: usize, step: usize) -> Vec<f64> {
        let [sx, sz] = self.shots[shot];
        let theta = std::f64::consts::PI
            * self.source_frequency
            * (step as f64 * self.dt - self.source_peak_time);
        let pulse = self.source_amplitude * (1.0 - 2.0 * theta * theta) * (-theta * theta).exp();
        let mut v = vec![0.0; self.size()];
        for z in 1..self.nz - 1 {
            for x in 1..self.nx - 1 {
                let dx = x as f64 - sx;
                let dz = z as f64 - sz;
                v[z * self.nx + x] = pulse * (-0.5 * (dx * dx + dz * dz)).exp();
            }
        }
        v
    }
}

#[derive(Clone)]
struct State {
    prev: Vec<f64>,
    cur: Vec<f64>,
}
impl State {
    fn zero(n: usize) -> Self {
        Self {
            prev: vec![0.0; n],
            cur: vec![0.0; n],
        }
    }
}

fn advance(
    e: &Experiment,
    damping: &[f64],
    speed: &[f64],
    s: &State,
    shot: usize,
    step: usize,
) -> State {
    for values in [&s.prev, &s.cur] {
        assert_eq!(values.len(), e.size());
    }
    assert_eq!(speed.len(), e.size());
    assert_eq!(damping.len(), e.size());
    let source = e.source(shot, step);
    let mut next = vec![0.0; e.size()];
    unsafe {
        enzyme_step(
            s.prev.as_ptr(),
            s.cur.as_ptr(),
            speed.as_ptr(),
            damping.as_ptr(),
            source.as_ptr(),
            e.dt,
            e.dx,
            e.nx,
            e.nz,
            next.as_mut_ptr(),
        );
    }
    State {
        prev: s.cur.clone(),
        cur: next,
    }
}
fn advance_born(
    e: &Experiment,
    damping: &[f64],
    speed: &[f64],
    m: &[f64],
    s: &State,
    ds: &State,
    shot: usize,
    step: usize,
) -> (State, State) {
    for values in [&s.prev, &s.cur, &ds.prev, &ds.cur] {
        assert_eq!(values.len(), e.size());
    }
    assert_eq!(speed.len(), e.size());
    assert_eq!(damping.len(), e.size());
    assert_eq!(m.len(), e.size());
    let source = e.source(shot, step);
    let (mut next, mut dnext) = (vec![0.0; e.size()], vec![0.0; e.size()]);
    unsafe {
        enzyme_step_jvp(
            s.prev.as_ptr(),
            ds.prev.as_ptr(),
            s.cur.as_ptr(),
            ds.cur.as_ptr(),
            speed.as_ptr(),
            m.as_ptr(),
            damping.as_ptr(),
            source.as_ptr(),
            e.dt,
            e.dx,
            e.nx,
            e.nz,
            next.as_mut_ptr(),
            dnext.as_mut_ptr(),
        );
    }
    (
        State {
            prev: s.cur.clone(),
            cur: next,
        },
        State {
            prev: ds.cur.clone(),
            cur: dnext,
        },
    )
}
fn reverse_one(
    e: &Experiment,
    damping: &[f64],
    speed: &[f64],
    s: &State,
    shot: usize,
    step: usize,
    weights: &[f64],
    adj: State,
    image: &mut [f64],
    record: bool,
) -> (State, Option<Vec<f32>>) {
    for values in [&s.prev, &s.cur] {
        assert_eq!(values.len(), e.size());
    }
    assert_eq!(adj.prev.len(), e.size());
    assert_eq!(adj.cur.len(), e.size());
    assert_eq!(speed.len(), e.size());
    assert_eq!(damping.len(), e.size());
    assert_eq!(image.len(), e.size());
    assert_eq!(weights.len(), e.steps * e.receivers.len());
    let source = e.source(shot, step);
    let mut dprev = vec![0.0; e.size()];
    let mut dcur = adj.prev;
    let mut dspeed = vec![0.0; e.size()];
    let mut next = vec![0.0; e.size()];
    let mut dnext = adj.cur;
    for (r, [x, z]) in e.receivers.iter().enumerate() {
        dnext[z * e.nx + x] += weights[step * e.receivers.len() + r];
    }
    unsafe {
        enzyme_step_vjp(
            s.prev.as_ptr(),
            dprev.as_mut_ptr(),
            s.cur.as_ptr(),
            dcur.as_mut_ptr(),
            speed.as_ptr(),
            dspeed.as_mut_ptr(),
            damping.as_ptr(),
            source.as_ptr(),
            e.dt,
            e.dx,
            e.nx,
            e.nz,
            next.as_mut_ptr(),
            dnext.as_mut_ptr(),
        );
    }
    for (a, b) in image.iter_mut().zip(dspeed) {
        *a += b;
    }
    let frame = if record {
        Some(dcur.iter().map(|v| *v as f32).collect())
    } else {
        None
    };
    (
        State {
            prev: dprev,
            cur: dcur,
        },
        frame,
    )
}

#[derive(Serialize)]
struct Action {
    action: &'static str,
    step: usize,
    saved_states: usize,
}
struct Scheduler<'a> {
    e: &'a Experiment,
    damping: &'a [f64],
    speed: &'a [f64],
    shot: usize,
    weights: &'a [f64],
    image: &'a mut [f64],
    saved: BTreeMap<usize, State>,
    actions: Vec<Action>,
    calls: usize,
    grads: usize,
    peak: usize,
}
impl Scheduler<'_> {
    fn log(&mut self, action: &'static str, step: usize) {
        self.peak = self.peak.max(self.saved.len());
        self.actions.push(Action {
            action,
            step,
            saved_states: self.saved.len(),
        });
    }
    fn recurse(
        &mut self,
        mut adj: State,
        mut delta: usize,
        mut tau: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
    ) -> State {
        // Replay from the deepest live ancestor, not unnecessarily from s_0.
        let beta = if sigma > beta {
            *self.saved.range(..sigma).next_back().unwrap().0
        } else {
            beta
        };
        if sigma > beta {
            delta -= 1;
            let mut working = self.saved.get(&beta).unwrap().clone();
            self.log("restore", beta);
            for j in beta..sigma {
                working = advance(self.e, self.damping, self.speed, &working, self.shot, j);
                self.calls += 1;
                self.log("call", j);
            }
            self.saved.insert(sigma, working);
            self.log("store", sigma);
        }
        let mut kappa = mid(delta, tau, sigma, phi);
        while tau > 0 && kappa < phi {
            adj = self.recurse(adj, delta, tau, sigma, kappa, phi);
            tau -= 1;
            phi = kappa;
            kappa = mid(delta, tau, sigma, phi);
        }
        self.log("restore", sigma);
        let state = self.saved.get(&sigma).unwrap();
        adj = reverse_one(
            self.e,
            self.damping,
            self.speed,
            state,
            self.shot,
            sigma,
            self.weights,
            adj,
            self.image,
            false,
        )
        .0;
        self.grads += 1;
        self.log("grad", sigma);
        if sigma > beta {
            self.saved.remove(&sigma);
            self.log("fetch", sigma);
        }
        adj
    }
}
fn mid(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    let den = delta + tau;
    if den == 0 {
        return phi;
    }
    let mut k = (delta * sigma + tau * phi + den - 1) / den;
    if k >= phi && delta > 0 {
        k = (phi - 1).max(sigma + 1);
    }
    k
}
fn binomial_fit(n: usize, delta: usize) -> usize {
    for tau in 1..=n {
        let mut b = 1u128;
        for k in 1..=delta {
            b = b.saturating_mul((tau + k) as u128) / (k as u128);
        }
        if b >= n as u128 {
            return tau;
        }
    }
    n
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}
fn save3f64(
    path: &Path,
    shape: (usize, usize, usize),
    v: Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_npy(path, &Array3::from_shape_vec(shape, v)?)?;
    Ok(())
}
fn save3f32(
    path: &Path,
    shape: (usize, usize, usize),
    v: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_npy(path, &Array3::from_shape_vec(shape, v)?)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut smoke = [0.0; 3];
    unsafe {
        enzyme_cube(2.0, smoke.as_mut_ptr());
    }
    assert_eq!(smoke, [8.0, 12.0, 12.0]);
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|s| s == "--smoke") {
        println!("Enzyme cube: {smoke:?}");
        return Ok(());
    }
    let exp_path = arg(&args, "--experiment").ok_or("missing --experiment")?;
    let mode = arg(&args, "--mode").ok_or("missing --mode")?;
    let out = PathBuf::from(arg(&args, "--out").ok_or("missing --out")?);
    let every = arg(&args, "--every")
        .map(|x| x.parse::<usize>())
        .transpose()?;
    let storage = arg(&args, "--storage").unwrap_or("full".into());
    let checkpoints = arg(&args, "--checkpoints")
        .map(|x| x.parse::<usize>())
        .transpose()?;
    if every == Some(0) {
        return Err("--every must be positive".into());
    }
    if storage == "treeverse" && checkpoints.unwrap_or(0) == 0 {
        return Err("treeverse needs positive --checkpoints".into());
    }
    if !["forward", "born", "adjoint"].contains(&mode.as_str()) {
        return Err("invalid mode".into());
    }
    fs::create_dir_all(&out)?;
    let raw: Value = serde_json::from_slice(&fs::read(&exp_path)?)?;
    let e: Experiment = serde_json::from_value(raw.clone())?;
    assert_eq!(e.background.len(), e.nz);
    assert!(e.background.iter().all(|r| r.len() == e.nx));
    let speed = e.speed();
    let m = e.perturbation();
    assert_eq!(m.len(), e.size());
    assert!(e.receivers.iter().all(|[x, z]| *x < e.nx && *z < e.nz));
    let damping = e.damping();
    let mut metadata = raw;
    metadata.as_object_mut().unwrap().remove("background");
    metadata.as_object_mut().unwrap().remove("perturbation");
    let frames: Vec<usize> = every
        .map(|k| (0..e.steps).step_by(k).collect())
        .unwrap_or_default();
    let mut recording_steps = frames.clone();
    if mode == "adjoint" {
        recording_steps.reverse();
    }
    let run = json!({"experiment_file":exp_path,"experiment":metadata,
        "recording":every.map(|k|json!({"every":k,"steps":recording_steps,
            "times":recording_steps.iter().map(|n|*n as f64*e.dt).collect::<Vec<_>>()}))});
    fs::write(out.join("run.json"), serde_json::to_vec_pretty(&run)?)?;
    let receivers = e.receivers.len();
    let mut result = json!({"mode":mode,"nx":e.nx,"nz":e.nz,"dx":e.dx,"dt":e.dt,
        "steps":e.steps,"shots":e.shots.len(),"receivers":receivers});
    println!("shot\tmode\tdata L2 norm");
    if mode == "forward" || mode == "born" {
        let mut traces = Vec::with_capacity(e.shots.len() * e.steps * receivers);
        let mut wavefield = Vec::new();
        let mut echo = Vec::new();
        for shot in 0..e.shots.len() {
            let mut s = State::zero(e.size());
            let mut ds = State::zero(e.size());
            let pert_speed: Vec<f64> = speed.iter().zip(&m).map(|(a, b)| a + b).collect();
            let mut echo_state = State::zero(e.size());
            let start = traces.len();
            for step in 0..e.steps {
                if mode == "born" {
                    (s, ds) = advance_born(&e, &damping, &speed, &m, &s, &ds, shot, step);
                } else {
                    s = advance(&e, &damping, &speed, &s, shot, step);
                    if shot == 0 && every.is_some() {
                        echo_state = advance(&e, &damping, &pert_speed, &echo_state, shot, step);
                    }
                }
                let sampled = if mode == "born" { &ds.cur } else { &s.cur };
                for [x, z] in &e.receivers {
                    traces.push(sampled[z * e.nx + x]);
                }
                if shot == 0 && frames.binary_search(&step).is_ok() && mode == "forward" {
                    wavefield.extend(s.cur.iter().map(|x| *x as f32));
                    echo.extend(
                        echo_state
                            .cur
                            .iter()
                            .zip(&s.cur)
                            .map(|(a, b)| (a - b) as f32),
                    );
                }
            }
            let norm = traces[start..].iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("{shot}\t{mode}\t{norm:.9}");
        }
        let name = if mode == "born" {
            "born_data.npy"
        } else {
            "traces.npy"
        };
        save3f64(&out.join(name), (e.shots.len(), e.steps, receivers), traces)?;
        if mode == "forward" && every.is_some() {
            save3f32(
                &out.join("wavefield.npy"),
                (frames.len(), e.nz, e.nx),
                wavefield,
            )?;
            save3f32(&out.join("echo.npy"), (frames.len(), e.nz, e.nx), echo)?;
        }
    } else {
        let data_path = arg(&args, "--data").ok_or("adjoint requires --data")?;
        let data: Array3<f64> = read_npy(data_path)?;
        if data.dim() != (e.shots.len(), e.steps, receivers) {
            return Err("data shape mismatch".into());
        }
        if storage == "full" && e.size() * e.steps * 16 > 1_000_000_000 {
            return Err("full history too large; use treeverse".into());
        }
        let mut image = vec![0.0; e.size()];
        let mut recordings = Vec::<(usize, Vec<f32>)>::new();
        let (mut per_shot, mut calls, mut reverse_calls, mut peak) = (Vec::new(), 0, 0, 0);
        for shot in 0..e.shots.len() {
            let shot_data = data
                .index_axis(Axis(0), shot)
                .to_owned()
                .into_raw_vec_and_offset()
                .0;
            let norm = shot_data.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("{shot}\t{mode}\t{norm:.9}");
            let (shot_calls, shot_peak, actions_file) = if storage == "treeverse" {
                let budget = checkpoints.unwrap();
                let mut scheduler = Scheduler {
                    e: &e,
                    damping: &damping,
                    speed: &speed,
                    shot,
                    weights: &shot_data,
                    image: &mut image,
                    saved: BTreeMap::from([(0, State::zero(e.size()))]),
                    actions: Vec::new(),
                    calls: 0,
                    grads: 0,
                    peak: 1,
                };
                let tau = binomial_fit(e.steps, budget);
                scheduler.recurse(State::zero(e.size()), budget, tau, 0, 0, e.steps);
                assert_eq!(scheduler.grads, e.steps);
                assert!(scheduler.peak <= budget + 1);
                assert_eq!(scheduler.saved.len(), 1);
                let file = format!("actions-{shot}.json");
                fs::write(out.join(&file), serde_json::to_vec(&scheduler.actions)?)?;
                (scheduler.calls, scheduler.peak, Some(file))
            } else {
                let mut states = Vec::with_capacity(e.steps + 1);
                states.push(State::zero(e.size()));
                for step in 0..e.steps {
                    let next = advance(&e, &damping, &speed, states.last().unwrap(), shot, step);
                    states.push(next);
                }
                let mut adj = State::zero(e.size());
                for step in (0..e.steps).rev() {
                    let (next, frame) = reverse_one(
                        &e,
                        &damping,
                        &speed,
                        &states[step],
                        shot,
                        step,
                        &shot_data,
                        adj,
                        &mut image,
                        shot == 0 && frames.binary_search(&step).is_ok(),
                    );
                    adj = next;
                    if let Some(frame) = frame {
                        recordings.push((step, frame));
                    }
                }
                (e.steps, e.steps + 1, None)
            };
            calls += shot_calls;
            reverse_calls += e.steps;
            peak = peak.max(shot_peak);
            let mut p = json!({"reverse_calls":e.steps,"scheduler_forward_calls":shot_calls,
                "peak_saved_states":shot_peak});
            if let Some(file) = actions_file {
                p["actions_file"] = json!(file);
            }
            per_shot.push(p);
        }
        write_npy(
            out.join("image.npy"),
            &Array2::from_shape_vec((e.nz, e.nx), image)?,
        )?;
        if every.is_some() && storage == "full" {
            recordings.sort_by(|a, b| b.0.cmp(&a.0));
            let rec: Vec<f32> = recordings.into_iter().flat_map(|(_, v)| v).collect();
            save3f32(&out.join("wavefield.npy"), (frames.len(), e.nz, e.nx), rec)?;
        }
        result["statistics"] = json!({"storage":storage,
            "checkpoints":if storage=="treeverse" {json!(checkpoints)} else {Value::Null},
            "reverse_calls":reverse_calls,"scheduler_forward_calls":calls,
            "peak_saved_states":peak,"peak_saved_bytes":peak*2*e.size()*8,"per_shot":per_shot});
    }
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
