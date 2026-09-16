# Week 3: 2D Ising Monte Carlo

This directory is the complete Week 3 submission for the revised learning sheet. It implements a reproducible two-dimensional square-lattice Ising simulator with periodic boundaries, `J = 1`, zero external field, and both Metropolis and Wolff updates.

The command-line and file contract is copied exactly from [`ising.design.toml`](ising.design.toml). The public course viewer is <https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html>.

## Update and output contract

The program has one `ising` command. One Metropolis step is exactly `L^2` random-site proposals. One Wolff step is exactly one cluster flip, so its `cluster_size` field makes the computational work observable. A fixed `--seed` controls one random stream across the full temperature ramp.

Every run writes:

- `run.json`: `L`, update, temperature grid, discard/measurement counts, seed, sampling interval, and time unit;
- `series.jsonl`: one measurement per line with `L`, `T`, local step, signed `M`, and energy per site `E`; Wolff rows also include `cluster_size`;
- `spins.jsonl` when `--every` is positive: row-major integer spins and a step counter cumulative across the whole ramp.

## Clean setup and tests

Run these commands from `week3/`:

```bash
cargo test --release
cargo clippy --all-targets --all-features --release -- -D warnings
cargo install --path . --quiet

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Exact data-generation commands

Small physics checks and the energy-distribution comparison:

```bash
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 5000 --seed 2026 --out runs/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 --discard 2000 --measure 5000 --seed 2026 --out runs/T3.0
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 --discard 2000 --measure 5000 --seed 2026 --out runs/T3.1
.venv/bin/python scripts/boltzmann.py
```

The committed 410-frame heating ramp:

```bash
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --seed 2026 --every 20 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

Metropolis coarse and critical-window measurements:

```bash
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 2026 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 2026 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 2026 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 2026 --out artifacts/window-l64
```

Wolff critical-window measurements, with one cluster flip per step:

```bash
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 2026 --out artifacts/wolff-l32
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 2026 --out artifacts/wolff-l64
```

Generate and validate all numerical evidence:

```bash
.venv/bin/python scripts/plots.py
.venv/bin/python scripts/peaks.py
.venv/bin/python scripts/errors.py
.venv/bin/python scripts/bootstrap.py
.venv/bin/python scripts/compare.py
.venv/bin/python scripts/viewer_proofs.py
.venv/bin/python scripts/validate.py
```

Raw `runs/` and `artifacts/` data are reproducible and intentionally ignored by Git. The scripts, compact text summaries, plots, viewer proof PNGs, and `spins.jsonl` are committed. Every committed file is below 5 MB.

## Results and uncertainty

The ordered-phase check at `T=1.8` gave `mean |M| = 0.956908`, while the disordered `T=3.0` run gave `0.042756`. The energy-histogram log-ratio slope was `-0.00894067`; the Boltzmann prediction for `log[P_3.0(E)/P_3.1(E)]` is `-0.01075269`.

Metropolis five-point quadratic susceptibility fits gave:

```text
L=32: T_peak = 2.349542
L=64: T_peak = 2.282021
linear 1/L extrapolation: Tc = 2.214500
```

The extrapolated value is `0.054685` below the exact Onsager value `2.269185`. This is reported as a finite-sample estimate, not as an exact determination. At `L=64, T=2.3`, blocking by 2,000 sweeps increased the uncertainty of `mean |M|` by `30.56x` over the naive independent-sample estimate, demonstrating critical autocorrelation.

The required 500-replicate block bootstraps were run at block lengths 2,000, 4,000, and 8,000. Their peak estimates were stable within bootstrap uncertainty; the full values are in [`evidence/bootstrap.txt`](evidence/bootstrap.txt).

Metropolis and Wolff agree directly at `T=2.3`: the standardized difference is `d=0.409` for `L=32` and `d=-0.002` for `L=64`. Wolff autocorrelation times in [`evidence/tau-compare.png`](evidence/tau-compare.png) are converted to lattice-sweep work units by multiplying cluster-move time by `mean(cluster_size)/L^2`, so the comparison does not give a full-lattice update to each cluster flip for free.

## Viewer evidence

The public course viewer was checked against the raw GitHub recording: it loaded all 410 frames and reproduced the selected frame readouts. `scripts/viewer_proofs.py` composes the same required proof fields for `T=1.8`, `T=2.3`, and `T=3.0`: lattice, full history, frame number, temperature, cumulative sweep, magnetization, lattice size, source file, and capture time.

## Evidence index

- [`evidence/boltzmann.png`](evidence/boltzmann.png): energy distributions and Boltzmann slope.
- [`evidence/magnetization.png`](evidence/magnetization.png), [`evidence/susceptibility.png`](evidence/susceptibility.png), [`evidence/peaks.txt`](evidence/peaks.txt): thermodynamic curves and five-point fits.
- [`evidence/trace.png`](evidence/trace.png), [`evidence/acf-binning.png`](evidence/acf-binning.png), [`evidence/tau.png`](evidence/tau.png), [`evidence/errors.txt`](evidence/errors.txt): correlation diagnostics.
- [`evidence/chi-bootstrap.png`](evidence/chi-bootstrap.png), [`evidence/bootstrap.txt`](evidence/bootstrap.txt): required block-length stability test.
- [`evidence/magnetization-compare.png`](evidence/magnetization-compare.png), [`evidence/tau-compare.png`](evidence/tau-compare.png), [`evidence/comparison.txt`](evidence/comparison.txt): Metropolis/Wolff agreement and work-normalized efficiency.
- `evidence/viewer-T1.8.png`, `evidence/viewer-T2.3.png`, `evidence/viewer-T3.0.png`: stamped course-viewer proof frames.
