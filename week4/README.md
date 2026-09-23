# Week 4: continuum fluid dynamics

This week solves periodic advection-diffusion on a line and then the two-dimensional incompressible vorticity equation. One Rust `Integrator` trait supplies Euler, explicit midpoint (`rk2`), and classical RK4 to both problems. Fourier derivatives and the streamfunction recover a divergence-free velocity; the fluid solver retains only modes with `|kx|, |ky| <= floor(n/3)` in the vorticity and the nonlinear product.

## Install and regenerate from a clean clone

Requirements: Rust/Cargo, Python 3, and Python `venv`. From the repository root:

```bash
cd week4
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cargo test --locked
cargo build --release --locked
cargo install --path . --locked
bash scripts/run_line.sh
target/release/derivatives
.venv/bin/python scripts/run_flows.py
.venv/bin/python scripts/plot_flows.py
```

The scripts run from `week4/` and use the compiled binaries in `target/release/`. The installation command also places `field`, `fluid`, `line`, and `derivatives` on Cargo's executable path. All intermediate `artifacts/` and build/virtual-environment files stay untracked. Retain `artifacts/` locally if you want to redraw the figures without rerunning the solver. The committed design contracts are [field.design.toml](field.design.toml) and [fluid.design.toml](fluid.design.toml).

## The two field-to-fluid pipelines

The Taylor-Green field has the exact velocity `u = cos(x) sin(y) exp(-2 nu t)`, `v = -sin(x) cos(y) exp(-2 nu t)`. Its initial state and the seeded random state are passed as one JSON object from `field` to `fluid`:

```bash
mkdir -p artifacts
target/release/field taylor-green --n 64 | target/release/fluid \
  --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
  --out artifacts/taylor-green > artifacts/taylor-green.tsv

target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 | target/release/fluid \
  --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
  --out artifacts/random > artifacts/random.tsv
```

Each run writes `run.json` and six-decimal `fields.jsonl` inside its output folder. The TSV records time, energy, and enstrophy. The `--every` interval is rounded to an integer number of full steps; the solver never shortens a step for a snapshot. It records the first non-finite energy and exits with code 1. The scripted stability scan intentionally includes such failed runs.

For the exact comparison, `scripts/run_flows.py` also calls `field taylor-green --n 64 --nu 0.1 --t 1` and saves the answer as `artifacts/taylor-green/exact-t1.json`.

## Results and checks

| Check | Result |
|---|---:|
| Fourier derivative of `sin(3x) cos(2y)` on 32 × 32 | maximum error below `2e-13` for `dx`, `dxx`, `dxdy`, and Laplacian |
| Centred differences from 32 to 64 | error ratios `3.91` to `3.97`, near the expected `4` |
| Taylor-Green at `t=1`, `nu=0.1` | `E=0.167580`, `Z=0.335160`; relative velocity error `7.04e-7` |
| Random flow, seed 2026, `t=0` to `10` | `E: 0.500000 -> 0.277588`, `Z: 6.634685 -> 1.048184` |
| Taylor-Green RK4 stability | `dt=0.032` reaches `t=8`; `dt=0.033` becomes non-finite at `t=7.821`; predicted limit `0.0316` |
| Random RK4 stability | initial component maximum `2.05686`, advective bound `0.02316`; `dt=0.032` survives and `dt=0.034` fails |
| Random Euler at `dt=0.01` | non-finite by `t=1.08` |
| RK4 order on Taylor-Green | fitted slope `4.104` |
| Random time-step refinement at `t=2` | fitted slope `4.027`; chosen `dt=0.0125`, predicted error `3.10e-6`, measured `3.04e-6` |

The random phase sequence is deterministic for seed 2026 and independent of grid size. Rust's seeded random generator gives different phases from the learning sheet's example, so its detailed energy and stability numbers differ while the required convergence and stability relationships hold. The random perturbation grows by about 89 times by `t=20`; the Taylor-Green perturbation decays to the six-decimal storage floor.

## Evidence regeneration index

Run the commands in the installation section in order. These scripts produce every file committed in `evidence/`:

| Evidence | Producing command | What it demonstrates |
|---|---|---|
| [line-stability.png](evidence/line-stability.png) | `bash scripts/run_line.sh` (calls `scripts/plot_line.py`) | measured RK4 growth map and pulse below/above the limit |
| [line-accuracy.png](evidence/line-accuracy.png) | `bash scripts/run_line.sh` (calls `scripts/plot_line.py`) | Fourier versus centred differences; Euler, midpoint, RK4, and equal-weight slopes |
| [taylor-green.png](evidence/taylor-green.png) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | exact field comparison and shared-scale vorticity |
| [blowup.png](evidence/blowup.png) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | diffusive and advective instability brackets, with Euler |
| [sensitivity.png](evidence/sensitivity.png) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | two physically perturbed solution pairs |
| [random.png](evidence/random.png) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | random vorticity at `t=0,2,5,10` |
| [order.png](evidence/order.png) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | RK4 order at `dt=0.4,0.25,0.2` |
| [convergence.json](evidence/convergence.json) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | errors against the `dt=0.0025` reference and Richardson choice |
| [convergence.png](evidence/convergence.png) | `.venv/bin/python scripts/run_flows.py` then `.venv/bin/python scripts/plot_flows.py` | random-flow time-step slope and selected step |

`scripts/run_flows.py` writes the named runs under `artifacts/`: the baseline cases, stability scans, perturbation pairs, Taylor-Green order series, and random-flow convergence series. `scripts/plot_flows.py` reads those retained fields and TSVs. `target/release/derivatives` prints the full-precision derivative comparison directly in the terminal.
