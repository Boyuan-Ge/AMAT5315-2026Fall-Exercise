# Week 3: Monte Carlo Simulation of the Ising Model

This crate simulates the two-dimensional square-lattice Ising model with periodic boundaries, coupling `J = 1`, no external field, and temperatures in units where Boltzmann's constant is one.

## Physics and reproducibility

For a proposed single-spin flip, the local energy change is

```text
delta_E = 2 * s_i * sum(four neighbouring spins).
```

The random-site Metropolis update accepts the flip with probability `min(1, exp(-delta_E/T))`. One Metropolis sweep is exactly `L^2` proposals. The Wolff update grows an aligned cluster with bond probability `1-exp(-2/T)` and always flips it; one Wolff sweep performs cluster flips until at least `L^2` spins have been touched.

Every run constructs `StdRng` from the requested seed. A fixed seed gives the same random-number stream and therefore the same output file, allowing tests and physics results to be reproduced.

## Commands

From `week3/`:

```bash
cargo test --release

cargo run --release -- relax --l 64 --t 1.8 \
  --sweeps 2000 --measure 2000 --seed 2026

cargo run --release -- snapshots
cargo run --release -- sweep
cargo run --release -- analyze artifacts --blocks 50
cargo run --release -- plot artifacts --blocks 50

cargo run --release -- sweep --wolff
cargo run --release -- analyze artifacts-wolff --blocks 50
```

The complete Metropolis workflow is:

```bash
make reproduce
```

The complete cluster workflow is:

```bash
make wolff
```

Large raw runs are regenerated into `artifacts/` and `artifacts-wolff/` and are intentionally ignored by Git.

## Artifact contract

`run.json` stores `sizes`, `t_grid`, `eq_sweeps`, `meas_sweeps`, `meas_sweeps_critical`, `sample_every`, `seed`, and `algorithm`. Each line of `series.jsonl` stores `L`, `T`, `sweep`, signed magnetization `M`, and energy per site `E`.

The published ramp uses `L`, `T`, `sweep`, `m`, and `spins`, with one row-major spin character per lattice site.

## Pages

The public lattice viewer is:

<https://boyuan-ge.github.io/AMAT5315-2026Fall-Exercise/week3/>

Append `?T=1.8`, `?T=2.3`, or `?T=3.0` to jump to an ordered, critical, or disordered frame after the page has loaded the ramp.
