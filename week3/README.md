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
cargo run --release -- compare-tau artifacts artifacts-wolff \
  --l 64 --output tau-compare.png --blocks 50
```

The complete Metropolis workflow is:

```bash
make reproduce
```

The complete cluster workflow is:

```bash
make wolff
make compare
```

Large raw runs are regenerated into `artifacts/` and `artifacts-wolff/` and are intentionally ignored by Git.

## Artifact contract

`run.json` stores `sizes`, `t_grid`, `eq_sweeps`, `meas_sweeps`, `meas_sweeps_critical`, `sample_every`, `seed`, and `algorithm`. Each line of `series.jsonl` stores `L`, `T`, `sweep`, signed magnetization `M`, and energy per site `E`.

The published ramp uses `L`, `T`, `sweep`, `m`, and `spins`, with one row-major spin character per lattice site.

## Measured results

The default Metropolis run produced 2,740,000 rows (153 MB) in 1 minute 48 seconds. Its ordered-phase values at `T=1.5` are `0.986641` for `L=32` and `0.986498` for `L=64`, both above the required `0.9`. The susceptibility peaks are:

```text
L=32: T_peak = 2.3497
L=64: T_peak = 2.3146
T_c = 2.2795
```

The extrapolated critical temperature differs from Onsager's `2.26919` by `0.45%`, inside the required `2%` window.

At `L=64, T=2.3`, Metropolis gives:

```text
mean_abs_m = 0.443748
naive_error = 0.000645
blocked_error = 0.021010
blocked/naive ratio = 32.59
tau_int = 983.76 sweeps
```

At `T=3.5`, the ratio falls to `2.22` and `tau_int` to `2.44` sweeps. This shows that the naive error bar fails specifically around the critical transition.

The Wolff window run produced 2,600,000 rows (146 MB) in 4 minutes 10 seconds. Its fitted peaks are `2.3924` for `L=32` and `2.3438` for `L=64`, giving `T_c=2.2952`, a `1.15%` deviation from Onsager. At `L=64, T=2.3`, it gives:

```text
mean_abs_m = 0.524456
blocked/naive ratio = 1.45
tau_int = 0.86 sweeps
```

The cluster update reduces the measured transition autocorrelation time from `983.76` to `0.86` sweeps, approximately a 1,144-fold reduction.

The byte-reproducible Metropolis series has SHA-256:

```text
8b05e4b1678d63f7460764016f9767839871a4fdbd7f5811291c4557094a6f08
```

## Code evidence

- Local flip energy: `src/lattice.rs:97`.
- Seeded generators: `src/protocol.rs:153`, `src/protocol.rs:243`, and `src/protocol.rs:324`.
- Metropolis/Wolff update selection: `src/protocol.rs:384`.
- The two-update comparison chart is `tau-compare.png`; regenerate it with `make compare`.

## Pages

The public lattice viewer is:

<https://boyuan-ge.github.io/AMAT5315-2026Fall-Exercise/week3/>

Append `?T=1.8`, `?T=2.3`, or `?T=3.0` to jump to an ordered, critical, or disordered frame after the page has loaded the ramp.
