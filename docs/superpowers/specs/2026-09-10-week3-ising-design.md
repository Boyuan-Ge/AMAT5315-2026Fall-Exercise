# Week 3 Ising Monte Carlo Design

## Purpose

Build a tested Rust command-line program named `ising` in `week3/` that simulates the two-dimensional square-lattice Ising model, records reproducible measurements, analyzes correlated Monte Carlo data, compares Metropolis and Wolff updates, and publishes an interactive temperature-ramp visualization.

The implementation must satisfy the Week 3 learning sheet's artifact contracts and final fresh-clone checks. The work remains in the existing `AMAT5315-2026Fall-Exercise` repository.

## Scope

The required work covers all five parts of the Week 3 assignment:

1. A reproducible single-temperature Metropolis relaxation.
2. A recorded heating ramp and public lattice viewer.
3. A two-size temperature sweep, physical-observable plots, and a critical-temperature estimate.
4. Naive and correlation-aware error bars plus integrated autocorrelation times.
5. A Wolff cluster sampler and a direct autocorrelation comparison with Metropolis.

Optional extensions and the neural-sampler challenge are outside the required scope.

## Model and Simulation Conventions

- Model: two-dimensional Ising model on an `L x L` square lattice.
- Spins: each site is either `-1` or `+1`.
- Coupling: `J = 1`.
- External field: none.
- Boundary conditions: periodic in both directions.
- Energy: `E = -sum_<ij> s_i s_j`, with each nearest-neighbour bond counted once.
- Temperature units: `J/k_B`, with `k_B = 1`.
- Magnetization: `m = sum_i(s_i) / L^2`.
- Reported order parameter: the time average of `|m|`.
- Annealing order: ascending temperature, beginning from an all-up lattice and warm-starting each temperature from the previous final lattice.
- Reproducibility: every stochastic command accepts a seed and uses a single explicitly seeded `StdRng` from `rand = "0.9"`.

## Architecture

The crate will separate the physical model, update rules, protocols, artifact I/O, analysis, plotting, and CLI parsing:

- `src/lattice.rs`: lattice storage, periodic neighbours, total energy, energy per site, magnetization, spin serialization, and local flip energy.
- `src/metropolis.rs`: five-entry acceptance table and random-site Metropolis sweeps.
- `src/wolff.rs`: single-cluster growth and work-normalized Wolff sweeps.
- `src/protocol.rs`: relaxation, snapshots, Metropolis sweep, and Wolff sweep orchestration.
- `src/artifacts.rs`: fixed JSON and JSON Lines schemas plus streaming readers and writers.
- `src/analysis.rs`: grouped observables, susceptibility, fitted peak positions, critical-temperature extrapolation, naive and blocked errors, and integrated autocorrelation time.
- `src/plot.rs`: generation of magnetization, susceptibility, autocorrelation, and comparison PNG files.
- `src/cli.rs`: command-line definitions and validation.
- `src/lib.rs`: stable module exports used by the binary and integration tests.
- `src/main.rs`: thin command dispatcher with user-facing output.

The large measurement series will be streamed to disk and analyzed one `(L, T)` group at a time so the full 160 MB dataset is never required in memory at once. Plotting will use Rust libraries so a fresh clone needs only Cargo and Make.

## Command-Line Interface

### `relax`

Example:

```text
ising relax --l 64 --t 1.8 --sweeps 2000 --measure 2000 --seed 2026
```

The command starts from all-up spins, performs the equilibration sweeps, averages `|m|` during measurement sweeps, and prints one summary line containing `L`, `T`, equilibration sweeps, measurement sweeps, `mean_abs_m`, and the accepted-proposal fraction. It then prints the final lattice with one character per spin.

### `snapshots`

Default protocol:

- `L = 64`.
- Temperatures `1.5` through `3.5` inclusive in increments of `0.05`.
- `2000` discarded equilibration sweeps per temperature.
- `200` recording sweeps per temperature.
- One frame every `20` recording sweeps.
- Seed `2026`.
- Output `artifacts/spins.jsonl`.

All values are adjustable by command-line flags. The default output contains exactly 410 frames. Each JSON object has exactly the viewer contract fields:

- `L`: integer lattice side.
- `T`: floating-point temperature.
- `sweep`: cumulative sweep count across the ramp.
- `m`: signed magnetization.
- `spins`: row-major string of `L^2` characters, `1` for up and `0` for down.

### `sweep`

The default Metropolis protocol uses:

- Sizes `[32, 64]`.
- Temperatures `1.5` through `3.5` in increments of `0.1`, refined to `0.05` from `2.0` through `2.6`, for 27 temperatures total.
- `2000` discarded equilibration sweeps at each temperature.
- `5000` measurement sweeps outside the critical window.
- `100000` measurement sweeps at the 13 temperatures from `2.0` through `2.6`.
- One recorded row after every measurement sweep.
- Seeds `1042` for `L = 32` and `42` for `L = 64`.
- Output folder `artifacts/`.

The command exposes tunable sizes, seeds, grid, critical window, sweep counts, sample interval, and output folder while preserving these defaults.

With `--wolff`, the defaults change to:

- Sizes `[32, 64]`.
- Temperatures `2.0` through `2.6` inclusive in increments of `0.05`.
- `2000` discarded equilibration sweeps.
- `100000` measurement sweeps.
- Output folder `artifacts-wolff/`.
- Algorithm label `wolff`.

### `plot`

The command reads a saved run and produces:

- Mean absolute magnetization against temperature, including the infinite-lattice Onsager curve and exact critical-temperature marker.
- Susceptibility against temperature for both lattice sizes, including fitted peak markers.
- Integrated autocorrelation time against temperature on a logarithmic vertical axis after analysis is available.

Generated chart paths are printed and kept inside the selected artifact folder unless a specific output path is supplied.

### `analyze`

The command reads a saved run and prints one line per `(L, T)` group containing:

- Mean absolute magnetization.
- Naive standard error.
- Blocked standard error.
- Ratio of blocked to naive error.
- Integrated autocorrelation time of `|m|`.

The block count is adjustable and defaults to 50. The command also prints the susceptibility peak for each size, the extrapolated critical temperature, Onsager's exact value `2.26919`, relative deviation, and ordered-phase checks when the run contains the lowest temperature.

## Update Rules

### Random-Site Metropolis

One proposal selects a uniformly random lattice site. For spin `s_i` with four-neighbour sum `h_i`, the proposed energy change is:

```text
delta_E = 2 * s_i * h_i
```

The flip is accepted with probability `min(1, exp(-delta_E / T))`. Because `delta_E` can only be `-8`, `-4`, `0`, `4`, or `8`, the five acceptance probabilities are precomputed once per temperature. One sweep is exactly `L^2` proposals, accepted or rejected.

### Wolff Single-Cluster Update

A cluster begins at one uniformly selected seed site. Aligned neighbours are added with probability `p = 1 - exp(-2/T)`, with each site added at most once. The completed cluster is always flipped.

One Wolff sweep performs cluster flips until the total number of touched spins is at least `L^2`. This definition makes the reported sweep unit comparable to a Metropolis sweep.

## Artifact Contracts

### `run.json`

The settings file contains these fixed field names:

- `sizes`: `[32, 64]` for default runs.
- `t_grid`: ordered list of temperatures.
- `eq_sweeps`.
- `meas_sweeps`.
- `meas_sweeps_critical`.
- `sample_every`: `1` for contract runs.
- `seed`: integer base seed `42` for the default contract run. `L = 64` uses this seed directly and `L = 32` deterministically uses `seed + 1000 = 1042`.
- `algorithm`: exactly `"metropolis"` or `"wolff"`.

### `series.jsonl`

One JSON object is written after every measurement sweep, in size then ascending-temperature order. Each object contains exactly:

- `L`.
- `T`.
- `sweep`: measurement-sweep index at the current temperature.
- `M`: signed mean spin, rounded to six decimals.
- `E`: energy per site, rounded to six decimals.

The default Metropolis run must contain exactly `2,740,000` rows. The file is expected to be approximately 160 MB and must remain untracked.

## Statistical Analysis

For each `(L, T)` group:

- `mean_abs_m = mean(|M|)`.
- `chi = L^2 * (mean(M^2) - mean(|M|)^2) / T`.
- The susceptibility peak is the vertex of a parabola fitted through the five temperature points centred on the largest grid value.
- With fitted peaks for `L = 32` and `L = 64`, `T_c = 2*T_peak(64) - T_peak(32)`.
- Naive error is the sample standard deviation of `|M|` divided by the square root of the sample count.
- Blocked error is the standard error of 50 equal-length block means, dropping only a final incomplete remainder if necessary.
- Integrated autocorrelation time uses `tau_int = 1/2 + sum rho(t)` and a self-consistent truncation window ending when the lag reaches at least `6*tau_int`.

The autocorrelation implementation must avoid quadratic work for 100000-row series. It will use an FFT-based autocovariance or another numerically equivalent `O(n log n)` method, with tests against a direct estimator on short deterministic series.

## Plots and Public Viewer

The repository will track:

- `docs/week3/index.html`: supplied Week 3 viewer.
- `docs/week3/spins.jsonl`: the 410-frame default ramp.
- `week3/tau-compare.png`: Metropolis versus Wolff autocorrelation time at `L = 64` over their shared temperature window, with logarithmic vertical axis.
- `week3/README.md`: explanation, reproduction commands, key results, and public Pages address.

The viewer must load without authentication from the existing GitHub Pages site and support `?T=1.8`, `?T=2.3`, and `?T=3.0` query parameters.

## Repository and Reproduction Rules

- `week3/Makefile` provides `make reproduce`.
- `make reproduce` runs the default snapshots protocol, the default Metropolis sweep, the analysis, and required plots.
- `week3/artifacts/` and `week3/artifacts-wolff/` are ignored by Git.
- The compact ramp copied to `docs/week3/spins.jsonl` is tracked.
- Cargo dependencies are version-pinned through `Cargo.lock`.
- Re-running the same protocol with the same seed must reproduce byte-identical `series.jsonl` output and the same SHA-256 checksum.

## Testing Strategy

Development follows red-green-refactor cycles. A failing test is observed before each production behavior is implemented.

Required tests include:

- Periodic neighbour lookup at edges and corners.
- Total energy on hand-checkable lattices.
- Local `delta_E` equals the before/after difference of independently recomputed total energies across multiple random lattices, sites, and all five possible energy changes.
- Metropolis accepts all non-positive energy changes and applies the seeded uphill coin consistently.
- Same seed reproduces an entire printed relaxation result; a different seed changes it.
- Snapshot frames have the exact five keys, correct spin-string length, monotonic cumulative sweep numbers, and correct default count on a reduced test protocol.
- Sweep metadata and row objects use the exact artifact field names and six-decimal values.
- Temperature-grid construction includes exactly the required 27 Metropolis and 13 Wolff temperatures without floating-point duplicates.
- Direct and fast autocorrelation estimators agree on short known series.
- Independent samples give `tau_int` near `0.5`; correlated test data produces a larger value.
- Blocked and naive errors agree for deterministic independent test data within tolerance and diverge for correlated data.
- Susceptibility and five-point parabolic peak fitting recover known synthetic peaks.
- Wolff cluster growth never adds a site twice, flips the complete constructed cluster, is reproducible for a seed, and reports touched-spin-normalized sweeps.
- CLI integration tests cover help text, validation errors, output paths, and small inexpensive end-to-end runs.

The red tests for local energy and seeded reproducibility are committed before the first sampler implementation, matching the Week 3 evidence requirement.

## Validation Gates

The completed work must demonstrate all of the following:

1. `cargo test --release` passes in a fresh clone.
2. The `relax` result at `L=64`, `T=1.8`, seed `2026` reports `mean_abs_m` between `0.95` and `0.965` and acceptance near 4%.
3. The equivalent `T=3.0` result reports `mean_abs_m < 0.06` and acceptance near 46%.
4. Two identical seeded relaxation commands produce byte-identical output; seed `2027` differs while remaining physically plausible.
5. The ramp contains exactly 410 valid frames and the public viewer displays ordered, critical, and disordered domain regimes.
6. The default Metropolis series contains exactly `2,740,000` rows and reproduces byte-for-byte.
7. Both sizes have `mean_abs_m >= 0.9` at the lowest temperature.
8. The extrapolated Metropolis critical temperature differs from `2.26919` by less than 2%.
9. At `L=64`, `T=2.3`, the Metropolis blocked-to-naive error ratio is at least 10 and the integrated autocorrelation time is at least 5 sweeps.
10. Away from the transition, including `T=3.5`, the error ratio falls below 3.
11. The Wolff run at `L=64`, `T=2.3` has integrated autocorrelation time below 2 sweeps and error ratio below 2.
12. The Wolff critical-temperature estimate differs from `2.26919` by less than 2%.
13. `tau-compare.png` shows the Metropolis critical spike and a nearly flat Wolff curve over the shared window.
14. A private browser can load the Pages address with `?T=2.3` without authentication.

## Error Handling

- Reject `L < 2`, non-positive temperatures, zero sweep counts where a measurement is required, zero frame intervals, inconsistent temperature bounds, and output paths that cannot be created.
- Refuse analysis when required files or fields are missing, when group ordering is invalid, or when there are too few observations for a requested statistic.
- Write artifacts through buffered streaming writers and propagate I/O errors with paths and actionable context.
- Create parent output directories when safe; never delete unrelated files.
- Write complete replacement artifacts deterministically rather than appending to stale runs.

## Documentation and Evidence

`week3/README.md` will explain:

- The model and both update rules.
- Why a seed is necessary.
- The local energy-change formula.
- The critical-temperature estimate and its deviation from Onsager's value.
- The difference between naive and blocked errors at `T = 2.3`.
- Why Wolff reduces critical slowing down.
- Exact commands for tests, reproduction, analysis, the Wolff run, and the viewer.
- The GitHub Pages address.
- Code locations for the energy change, seeding, and update-rule selection.

The final evidence will be collected from a fresh clone and will include passing tests, the exact row count, both critical-temperature checks, the Wolff autocorrelation result, and public-page verification.
