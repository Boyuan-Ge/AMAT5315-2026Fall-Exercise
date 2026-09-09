# Week 2 Molecular Dynamics Design

## Goal

Extend `week2/md/` into a reproducible two-dimensional Lennard-Jones
molecular-dynamics program. The crate must demonstrate integrator accuracy,
generate and independently check an equilibrium trajectory, compare naive and
cell-list force calculations, render the required figures and videos, and
publish a 400-particle heating trajectory through GitHub Pages.

The optional Euler-detonation, time-reversal, and structure-factor extensions
are outside scope.

## Architecture

The crate remains one library plus one CLI binary. The library is divided into
small modules with these responsibilities:

- `system`: owns particle positions, velocities, forces, mass, boundary mode,
  box dimensions, and cutoff settings.
- `potential`: evaluates the plain Lennard-Jones pair energy and radial force,
  plus the shifted-cutoff energy used in periodic simulations.
- `force`: evaluates all particle forces and potential energy through either a
  naive all-pairs search or a cell list.
- `integrator`: defines an `Integrator` trait implemented by forward Euler and
  velocity-Verlet. A shared simulation driver accepts either implementation.
- `initialise`: creates square-count triangular lattices, deterministic
  Gaussian velocities, removes centre-of-mass velocity, and rescales kinetic
  temperature.
- `trajectory`: defines the JSON metadata and frame schemas and performs
  streaming reads and writes.
- `analysis`: recomputes energy drift, speed temperature, the 24-bin
  Maxwell-Boltzmann statistic, and radial distribution data from saved frames.
- `render`: renders paired particle/`g(r)` frames and calls `ffmpeg` to encode
  an MP4.
- `cli`: parses and validates the `run`, `check`, and `video` commands.

`System` is the sole owner of the main particle arrays. Analysis receives
shared borrows because it only reads states. Integrator steps receive mutable
borrows because they update positions, velocities, and forces.

## Physics Model

All quantities use reduced units with particle mass, Lennard-Jones epsilon,
and sigma equal to one. The plain pair potential and radial force are

`U(r) = 4 (r^-12 - r^-6)` and
`F(r) = 24/r (2 r^-12 - r^-6)`.

The isolated dimer uses open boundaries, no cutoff, positions `(0, 0)` and
`(1.2, 0)`, zero velocities, `dt = 0.01`, and mass one. Euler and
velocity-Verlet must run through the same `Integrator` interface. Over 500
steps, Verlet's maximum relative energy error must be below `1e-3`, while
Euler's final relative error must exceed `0.5`. A second Verlet run covers
5,000 steps for the long-time plot.

The fluid uses an even, square particle count. For `m = sqrt(N)`, spacing
`a = sqrt(2 / (sqrt(3) rho))`, row height `h = sqrt(3) a / 2`, and box
`[m a, m h]`. Odd rows are offset by `a/2`. Positions wrap into the box, and
pair displacements use `d - L round(d/L)` independently on each axis.

The periodic force uses cutoff `r_c = 2.5`. For `r < r_c`, its energy is
`U(r) - U(r_c)` and its force is the plain Lennard-Jones force. For
`r >= r_c`, both are zero. Equal and opposite pair forces are accumulated so
the total internal force vanishes within rounding tolerance.

Initial velocity components are deterministic Gaussian samples with variance
equal to the requested temperature. The seed defaults to 2026. The
centre-of-mass velocity is removed before an initial rescale. During 2,000
default equilibration steps, velocities are rescaled every 50 steps using
`T_thermo = 2 E_kin / (2N - 2)`. The unheated production run has no thermostat.
It runs 10,000 steps and saves steps 50, 100, ..., 10,000, for exactly 200
frames. A heating run linearly raises the target from `--temperature` at
production step zero to `--ramp-to` at the last production step and rescales
every 50 production steps.

## Command-Line Contract

The executable provides:

```text
md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01]
       [--eq-steps 2000] [--steps 10000] [--sample-every 50]
       [--seed 2026] [--force cells|naive] [--ramp-to VALUE]
       --out DIRECTORY
md check DIRECTORY
md video DIRECTORY --out FILE.mp4
```

`velocity-verlet` is the production integrator and `cells` becomes the default
force path after the optimization is verified. Invalid numeric parameters,
unsupported particle counts, malformed files, and failed physics checks return
a nonzero exit status with a clear message.

`run.json` contains `n`, `rho`, `box` as `[Lx, Ly]`, `dt`, `temperature`,
`eq_steps`, `steps`, `sample_every`, `seed`, `integrator` equal to
`"velocity-verlet"`, `force`, and optional `ramp_to`.

`traj.jsonl` contains one object per saved production frame with `step`, `t`,
wrapped `pos`, `vel`, `E_pot`, and `E_kin`. Generated trajectories, ordinary
build products, and the default `week2/artifacts/` directory remain outside
Git. The final heating `docs/run.json` and `docs/traj.jsonl` are deliberate
published exceptions.

## Independent Physics Check

`md check` treats positions and velocities as authoritative. It recomputes
kinetic and shifted-cutoff potential energy for every frame and only uses the
stored energies as consistency checks.

With `E0` from the first saved frame and
`k = max(1, floor(frame_count / 10))`, the checker reports:

- secular drift
  `abs(mean(last k energies) - mean(first k energies)) / abs(E0) < 2e-3`;
- speed temperature `abs(T_speed - 0.5) < 0.05`, where
  `T_speed = mean(v^2) / 2`;
- speed-shape `chi2 / 22 < 2` using 24 equal predicted-probability bins at
  the measured `T_speed`.

It prints every measurement beside its limit followed by `PASS` or `FAIL`.

The radial distribution function counts minimum-image neighbour distances up
to half the shorter box side, normalises by the uniform-density ring count, and
averages over particles and selected frames.

## Force Search and Performance

The naive path visits each unordered pair exactly once. The cell-list path uses
`nx = floor(Lx / r_c)` and `ny = floor(Ly / r_c)`, with cell widths
`Lx/nx` and `Ly/ny`. It searches the home cell and eight wrapped neighbour
cells. Wrapped cell indices are deduplicated before pair generation, and each
particle pair is still visited only once.

Tests compare naive and cell-list energies and force arrays on perturbed
lattices, including a pair crossing a periodic boundary, a pair at the cutoff,
and a box only two cells wide. Agreement uses an explicit floating-point
rounding tolerance.

Performance evidence consists of:

- three timings each for the supplied NumPy baseline, debug Rust, and release
  Rust contract run;
- before/after sampling-profiler screenshots and elapsed times at `N = 400`;
- three timings for naive and cell-list runs at `N = 100`, `400`, and `1600`,
  using 100 equilibration and 500 production steps;
- a labelled seconds-per-step scaling plot and honestly computed speedups.

No speedup is claimed unless the measurements show it. The release median must
be below one third of the debug median; cell-list time must beat naive time;
and the speedup must rise with `N` and exceed two at `N = 1600`.

## Rendering and Published Evidence

The crate examples reproducibly generate `week2/field.png` and
`week2/dimer.png`. The benchmark tooling generates `week2/scaling.png`.
`md video` renders one visual frame per saved trajectory frame with particles
beside `g(r)`, then invokes `ffmpeg` with settings that keep each required MP4
under 2 MB.

`week2/Makefile` provides `make reproduce`, which runs the release crate with
default flags and writes `week2/artifacts/`. `week2/README.md` records every
reproduction command, Timing, Profile, Benchmark, and Pages sections, and the
final recording link.

`docs/index.html` is the supplied viewer. The published heating data use
`N = 400`, start temperature `0.2`, final temperature `1.2`, 20,000 production
steps, and sampling every 100 steps, producing 200 frames. The page must load
without authentication. Long-range `g(r)` contrast must decrease during the
ramp. Separate `cold.mp4` and `hot.mp4` runs show persistent distant peaks at
`T = 0.2` and a tail approaching one after the first shell at `T = 1.0`.

## Test and Commit Strategy

Development follows red-green-refactor with small commits. Required automated
tests cover:

1. the existing greeting, pair-energy minimum, and analytic-force derivative;
2. both integrators applied to the same dimer setup and their energy bounds;
3. zero total internal force and shifted-potential continuity just inside the
   cutoff;
4. lattice geometry, wrapping, deterministic velocity preparation, and frame
   sampling count;
5. a CLI integration run that produces readable metadata and trajectory files;
6. the contract trajectory passing all three independently recomputed checks;
7. cell-list/naive equality in the required boundary and small-box cases;
8. the linear heating schedule and recorded `ramp_to` value.

Each new behavior is first committed with a failing test, then committed with
the minimal passing implementation. Earlier tests must stay green.

## Final Verification

A fresh clone must pass release tests, `make reproduce`, and `md check`.
`week2/REVIEW.md` records each review finding as fixed with a commit hash or not
fixed with a concrete reason. GitHub must contain the required source, plans,
figures, profile screenshots, videos, public Pages data, and README evidence.
The final two-minute recording is attached to a GitHub release rather than
committed to Git, and its release URL is added to the README.
