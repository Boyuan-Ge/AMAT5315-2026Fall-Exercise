# Week 12 — Branch-and-bound for spin glasses

Work in progress. Parts 1 and 2 have been run locally; the learned branching and transfer experiments remain.

Build with Rust, CMake, a C/C++ compiler, and libclang. The LP and MILP use the pinned Rust HiGHS crate with one thread and presolve disabled. All random benchmarks below use the learning sheet's SplitMix64 instances and seed 2026.

```sh
cargo test --release --locked
cargo install --path . --locked --force
branchopt --mode gap --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026 --out artifacts/bnb_most
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong --seed 2026 --out artifacts/bnb_strong
python scripts/plot_gap.py
python scripts/plot_bnb.py
```

The figure scripts need Matplotlib. Run from this directory and use fresh output directories for solver runs.

The standalone pred 0.6.0 checks give -2 (square) and -138 (100 spins). The gap benchmark has a mean LP gap of 13.75. The exact branch-and-bound solvers agree with HiGHS on every instance. Most-frac explores 50.875 nodes on average, versus 16.0 for strong branching, which pays for extra LP probes. The tetrahedron trace has exactly five explored nodes and optimum -3.

![LP lower bounds, exact optima and rounding excess](artifacts/gap.png)

![Sorted explored nodes and LP solves](artifacts/bnb.png)

Raw outputs and figures are kept in the ignored local artifacts directory.
