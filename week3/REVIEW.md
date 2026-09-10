# Week 3 Code Review

## Scope

Reviewed the complete Week 3 implementation against:

- `docs/superpowers/specs/2026-09-10-week3-ising-design.md`
- `docs/superpowers/plans/2026-09-10-week3-ising.md`
- The Week 3 artifact contracts and numerical gates

The review covered lattice physics, both update rules, deterministic random-number use, streaming JSON output, statistical formulas, plotting, CLI validation, generated-artifact tracking, and the reproduction workflow.

## Findings

### Resolved: per-size seeds were not independently tunable

The default base seed already produced the required `1042` and `42` streams, but the CLI initially exposed only `--seed`. The assignment asks for seeds to be tunable for multiple lattice sizes. A failing test was added before the fix. `SweepConfig::resolved_seeds` and `--seeds` now accept an explicit comma-separated seed for every requested size, reject length mismatches, and retain the required default mapping.

### Accepted internal invariants

The remaining `panic!`/`expect` sites are limited to impossible states produced by internal fixed Ising values, nonempty ranges, and compile-time protocol grids. User-controlled inputs return contextual errors instead of panicking.

### Artifact safety

`week3/artifacts/`, `week3/artifacts-wolff/`, and `week3/target/` are ignored. The 153 MB and 146 MB raw series are not staged. Only the 410-frame public ramp and the 88 KB comparison chart are tracked.

## Verification before merge

- `cargo fmt --check`: passed.
- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `cargo test --release`: 22 passed, 0 failed.
- Metropolis series: 2,740,000 rows; byte-identical SHA-256 on two full runs.
- Metropolis critical temperature: `2.2795`, `0.45%` from Onsager.
- Metropolis at `L=64,T=2.3`: error ratio `32.59`, `tau_int=983.76`.
- Wolff series: 2,600,000 rows.
- Wolff critical temperature: `2.2952`, `1.15%` from Onsager.
- Wolff at `L=64,T=2.3`: error ratio `1.45`, `tau_int=0.86`.
- All five generated charts were visually inspected for labels, clipping, axis scaling, and expected curve ordering.

## Verification after local integration

- The feature branch was fast-forwarded into local `main` at `eb5c564`.
- A clean, non-local clone of that commit passed all 22 release-mode tests.
- `make reproduce` regenerated 410 spin frames and exactly 2,740,000 Metropolis rows.
- The regenerated Metropolis SHA-256 was `8b05e4b1678d63f7460764016f9767839871a4fdbd7f5811291c4557094a6f08`, identical to the earlier full runs.
- `make wolff` regenerated exactly 2,600,000 Wolff rows with SHA-256 `9689c3e256a1f3a29a0e5ef4081273403d5d5e92643491e62d8a81f2149adf85`.
- `make compare` regenerated a byte-identical `tau-compare.png`.
- The tracked public viewer contains the same 410-frame `spins.jsonl` as the clean reproduction.

Public GitHub Pages verification remains pending until the user approves the upload.
