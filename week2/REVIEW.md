# Week 2 final review

Review date: 2026-09-09

Scope: the implementation was checked against
`docs/superpowers/specs/2026-09-09-week2-md-design.md` and
`docs/superpowers/plans/2026-09-09-week2-md.md`, with emphasis on force signs,
periodic boundaries, integration order, independent trajectory checks, test
quality, and measured README claims.

## Findings

1. **Fixed — strict Rust lint failure.** `cargo clippy --release --all-targets
   -- -D warnings` rejected a manual modulo-based evenness check. Commit
   `4c29fb5` replaces it with the standard `is_multiple_of` expression. The
   full lint and test commands pass after the change.

2. **Fixed — periodic cell-list duplicate/missing-pair risk.** Wrapped neighbour
   indices can repeat when an axis has only two cells. Commits `5d3558b` and
   `5fb0dd8` add the failing equality cases and deduplicate the nine wrapped
   neighbours while visiting each cell pair once. The cell and naive force
   arrays and energies agree for a perturbed lattice, a cross-boundary pair, a
   cutoff pair, and a two-cell-wide box.

3. **Fixed — stored energies must not validate themselves.** Commits `fd4db2a`
   and `153931f` make `md check` reconstruct the system from each saved frame
   and recompute both potential and kinetic energy. A corruption test confirms
   that changing a stored energy is rejected.

4. **Fixed — heating schedule endpoint and provenance.** Commits `423747f` and
   `5ad0656` test the start, midpoint, and final target; apply the ramp only to
   production; and record `ramp_to` in `run.json`.

5. **Not fixed — saved-frame potential energy is recomputed with the naive
   all-pairs path.** This is intentional: it keeps saved energies independent
   of the optimized force path and makes the checker a stronger correctness
   test. It runs only at sampling points, accounts for about 5% of the cell-list
   profile at `N=400`, and does not invalidate the measured speedup comparison
   because both benchmark paths use the same output contract.

## Verification result

- `cargo fmt --check`: pass
- `cargo clippy --release --all-targets -- -D warnings`: pass
- `cargo test --release`: 23 passed, 0 failed
- Pair-force derivative, Newton's third law, minimum-image, cutoff continuity,
  integrator, deterministic initialization, cell-list equality, heating, file
  contract, independent checker, and MP4 smoke tests are present.
- Timing claims trace to `benchmark-results.csv`; profile claims trace to the
  two Samply screenshots; heating claims trace to the published 400-particle,
  200-frame trajectory.

No unresolved correctness or unsupported-claim finding remains.
