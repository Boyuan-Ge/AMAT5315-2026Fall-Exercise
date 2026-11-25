# Week 12 — Branch-and-bound for spin glasses

An LP relaxation gives a lower bound, but rounding its solution does not generally recover the optimum. Exact branch-and-bound uses those bounds to prune. A neural model trained to imitate strong branching changes only the split decisions: the LP still controls pruning and the final optima remain exact.

The runs below use the learning sheet's SplitMix64 generator, single-threaded HiGHS with presolve off, and JAX on CPU. Times are measurements from one WSL Ubuntu run, not acceptance targets for other machines.

## 1. Relaxation and rounding

With `x` denoting a binary spin bit and `z` a bond XOR, the four linear inequalities in [the solver design](branchopt.design.toml) enforce the XOR when the spin bits are integral. In the LP all variables are relaxed to the interval from 0 to 1. On the frustrated square the LP gives −4 with all four spin bits at 0.5; the integer optimum is −2, while rounding every half to 1 gives +2. The diet LP gives `x₁ = 89/28`, `x₂ = 8/7`, and minimum cost `323/14 ≈ 23.0714`. The opening `pred 0.6.0` runs give minima −2 (square) and −138 (100 spins).

For the 16 graphs with 22 spins, 52 bonds, and seed 2026, the mean LP bound is −52.00 and the mean integer optimum is −38.25: a gap of **13.75**. Each LP has 22 fractional spin bits. Rounding misses the optimum on all 16 graphs, by 34.25 energy units on average.

![LP and integer energies, integrality gap, and rounding excess across 16 instances](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week12-results/gap.png)

## 2. Exact branch-and-bound

The tetrahedron explores exactly five nodes, with bounds **−6, −4, −3, −3, −2**. The third node gives the incumbent −3. The last two nodes cannot beat it and are pruned. Across 48 graphs (22 spins, 52 bonds, seed 2026), both rules match the HiGHS MILP optimum on every instance. Strong branching cuts the mean node count from **50.875 to 16.0**, but it solves many more LPs to decide where to split.

| Policy | Mean nodes | Mean LP solves | Mean search time |
|---|---:|---:|---:|
| Most-fractional | 50.875 | 50.875 | 17.02 ms |
| Strong | 16.000 | 287.417 | 95.90 ms |

![Explored nodes and LP solves for most-fractional and strong branching](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week12-results/bnb.png)

## 3. Learning split decisions

Strong branching exported **276 training nodes from 36 graphs** and **84 validation nodes from 12 different graphs**. The [trainer design](train_branch.design.toml) specifies the shared 12→48→1 MLP, masked cross-entropy, and training-only normalization. Epoch 85 gave the selected checkpoint: training cross-entropy 1.6215, validation cross-entropy 1.7608. Rust and JAX agree on all six probability probes (maximum absolute error `3.89e-16`).

On 16 unseen 22-spin graphs (seed 3026), neural branching visited **15.625 nodes** on average versus **50.875** for most-fractional, with no extra LP solves for split selection. The largest reduction was on instance 4: **157 → 23 nodes**, with both policies finding energy −38. The first two split spins changed from 0 and 1 to 4 and 21.

![Training and validation cross-entropy and per-instance node counts on 16 held-out graphs](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week12-results/neural_bnb.png)

## 4. Transfer and timing

The same model, without retraining, was applied to eight 30-spin, 72-bond graphs (seed 4026). Root-only neural branching scores once at the root and combines those scores with the current fractional distance at each later split. All four policies match the MILP references at both sizes.

| Spins | Policy | Mean nodes | Mean total ms | Mean LP ms | Mean inference ms |
|---:|---|---:|---:|---:|---:|
| 22 | Most-fractional | 50.875 | 17.983 | 17.954 | 0 |
| 22 | Strong | 15.625 | 101.687 | 101.652 | 0 |
| 22 | Neural | 15.625 | 5.683 | 5.492 | 0.180 |
| 22 | Root-only neural | 15.500 | 5.404 | 5.366 | 0.027 |
| 30 | Most-fractional | 177.750 | 86.755 | 86.611 | 0 |
| 30 | Strong | 38.000 | 445.840 | 445.710 | 0 |
| 30 | Neural | 42.000 | 21.138 | 20.500 | 0.600 |
| 30 | Root-only neural | 63.500 | 30.146 | 30.052 | 0.038 |

At 30 spins, both learned rules explore fewer nodes than most-fractional and are more than **5× faster than strong branching** (21.1× and 14.8× respectively on this run). Root-only inference is cheaper, but on these larger instances it visits more nodes than full neural inference, so its total time is higher.

![Mean node counts and LP, inference, and other search time for all four policies at both graph sizes](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week12-results/eval.png)

## Reproduce and verify

Use a fresh checkout on Linux or WSL with `pred 0.6.0`, Rust/Cargo, CMake, a C/C++ compiler, libclang, Python 3, and `uv` on `PATH`. The tested versions include Rust 1.98.1, Python 3.14.4, JAX 0.12.1, and Matplotlib 3.11.2. Cargo installs `branchopt` from the pinned lockfile. The script runs the Rust tests, then all four studies, regenerates the PNGs, and audits the results.

```sh
cd week12
bash scripts/reproduce.sh
```

To reuse an existing Matplotlib installation, set `PLOT_PYTHON` to its Python executable. To rerun just the audit after reproduction, run `python3 scripts/audit.py` from `week12/`. It checks the hand-verifiable cases, independently regenerates the seeded graph inputs, checks returned assignments and reference energies, verifies the split and Rust/JAX parity, and checks the performance comparisons and four PNGs.

Raw assignments, teacher labels, model weights, per-run logs, and the audit transcript are generated under the ignored `artifacts/` directory. Only the four result figures are attached to the [Week 12 results release](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/tag/week12-results).
