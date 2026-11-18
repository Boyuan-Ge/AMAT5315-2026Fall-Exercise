# Week 10 - Approximate tensor contraction for the Ising model

Boyuan's implementation of all three hands-on parts: Rust SVD evidence, canonical boundary-MPS contraction of open squares, and TRG contraction of periodic squares. The calculation contracts Boltzmann weights; it does not sample spin configurations.

**Main result:** open-boundary MPS gives **Tc = 2.267506**, and periodic TRG gives **Tc = 2.269205**, compared with **2/log(1+sqrt(2)) = 2.269185314**. Their relative errors are **0.0740%** and **0.000855%**, respectively. All required tolerances pass: five Rust tests and 7,890 independent numerical checks.

The three figures are displayed directly below. The learning sheet requires generated tensors, data, figures and archives to remain out of Git. Therefore the PNGs, numerical summary and compact scan-table archive are [Release assets](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/tag/week10-results), not tracked files; the README embeds those images, rather than asking the reader to open image links. Full original evidence remains locally in `week10/artifacts/` and `week10/runs/` and can be regenerated from the committed source.

## Reproduction and environment

From the repository root, on Linux/WSL with Rustup and uv available:

```bash
cd week10
rustup toolchain install 1.96.0 --profile minimal
uv sync --frozen
bash scripts/run.sh
```

This installs the locked Rust crate, runs tests, runs the two prescribed CLI examples and SVD build, performs all coarse/refined scans, independently verifies the saved evidence, and makes the three figures. The runner resumes complete run directories without overwriting them. For a completely fresh experiment, clone into a new directory and run the same commands. The CLI refuses an already-existing output directory to protect previous evidence.

The stated individual cases are:

```bash
cargo install --locked --path . --force
thermo --method svd --out runs/svd-new
thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64-new
thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024-new
```

To independently verify already-generated **full local evidence** without rerunning the solver:

```bash
OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/evidence.py
.venv/bin/python scripts/audit.py
```

For the compact results, download `week10-tables.tar.gz` from the Release and run `tar -xzf week10-tables.tar.gz` **inside `week10/`**. It contains the coarse/refined g(K) tables, timings, small-run table, full numerical-check summary and the original raw-file SHA-256 inventory. This approximately 785 KiB archive is for inspecting results; it does **not** contain the roughly 1.1 GiB of full raw tensors, scales and factors. Independent tensor replay and hash verification require the retained local `runs/`, or regenerating it with `scripts/run.sh`. A complete raw-evidence archive is also retained locally, but is not uploaded. No generated archive or extracted data should be added to Git.

- Measured environment: Ubuntu on WSL2 x86_64, Intel Core i5-13600K (20 logical CPUs), Rust **1.96.0**, Python **3.14.4**, NumPy **2.5.3**, Matplotlib **3.11.2**. Dependencies are pinned in `Cargo.lock` and `uv.lock`.
- Rust linear algebra: **t4a-faer 0.33.0**, CPU float64, sequential thin SVD and matrix multiplication; no CUDA, autodiff or Monte Carlo. Up to six independent solver processes run simultaneously. Reported per-run times include computation and raw-evidence writes; summed times are not elapsed wall-clock times or isolated performance benchmarks.
- Data sources: no external dataset. The Gaussian matrix is generated in Rust with seed **2026**, **rand_chacha 0.3.1 ChaCha8Rng**, rand 0.8.8 f64 uniforms and a stated Box-Muller transform. Ising tensors are computed analytically from K.
- [thermo.design.toml](thermo.design.toml) is the requested specification. [src/main.rs](src/main.rs) implements learning-sheet Algorithms 1-5. [study.py](scripts/study.py) schedules Rust CLI runs; [evidence.py](scripts/evidence.py) independently reads and verifies the resulting binary tensors using NumPy; [audit.py](scripts/audit.py) checks hashes, grids, thresholds, image embeds and the source-only Git policy.
- TRG ports the two cuts and ring regrouping from the [specified BackwardsLinalg.jl example at b16273a](https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl). Its sample closure is replaced by the learning-sheet sum over **two independent opposite-leg pairs**, `sum_i,j A[i,j,i,j]`. The Rust version retains `min(chi, spectrum_length)` directions rather than introducing the Julia example's additional absolute singular-value cutoff.

## 1. SVD: accuracy does not necessarily save storage

![Singular spectrum and measured reconstruction errors with the discarded-spectrum prediction, storage break-even rank and 10 percent error threshold](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week10-results/svd.png)

The original **512 x 512** independent standard-normal matrix, all 512 singular values and reconstructed matrices at the five required ranks are saved in `runs/svd/`. Files are little-endian float64 in C order (last axis fastest); each shape and filename is in `run.json`. NumPy recomputes each error from the saved original and reconstruction, not from the Rust-reported spectrum alone.

| Retained rank | Measured relative Frobenius error | Discarded-spectrum prediction | Two-factor/dense storage |
|---:|---:|---:|---:|
| 16 | 0.941957406228 | 0.941957406228 | 0.0625 |
| 64 | 0.789202251044 | 0.789202251044 | 0.2500 |
| 128 | 0.613582583871 | 0.613582583871 | 0.5000 |
| 256 | 0.325165324143 | 0.325165324143 | 1.0000 |
| 512 | 7.86e-15 | 0 | 2.0000 |

All reconstruction errors agree with `sqrt(sum_discarded s² / sum_all s²)` within **1e-8**; full-rank error is **7.86e-15 < 1e-8**. NumPy also independently recomputes the singular spectrum. The 10% threshold is first reached at **rank 395**, requiring **1.54296875 times** the dense storage. At rank 256 storage is already equal to dense, while error is still 32.5%. Thus this matrix **does not admit a storage-saving truncated-SVD approximation at 10% error**.

For an m x n matrix, storing `A=U_chi diag(s)` and `B=V_chi^T` uses `(m+n)chi` entries rather than `mn`. For two n x n rank-chi matrices, computing `(AB)(CD)` as `[A(BC)]D` costs `2n chi²` multiplications while keeping the answer factored, instead of `n³` for ordinary dense multiplication; these counts exclude initial SVDs and output expansion. Grouping tensor legs is lossless; discarding singular values is the approximation. An inserted invertible gauge and its inverse cancel, so a gauge change preserves the represented matrix; the independent verification includes this check.

## 2. Open squares: boundary-MPS row absorption

![Open-square specific heat grows and its peak moves upwards with size; inverse-size fits approach the Onsager transition](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week10-results/boundary-critical.png)

### Model and one row update

Use `H=-sum_edges s_i*s_j`, `s_i=+/-1`, `J=k_B=1`, zero field and **open edges**. Set

```text
W = [[sqrt(cosh K),  sqrt(sinh K)],
     [sqrt(cosh K), -sqrt(sinh K)]]
A[r,u,l,d] = sum_s W[s,r] W[s,u] W[s,l] W[s,d]
b = [1/sqrt(cosh K), 0]
```

`WW^T` equals `exp(K s_i s_j)`, and `Wb=[1,1]`, so each absent edge has weight one. Row tensors are ordered `(left,up,down,right)`. Top and bottom edges are closed by b, and side edges are closed in the row tensors.

One row absorbs the physical index of each boundary-MPS core into the site's up leg. Combining the old MPS bond with the horizontal lattice bond grows its dimension from at most chi to at most 2chi. A **right-to-left exact SVD sweep** first makes the right environment orthonormal; a **left-to-right truncated SVD sweep** then keeps at most chi singular directions. The first L-1 cores are left-isometric. Divide **only the last core** by its Frobenius norm once per row and record the log scale. The final bottom-edge contraction and saved scales reconstruct log Z. Individual MPS entries can be signed; they are not probabilities.

Why this differs from the Gaussian matrix: Ising entries are correlated through local interactions. The first-row boundary has rank at most two across a cut (one at K=0), rather than a broad independent-entry spectrum. Later rows and critical correlations can require larger ranks, which is why rank convergence is tested rather than assumed.

### Scan, refinement and peak fits

For **L=16,32,64**, scan **K=0.40..0.50 inclusive, h=0.0025, chi=16** (41 points per size). Refine **nine points at spacing 0.00125**, centered on the nearest coarse peak-grid point, at **chi=16 and 32**. Centers are K=0.4775, 0.4600 and 0.4500 respectively.

Compute `c_V(K;h)=K²[g(K+h)-2g(K)+g(K-h)]/h²`, where `g=log(Z)/L²`. Fit a parabola to the maximum c_V and its two neighbors **in K**, then convert the vertex using `T_peak=1/K_peak`. At chi=32, using every other refined g sample repeats the h=0.0025 estimate without mixing rank and step changes.

| L | Tpeak chi=16, h=.0025 | Tpeak chi=16, h=.00125 | Tpeak chi=32, h=.00125 | Tpeak chi=32, h=.0025 | Final peak height |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.093074661 | 2.093104912 | 2.093104912 | 2.093074661 | 1.115448 |
| 32 | 2.179655090 | 2.179676209 | 2.179676246 | 2.179655128 | 1.387532 |
| 64 | 2.224193362 | 2.224219480 | 2.224220423 | 2.224194294 | 1.683394 |

| L | Rank-change peak shift (%) | Step-change peak shift (%) | Coarse chi=16 time (s) | Nine-point chi=16 time (s) | Nine-point chi=32 time (s) |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.69e-9 | 0.001445 | 0.853 | 0.208 | 0.471 |
| 32 | 1.70e-6 | 0.000969 | 5.419 | 1.137 | 4.175 |
| 64 | 0.0000424 | 0.001175 | 24.523 | 5.426 | 20.316 |

Every peak shift is below **1%**. The final chi=32 size fit `Tpeak=Tc+a/L` gives:

| Fit sizes | Tc | a | Relative error vs Onsager |
|---|---:|---:|---:|
| 16,32,64 | 2.267506089 | -2.793295425 | 0.074001% |
| 32,64 | 2.268764600 | -2.850827331 | 0.018540% |

Both meet the **1%** target. The all-size intercept becomes **2.267505128** at chi=16 and the fine step, and **2.267484527** at chi=32 and the wider step. The fraction of spins affected by open boundaries falls with increasing L: the rounded peaks rise and move towards Tc **from below**. A linear finite-range fit is not an exact finite-size law or a statistical confidence interval.

### Small open-square independent checks

NumPy enumerates all **65,536** configurations of the open 4 x 4 square; Rust boundary MPS uses chi=8.

| K | MPS g | Enumeration g | Absolute error | Required bound |
|---:|---:|---:|---:|---:|
| 0 | 0.6931471805599453 | 0.6931471805599453 | 0 | 1e-8 |
| .44 | 0.8542220240137677 | 0.8542220240137676 | 1.11e-16 | 1e-8 |
| .7 | 1.1317230403126313 | 1.1317230403126310 | 2.22e-16 | 1e-8 |

At K=.44 and h=.0025, curvature gives **c_V=0.439393073**; `K² Var(H)/16` gives **0.439396912**. Difference **3.84e-6 < 1e-3**. K=0 gives g=log 2, and the Rust unit tests also verify this limit at several sizes.

The prescribed L=64, K=.45, chi=16 example gives **g=0.9368851604782745**. Its full final MPS, L scales and row/bond truncation diagnostics are in `runs/boundary64/`. All full runs supporting the L=64 peak fits, at both ranks, remain under `runs/boundary/`.

## 3. Periodic squares: TRG coarse-graining

![Periodic TRG specific heat peaks grow and approach the transition from above; both size fits converge close to Onsager](https://github.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/releases/download/week10-results/trg-critical.png)

### One TRG round and normalization

Use the same Ising weights but **periodic boundaries**, with power-of-two L and **n=2 log2 L rounds**. Normalize by `max(abs(A))` before each round, adding **N log(scale)** for the current number N of identical tensors. Split the tensor along `(l,u)|(r,d)` and `(d,l)|(u,r)` with the retained singular values shared as square roots between factors. Contract the four-factor ring

```text
Anew[r,u,l,d] = sum_w,a,b,g S1[w,a,r] S2[a,b,u] S3[l,b,g] S4[d,g,w]
```

The split can discard information; regrouping itself is exact. Each round halves the tensor count and rotates the coarse lattice; **two rounds halve its side**. L=1024 has **1,048,576 spins** and requires **20 rounds**. The final closure is `sum_i,j A[i,j,i,j]`, not the all-equal diagonal. The saved evidence contains each round's site count, scale, both complete singular spectra, relative discarded squared weights, all four retained split factors and the final tensor. Every final closure is independently recomputed with NumPy; all small-system TRG rounds are independently replayed, including their split reconstruction errors and regrouping.

For background, [Levin and Nave's original TRG paper](https://arxiv.org/abs/cond-mat/0611687) explains tensor splitting and regrouping (square lattice in Figure 3; honeycomb in Figure 2). Its demonstration uses the **triangular** Ising model; this exercise uses the **square-lattice** critical temperature, not the triangular one.

### Periodic peak study

For **L=64,256,1024**, use the same coarse and refined K grids, ranks, curvature formula and three-point fits as in Part 2. Refined centers are **K=.4375,.4400,.4400**. All final estimates use **chi=32, h=.00125**; chi=16 provides the coarse view and a convergence comparison.

| L | Tpeak chi=16, h=.0025 | Tpeak chi=16, h=.00125 | Tpeak chi=32, h=.00125 | Tpeak chi=32, h=.0025 | Final peak height |
|---:|---:|---:|---:|---:|---:|
| 64 | 2.281891945 | 2.283843868 | 2.281601855 | 2.281598316 | 2.251392 |
| 256 | 2.272070771 | 2.272536962 | 2.272260190 | 2.271899889 | 2.903819 |
| 1024 | 2.270994870 | 2.270812696 | 2.270012404 | 2.270708648 | 3.162089 |

| L | Rank-change peak shift (%) | Step-change peak shift (%) | Coarse chi=16 time (s) | Nine-point chi=16 time (s) | Nine-point chi=32 time (s) |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.098265 | 0.000155 | 6.262 | 1.478 | 86.131 |
| 256 | 0.012180 | 0.015856 | 8.858 | 2.173 | 119.974 |
| 1024 | 0.035255 | 0.030671 | 11.321 | 2.814 | 140.770 |

All peak shifts meet **1%**. Peak heights at the largest size remain noticeably step-dependent: this work establishes the requested **peak-position** stability, not convergence of peak heights. The optional logarithmic peak-height extension was not undertaken.

| Final chi=32 fit sizes | Tc | a | Relative error vs Onsager |
|---|---:|---:|---:|
| 64,256,1024 | 2.269204722 | 0.792882428 | 0.000855% |
| 256,1024 | 2.269263142 | 0.767244169 | 0.003430% |

Both meet **1%**. The all-size intercept becomes **2.269502967** at chi=16 and the fine step, and **2.269489324** at chi=32 and the wider step (shifts of 0.01314% and 0.01254%). Periodic boundaries eliminate the open surface. These finite-size peaks approach Tc **from above**, while the open-square peaks approach from below. The two geometries must be fitted separately, but their bulk estimates agree closely.

### Small periodic-square independent checks

Enumeration counts one bond per site in each positive direction. At **L=2**, the positive-direction construction contains **distinct parallel bonds**; they must not be deduplicated into one edge.

| L | chi | K | TRG g | Enumeration g | Absolute error | Required bound |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | .44 | 1.0943416226313230 | 1.0943416226313225 | 4.44e-16 | 1e-8 |
| 4 | 8 | .44 | 0.9689397693774947 | 0.9690454086698851 | 0.0001056393 | 0.002 |

At L=2, K=.44 and h=.0025, curvature gives **c_V=0.404281937**; `K² Var(H)/4` gives **0.404278233**. Difference **3.70e-6 < 1e-3**.

The prescribed million-spin case L=1024, K=.44, chi=16 gives **g=0.9287239625770719**. Full step evidence is in `runs/trg1024/`. All full runs supporting the L=1024 peak fits, including all nine points at each rank, remain under `runs/trg/`.

## Evidence inventory and classroom preparation

| Requirement | Evidence / verification |
|---|---|
| Seeded Rust Gaussian matrix and five saved reconstructions | `runs/svd/run.json` and six full float64 matrices |
| Independent measured SVD errors and tail predictions | `artifacts/summary.json`, `artifacts/svd.png` |
| Open 16,32,64 and periodic 64,256,1024 full coarse grids | `artifacts/{boundary,trg}-coarse.json` |
| Nine-point refinement at both ranks, per size | `artifacts/{boundary,trg}-refined.json` |
| Rank, step, all-size and largest-two comparisons | `artifacts/summary.json` and tables beside the figures above |
| Small-square g and energy-variance checks | `artifacts/small-runs.json`, `artifacts/summary.json` |
| Full peak-stencil tensors, scales, diagnostics and timings | Retained locally in `runs/boundary/`, `runs/trg/`; reproducible from source; scan tables and timings are also in the compact Release archive |
| Independent final contractions and small TRG round replay | `scripts/evidence.py`; 7,890 recorded numerical checks |
| Source-only repository and reproducibility | Git ignores `runs/`, `artifacts/`, `target/`, `.venv/`; locks, specification and comparison scripts are committed |

For the next session, show the three figures and explain: (1) why the Gaussian spectrum does not support cheap 10% compression; (2) how one row absorption grows and recompresses the MPS; (3) how one TRG split/regroup round changes the number of sites; and (4) how the peaks grow and approach the same bulk transition from opposite sides. The code, evidence and explanation are prepared; the **live classroom demonstration remains to be given by the student**.
