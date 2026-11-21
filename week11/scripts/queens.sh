#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS=8
for n in {4..10}; do
  [[ -f runs/count_nq$n/run.json ]] || tnsample --mode count --n "$n" --seed 2026 --out "runs/count_nq$n"
done
for n in 6 8; do
  [[ -f runs/sample_nq$n/run.json ]] || tnsample --mode sample --n "$n" --samples 200 --seed 2026 --out "runs/sample_nq$n"
done
