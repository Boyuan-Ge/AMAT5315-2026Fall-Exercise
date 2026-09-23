#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --quiet
mkdir -p artifacts/line evidence
bin=target/release/line
"$bin" growth-map --count 241 > artifacts/line/growth-map.json
"$bin" run --n 64 --c 1 --nu 0.05 --dt 0.045 --t-end 6 --sigma 0.35 --method rk4 --frames > artifacts/line/stable.json
"$bin" run --n 64 --c 1 --nu 0.05 --dt 0.056 --t-end 6 --sigma 0.35 --method rk4 --frames > artifacts/line/unstable.json
"$bin" run --n 64 --c 1 --nu 0.002 --dt 0.02 --t-end 6.283185307179586 --sigma 0.25 --method rk4 > artifacts/line/pulse-fourier.json
"$bin" run --n 64 --c 1 --nu 0.002 --dt 0.02 --t-end 6.283185307179586 --sigma 0.25 --method rk4 --derivative centred > artifacts/line/pulse-centred.json
"$bin" run --n 64 --c 1 --nu 0.002 --dt 0.005 --t-end 6.283185307179586 --sigma 0.25 --method euler > artifacts/line/pulse-euler.json
for method in euler rk2 rk4 rk4-equal; do
  for dt in 0.02 0.01 0.005 0.0025; do
    "$bin" run --n 64 --c 1 --nu 0.05 --dt "$dt" --t-end 1 --sigma 0.35 --method "$method" > "artifacts/line/error-${method}-${dt}.json"
  done
done
.venv/bin/python scripts/plot_line.py
