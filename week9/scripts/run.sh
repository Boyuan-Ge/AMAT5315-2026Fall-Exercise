#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/fetch.sh
cargo +1.96.0 build --release --locked
cargo +1.96.0 test --release --locked
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
infer=target/release/infer
bp=(--method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000)
for name in prior AD ADX; do
  "$infer" --model asia.uai --evidence "$name" --out "artifacts/$name"
  "$infer" --model asia.uai --evidence "$name" "${bp[@]}" --out "artifacts/bp-$name"
done
"$infer" --model asia.uai --evidence AD --order random --seed 42 --out runs/random
"$infer" --model asia.uai --evidence AD --order greedy --out runs/greedy
"$infer" --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis
"$infer" --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay
benchmark=(--model promedus/Promedus_14.uai --evidence promedus --observations promedus/Promedus_14.uai.evid --queries 59,102,139)
for order in greedy treesa; do
  "$infer" "${benchmark[@]}" --order "$order" --out "artifacts/promedus-$order"
done
"$infer" "${benchmark[@]}" "${bp[@]}" --out artifacts/promedus-bp
