#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS=8
cargo test --locked --release
cargo install --path . --locked --force
mkdir -p runs/pred
pred --version | tee runs/pred/version.txt
pred path Factoring SpinGlass | tee runs/pred/path.txt
for pair in 6:2 7:2 15:3; do
  t=${pair%:*}; n=${pair#*:}
  pred create Factoring --target "$t" --m 2 --n "$n" -o "runs/pred/factor_$t.json"
  pred reduce "runs/pred/factor_$t.json" --to SpinGlass -o "runs/pred/factor_$t.bundle.json"
  pred solve "runs/pred/factor_$t.bundle.json" | tee "runs/pred/solve_$t.txt"
done
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1 | tee runs/pred/evaluate_15.txt
run() {
  local out=$1; shift
  [[ -f runs/$out/run.json ]] || tnsample "$@" --seed 2026 --out "runs/$out"
}
run gadgets --mode gadgets
run count_p0 --mode count --l 10 --p-af 0 --beta 2
run count_f15 --mode count --input runs/pred/factor_15.bundle.json --beta 2
run sample_f15 --mode sample --input runs/pred/factor_15.bundle.json --beta 2 --samples 2000
for k in {2..8}; do
  t=$(( ((1 << k) - 1) ** 2 ))
  pred create Factoring --target "$t" --m "$k" --n "$k" -o "runs/pred/scale_$k.json"
  pred reduce "runs/pred/scale_$k.json" --to SpinGlass -o "runs/pred/scale_$k.bundle.json"
  for order in greedy treesa; do
    run "plan_k${k}_$order" --mode plan --order "$order" --input "runs/pred/scale_$k.bundle.json"
  done
done
for n in {4..10}; do run "count_nq$n" --mode count --n "$n"; done
for n in 6 8; do run "sample_nq$n" --mode sample --n "$n" --samples 200; done
