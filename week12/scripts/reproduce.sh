#!/usr/bin/env bash
# Run in a fresh checkout with pred 0.6.0 and uv on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e artifacts ]]; then
    printf '%s\n' 'artifacts/ already exists; use a fresh checkout to preserve earlier evidence.' >&2
    exit 1
fi
[[ $(pred --version) == 'pred 0.6.0' ]]
cargo test --locked --release
cargo install --locked --path . --force
mkdir -p artifacts/pred
(
    cd artifacts/pred
    pred create SpinGlass --graph 0-1,1-2,2-3,3-0 --couplings=1,1,1,-1 -o square.json
    pred solve square.json | tee square.solve.txt
    pred create SpinGlass --random --num-vertices 100 --edge-prob 0.04 --seed 2026 -o r100.json
    pred solve r100.json | tee r100.solve.txt
)
branchopt --mode gap --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026 --out artifacts/bnb_most
branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy strong --export-labels --seed 2026 --out artifacts/bnb_strong
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 uv run --with jax==0.12.1 python train_branch.py --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json
branchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 --model artifacts/model.json --out artifacts/eval_test
branchopt --mode eval --spins 30 --edges 72 --count 8 --seed 4026 --model artifacts/model.json --out artifacts/eval_transfer
for script in plot_gap plot_bnb plot_learned; do
    if [[ -n ${PLOT_PYTHON:-} ]]; then
        "$PLOT_PYTHON" "scripts/$script.py"
    else
        uv run --with matplotlib==3.11.2 python "scripts/$script.py"
    fi
done
python3 scripts/audit.py | tee artifacts/verification.txt
