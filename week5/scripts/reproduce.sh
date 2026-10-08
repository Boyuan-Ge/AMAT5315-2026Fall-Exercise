#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ first." >&2; exit 1; }

if [[ ! -f inputs/reflector.json || ! -f inputs/marmousi.json ]]; then
  curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip -o week5-inputs.zip
  python3 -m zipfile -e week5-inputs.zip .
fi

uv sync --frozen
(cd seismic && cargo build --release)
SEISMIC=seismic/target/release/seismic

uv run python scripts/ad_experiment.py
$SEISMIC --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
$SEISMIC --experiment inputs/reflector.json --mode born --out artifacts/born
$SEISMIC --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
for budget in 1 3 5 10; do
  $SEISMIC --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage treeverse --checkpoints "$budget" --out "artifacts/checkpoint-$budget"
done
$SEISMIC --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
$SEISMIC --experiment inputs/marmousi.json --mode adjoint --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --out artifacts/marmousi-image
uv run python scripts/verify_and_plot.py
