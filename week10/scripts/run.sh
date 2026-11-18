#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo test --locked --release
cargo install --locked --path . --force
if [[ ! -f runs/svd/run.json ]]; then
  thermo --method svd --out runs/svd
fi
if [[ ! -f runs/boundary64/run.json ]]; then
  thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
fi
if [[ ! -f runs/trg1024/run.json ]]; then
  thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024
fi
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/study.py --workers 6
OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/evidence.py
.venv/bin/python scripts/audit.py
