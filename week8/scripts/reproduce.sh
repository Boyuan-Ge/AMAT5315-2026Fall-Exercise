#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UV="${UV:-$(command -v uv || printf '%s' "$HOME/.local/bin/uv")}"
"$UV" sync --frozen
WEEK8_CPUS="${WEEK8_CPUS:-$(.venv/bin/python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:8])))')}"
run() { taskset -c "$WEEK8_CPUS" .venv/bin/python "$@"; }
run scripts/prepare_data.py
run scripts/embeddings.py
run scripts/ngrams.py
OPENBLAS_NUM_THREADS=8 run scripts/memory.py
run scripts/train_language.py mlp
run scripts/train_language.py transformer
run scripts/language_analysis.py
run scripts/inspect_mlp.py
run scripts/scaling.py
run scripts/check_contracts.py
run scripts/verify.py --with-artifacts
