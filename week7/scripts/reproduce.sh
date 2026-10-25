#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UV="${UV:-$(command -v uv || printf '%s' "$HOME/.local/bin/uv")}"
"$UV" sync --frozen
WEEK7_CPUS="${WEEK7_CPUS:-$(.venv/bin/python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:8])))')}"
run() { taskset -c "$WEEK7_CPUS" .venv/bin/python "$@"; }
run scripts/prepare_inputs.py
run scripts/check_contracts.py
run scripts/train.py diffusion
run scripts/train.py flow
run scripts/mixture_experiments.py
run scripts/photo_experiments.py
run scripts/plot_results.py
run scripts/verify.py --with-artifacts
