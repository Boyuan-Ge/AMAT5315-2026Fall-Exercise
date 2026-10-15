#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --frozen
uv run python scripts/datasets.py
uv run python scripts/check_contracts.py
uv run python scripts/rbm.py
uv run python scripts/hopfield.py --part random
uv run python scripts/hopfield.py --part modern
uv run python scripts/plot_results.py
uv run python scripts/verify.py
