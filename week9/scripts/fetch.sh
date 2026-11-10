#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
asia=https://raw.githubusercontent.com/TensorBFS/TensorInference.jl/935ed86f8e473f523cac9461c45a1f470e74f0f9/examples/asia-network/model.uai
bench=https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition/22005844423f5bca62798548df808cc418fc0d7d/uai2014
fetch() {
  if [ ! -s "$2" ]; then
    curl --fail --location --retry 2 --output "$2.partial" "$1"
    mv -- "$2.partial" "$2"
  fi
}
mkdir -p promedus
fetch "$asia" asia.uai
for suffix in uai uai.evid uai.PR; do
  fetch "$bench/PR/Promedus_14.$suffix" "promedus/Promedus_14.$suffix"
done
fetch "$bench/MAR/Promedus_14.uai.MAR" promedus/Promedus_14.uai.MAR
