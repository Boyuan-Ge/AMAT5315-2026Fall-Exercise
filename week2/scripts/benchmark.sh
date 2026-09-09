#!/usr/bin/env bash

set -euo pipefail

week2_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_file="${week2_dir}/benchmark-results.csv"
scratch_dir="/tmp/amat5315-week2-benchmark"
numpy_python="/tmp/amat5315-week2-venv/bin/python"
numpy_script="${week2_dir}/week2-sim.py"
debug_md="${week2_dir}/md/target/debug/md"
release_md="/home/boyuan/.cargo/bin/md"

mkdir -p "${scratch_dir}"

if [[ ! -x "${numpy_python}" ]]; then
    echo "Missing ${numpy_python}; create the NumPy environment first." >&2
    exit 1
fi

if [[ ! -f "${numpy_script}" ]]; then
    echo "Missing ${numpy_script}; download the course NumPy baseline first." >&2
    exit 1
fi

if [[ ! -x "${debug_md}" || ! -x "${release_md}" ]]; then
    echo "Build the debug binary and install the release binary first." >&2
    exit 1
fi

printf 'group,program,n,run,seconds\n' > "${results_file}"

measure() {
    local group="$1"
    local program="$2"
    local particle_count="$3"
    local run_number="$4"
    shift 4

    local timer_file="${scratch_dir}/${group}-${program}-${particle_count}-${run_number}.time"
    echo "Measuring ${group}: ${program}, N=${particle_count}, run ${run_number}" >&2
    /usr/bin/time -f '%e' -o "${timer_file}" "$@" >/dev/null
    printf '%s,%s,%s,%s,%s\n' \
        "${group}" "${program}" "${particle_count}" "${run_number}" \
        "$(tr -d '[:space:]' < "${timer_file}")" >> "${results_file}"
}

for run_number in 1 2 3; do
    mkdir -p "${scratch_dir}/numpy-${run_number}"
    measure timing numpy 100 "${run_number}" \
        bash -c 'cd "$1" && exec "$2" "$3"' benchmark-numpy \
        "${scratch_dir}/numpy-${run_number}" "${numpy_python}" "${numpy_script}"
    measure timing rust-debug 100 "${run_number}" \
        "${debug_md}" run \
        --out "${scratch_dir}/timing-debug-${run_number}"
    measure timing rust-release 100 "${run_number}" \
        "${release_md}" run \
        --out "${scratch_dir}/timing-release-${run_number}"
done

for force_method in naive cells; do
    for particle_count in 100 400 1600; do
        for run_number in 1 2 3; do
            measure scaling "${force_method}" "${particle_count}" "${run_number}" \
                "${release_md}" run \
                --force "${force_method}" \
                --n "${particle_count}" \
                --eq-steps 100 \
                --steps 500 \
                --out "${scratch_dir}/scaling-${force_method}-${particle_count}-${run_number}"
        done
    done
done

echo "Wrote ${results_file}" >&2
