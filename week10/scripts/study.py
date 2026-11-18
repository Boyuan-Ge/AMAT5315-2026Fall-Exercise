"""Run only the specified Rust solver; never substitute an analytical answer.

Coarse/refined tables reference immutable, full-precision run directories.
Restartable: existing complete runs are validated and reused, not overwritten.
"""
import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def run_one(method, l, k, chi):
    k = round(float(k), 8)
    directory = Path(f"runs/{method}/L{l}-chi{chi}-K{k:.8f}")
    manifest = directory / "run.json"
    if not manifest.exists():
        subprocess.run([str(ROOT / "target/release/thermo"), "--method", method,
                        "--l", str(l), "--k", str(k), "--chi", str(chi),
                        "--out", str(directory)], check=True, stdout=subprocess.DEVNULL)
    result = json.loads(manifest.read_text())
    assert (result["method"], result["l"], result["k"], result["chi"]) == (method, l, k, chi)
    return {key: result[key] for key in ("method", "l", "k", "chi", "g", "log_z", "solver_seconds")} | {"run": str(directory)}


def execute(jobs, workers):
    start = time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda j: run_one(*j), jobs))
    print(f"Finished {len(jobs)} jobs in {time.monotonic()-start:.1f} s", flush=True)
    return results


def curvature(rows):
    rows = sorted(rows, key=lambda r: r["k"])
    k = np.array([r["k"] for r in rows])
    g = np.array([r["g"] for r in rows])
    h = k[1] - k[0]
    assert np.max(np.abs(np.diff(k)-h)) < 1e-10
    return k[1:-1], k[1:-1]**2 * np.diff(g, n=2) / h**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    Path("artifacts").mkdir(exist_ok=True)
    jobs = [("boundary", 4, k, 8) for k in [0., .4375, .44, .4425, .7]]
    jobs += [("trg", l, k, chi) for l, chi in [(2, 16), (4, 8)] for k in [.4375, .44, .4425]]
    small = execute(jobs, args.workers)
    Path("artifacts/small-runs.json").write_text(json.dumps(small, indent=2))
    for method, sizes in [("boundary", [16, 32, 64]), ("trg", [64, 256, 1024])]:
        jobs = [(method, l, .4 + i*.0025, 16) for l in sizes for i in range(41)]
        coarse = execute(jobs, args.workers)
        Path(f"artifacts/{method}-coarse.json").write_text(json.dumps(coarse, indent=2))
        centers = {}
        for l in sizes:
            k, cv = curvature([r for r in coarse if r["l"] == l])
            centers[l] = float(k[np.argmax(cv)])
        jobs = [(method, l, centers[l]+i*.00125, chi)
                for l in sizes for chi in [16, 32] for i in range(-4, 5)]
        refined = execute(jobs, args.workers)
        Path(f"artifacts/{method}-refined.json").write_text(json.dumps(refined, indent=2))
        print(method, "refined centers", centers, flush=True)


if __name__ == "__main__":
    main()
