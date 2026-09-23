"""Generate all fluid runs; invoke from week4 with .venv/bin/python scripts/run_flows.py."""
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release"
ART = ROOT / "artifacts"


def field(case, n):
    args = [str(BIN / "field"), case, "--n", str(n)]
    if case == "random":
        args += ["--seed", "2026", "--k-min", "2", "--k-max", "6"]
    return json.loads(subprocess.check_output(args))


def run(data, method, nu, dt, end, every, name):
    dest = ART / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN / "fluid"), "--method", method, "--nu", str(nu), "--dt", str(dt),
           "--t-end", str(end), "--every", str(every), "--out", str(dest)]
    result = subprocess.run(cmd, input=json.dumps(data), text=True, capture_output=True)
    (ART / f"{name}.tsv").write_text(result.stdout)
    if result.stderr:
        print(f"{name}: {result.stderr.strip()}")
    final = result.stdout.strip().splitlines()[-1]
    print(f"{name}: {final}")
    return result.returncode == 0


def perturbed(data):
    obj = {**data, "u": list(data["u"]), "v": list(data["v"])}
    n = data["n"]
    amplitude = 7e-5 * max(max(map(abs, obj["u"])), max(map(abs, obj["v"])))
    for y in range(n):
        yy = 2*math.pi*y/n
        for x in range(n):
            xx = 2*math.pi*x/n
            i = y*n+x
            obj["u"][i] += 4*amplitude/25*math.cos(3*xx)*math.sin(4*yy)
            obj["v"][i] -= 3*amplitude/25*math.sin(3*xx)*math.cos(4*yy)
    return obj


def main():
    ART.mkdir(exist_ok=True)
    tg64 = field("taylor-green", 64)
    random128 = field("random", 128)
    speed = max(max(map(abs, random128["u"])), max(map(abs, random128["v"])))
    print("random initial component max", speed)
    print("advective step bound", 2.83/(speed*42*math.sqrt(2)))
    run(tg64, "rk4", .1, .01, 1, .1, "taylor-green")
    exact = json.loads(subprocess.check_output([str(BIN / "field"), "taylor-green", "--n", "64", "--nu", "0.1", "--t", "1"]))
    (ART / "taylor-green" / "exact-t1.json").write_text(json.dumps(exact))
    run(random128, "rk4", .004, .01, 10, .1, "random")
    run(tg64, "rk4", .1, .04, 4, .1, "unstable/taylor-green")
    for dt in [.032, .033]:
        run(tg64, "rk4", .1, dt, 8, .5, f"scan/tg-rk4-{dt:.3f}")
    run(tg64, "rk4", .1, .033, 8, .1, "scan/plot-tg-rk4-0.033")
    random_results = {}
    for dt in [.038, .040]:
        random_results[dt] = run(random128, "rk4", .004, dt, 10, .5, f"scan/random-rk4-{dt:.3f}")
    if not any(random_results.values()):
        for dt in [.036, .034, .032, .030, .028]:
            random_results[dt] = run(random128, "rk4", .004, dt, 10, .5, f"scan/random-rk4-{dt:.3f}")
            if random_results[dt]:
                break
    if all(random_results.values()):
        for dt in [.042, .044, .046, .048]:
            random_results[dt] = run(random128, "rk4", .004, dt, 10, .5, f"scan/random-rk4-{dt:.3f}")
            if not random_results[dt]:
                break
    if not any(random_results.values()) or all(random_results.values()):
        raise RuntimeError("failed to bracket random RK4 stability boundary")
    stable = max(k for k, v in random_results.items() if v)
    unstable = min(k for k, v in random_results.items() if not v and k > stable)
    run(random128, "rk4", .004, unstable, 10, .05, f"scan/plot-random-rk4-{unstable:.3f}")
    run(random128, "euler", .004, .01, 10, .5, "scan/random-euler-0.010")
    run(random128, "euler", .004, .01, 10, .05, "scan/plot-random-euler-0.010")
    (ART / "scan" / "summary.json").write_text(json.dumps({
        "initial_component_max": speed,
        "advective_bound": 2.83/(speed*42*math.sqrt(2)),
        "random_rk4_survives": {f"{k:.3f}": v for k,v in random_results.items()}
    }, indent=2))
    for case, data, nu in [("tg", tg64, .1), ("random", random128, .004)]:
        run(data, "rk4", nu, .01, 20, .5, f"sensitivity/{case}-base")
        run(perturbed(data), "rk4", nu, .01, 20, .5, f"sensitivity/{case}-perturbed")
    tg8 = field("taylor-green", 8)
    for dt in [.4, .25, .2]:
        run(tg8, "rk4", .5, dt, 2, 2, f"order/rk4-dt{dt}")
    for dt in [.02, .0125, .01, .0025]:
        run(random128, "rk4", .004, dt, 2, 2, f"convergence/rk4-dt{dt}")


if __name__ == "__main__":
    main()
