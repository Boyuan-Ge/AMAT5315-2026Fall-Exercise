"""Independent NumPy verification of raw Rust outputs; render the three required plots.

No generated evidence is committed. This script deliberately does not invoke Rust.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from study import curvature

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
ART = Path("artifacts")
TC = 2 / np.log(1 + np.sqrt(2))
CHECKS = []
plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def check(name, value, tolerance):
    assert np.isfinite(value) and value < tolerance, (name, value, tolerance)
    CHECKS.append({"name": name, "error": float(value), "tolerance": float(tolerance)})


def load_tensor(directory, descriptor):
    arr = np.fromfile(Path(directory)/descriptor["file"], dtype="<f8")
    assert arr.size == np.prod(descriptor["shape"])
    assert np.all(np.isfinite(arr))
    return arr.reshape(descriptor["shape"])


def replay(directory):
    result = json.loads((Path(directory)/"run.json").read_text())
    ev = result["evidence"]
    if result["method"] == "boundary":
        cores = [load_tensor(directory, d) for d in ev["cores"]]
        v = np.ones(1)
        for i, core in enumerate(cores):
            if i+1 < len(cores):
                m = core.reshape(-1, core.shape[-1])
                check(f"{directory}/isometry-{i}", np.max(np.abs(m.T@m-np.eye(m.shape[1]))), 1e-10)
            v = np.einsum("a,apb,p->b", v, core, ev["edge"], optimize=True)
        z = float(v[0])
        logscale = np.log(ev["scales"]).sum()
        check(f"{directory}/finalnorm", abs(np.linalg.norm(cores[-1])-1), 1e-12)
    else:
        final = load_tensor(directory, ev["final_tensor"])
        z = float(np.einsum("ijij->", final))
        logscale = sum(s["site_count"]*np.log(s["scale"]) for s in ev["steps"])
        assert len(ev["steps"]) == 2*int(np.log2(result["l"]))
        assert [s["site_count"] for s in ev["steps"]] == [result["l"]**2//2**i for i in range(len(ev["steps"]))]
    logz = logscale+np.log(z)
    check(f"{directory}/closure", abs(z-ev["closure"])/max(1., abs(z)), 1e-10)
    check(f"{directory}/g-replay", abs(logz/result["l"]**2-result["g"]), 1e-11)
    return result


def enumeration(l, k, periodic):
    states = np.arange(1 << (l*l), dtype=np.uint64)
    spins = 2*((states[:, None] >> np.arange(l*l, dtype=np.uint64)) & 1).astype(float)-1
    h = np.zeros(len(states))
    for y in range(l):
        for x in range(l):
            i = y*l+x
            if periodic or x+1 < l:
                h -= spins[:, i]*spins[:, y*l+(x+1)%l]
            if periodic or y+1 < l:
                h -= spins[:, i]*spins[:, ((y+1)%l)*l+x]
    w = np.exp(-k*h)
    z = w.sum()
    mean = np.dot(w, h)/z
    var = np.dot(w, (h-mean)**2)/z
    return {"g": float(np.log(z)/(l*l)), "c_v": float(k*k*var/(l*l)), "mean_energy": float(mean)}


def svd_evidence():
    directory = Path("runs/svd")
    info = json.loads((directory/"run.json").read_text())
    m = load_tensor(directory, info["matrix"])
    s = np.array(info["singular_values"])
    s_numpy = np.linalg.svd(m, compute_uv=False)
    check("SVD independent spectrum", np.max(np.abs(s-s_numpy)), 1e-10)
    check("SVD spectrum norm", abs(np.linalg.norm(m)-np.linalg.norm(s)), 1e-10)
    tail = np.r_[np.cumsum((s*s)[::-1])[::-1], 0.]
    predicted = np.sqrt(tail/tail[0])
    errors = []
    for entry in info["reconstructions"]:
        r = entry["rank"]
        recon = load_tensor(directory, entry["matrix"])
        error = float(np.linalg.norm(m-recon)/np.linalg.norm(m))
        check(f"SVD rank {r} tail prediction", abs(error-predicted[r]), 1e-8)
        if r == 512:
            check("SVD full rank", error, 1e-8)
        errors.append({"rank": r, "measured_error": error, "tail_prediction": float(predicted[r]), "storage_ratio": 2*r/512})
    crossing = int(np.flatnonzero(predicted < .1)[0])
    assert crossing > 256, "This Gaussian draw unexpectedly admits storage-saving 10% compression"
    # Gauge insertion preserves a factorized matrix, unlike SVD truncation.
    u, ss, vt = np.linalg.svd(m[:12, :10], full_matrices=False)
    a, b = u[:, :4]*ss[:4], vt[:4]
    gauge = np.diag([.5, 1., 2., 3.])
    check("gauge invariance", np.max(np.abs(a@b-(a@gauge)@np.linalg.solve(gauge, b))), 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(np.arange(1, 513), s, color="#2563a6")
    ax[0].set(xlabel="Singular-value index", ylabel="Singular value", title="(a) Independent Gaussian entries")
    ax[1].plot(np.arange(513), predicted, label="Discarded-spectrum prediction", color="#2563a6")
    ax[1].scatter([e["rank"] for e in errors], [e["measured_error"] for e in errors], marker="x", s=65, c="#cf4933", label="Saved reconstruction")
    ax[1].axvspan(256, 512, color="gray", alpha=.12, label="No storage saving")
    ax[1].axvline(256, ls="--", color="gray")
    ax[1].axhline(.1, ls=":", color="black", label="10% error")
    ax[1].annotate(f"First <10%: rank {crossing}\nstorage {2*crossing/512:.3f}× dense", (crossing, .1), xytext=(285, .4), arrowprops={"arrowstyle":"->"})
    ax[1].set(xlabel="Retained rank χ", ylabel="Relative Frobenius error", title="(b) Accuracy versus storage", xlim=(0, 520), ylim=(-.025, 1.04))
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ART/"svd.png"); plt.close(fig)
    return {"rng": info["rng"], "seed": info["seed"], "errors": errors,
            "first_rank_below_10_percent": crossing, "storage_ratio_at_crossing": 2*crossing/512}


def peak(rows):
    k, c = curvature(rows)
    index = int(np.argmax(c))
    assert 0 < index < len(k)-1, "Peak outside refinement window; extend scan rather than extrapolate"
    x = k[index-1:index+2]
    coef = np.polyfit(x-k[index], c[index-1:index+2], 2)
    assert coef[0] < 0
    offset = -coef[1]/(2*coef[0])
    assert abs(offset) < (k[1]-k[0])
    kp = float(k[index]+offset)
    return {"k_peak": kp, "t_peak": 1/kp, "c_peak": float(np.polyval(coef, offset)),
            "h": float(k[1]-k[0]), "stencil_runs": [r["run"] for r in sorted(rows, key=lambda r:r["k"])]}


def thermal_evidence(method, sizes):
    coarse = json.loads((ART/f"{method}-coarse.json").read_text())
    refined = json.loads((ART/f"{method}-refined.json").read_text())
    records = []
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for l, color in zip(sizes, ["#2563a6", "#cf4933", "#28916a"]):
        rows = [r for r in coarse if r["l"] == l]
        pcoarse = peak(rows)
        k, c = curvature(rows)
        ax[0].plot(1/k, c, color=color, lw=1.2, alpha=.7, label=f"L={l}, χ=16, h=.0025")
        p = {}
        for chi in [16, 32]:
            r = sorted([r for r in refined if r["l"] == l and r["chi"] == chi], key=lambda r:r["k"])
            p[chi] = peak(r)
            if chi == 32:
                wide = peak(r[::2])
                k, c = curvature(r)
                ax[0].plot(1/k, c, "o-", ms=3, color=color, label=f"L={l}, χ=32, h=.00125")
        rank_shift = abs(p[16]["t_peak"]-p[32]["t_peak"])/p[32]["t_peak"]
        step_shift = abs(wide["t_peak"]-p[32]["t_peak"])/p[32]["t_peak"]
        check(f"{method} L{l} rank stability", rank_shift, .01)
        check(f"{method} L{l} step stability", step_shift, .01)
        records.append({"l": l, "coarse16": pcoarse, "refined16": p[16], "refined32": p[32],
                        "wide32": wide, "relative_rank_shift": rank_shift, "relative_step_shift": step_shift,
                        "coarse_solver_seconds": sum(r["solver_seconds"] for r in rows),
                        "refined16_seconds": sum(r["solver_seconds"] for r in refined if r["l"] == l and r["chi"] == 16),
                        "refined32_seconds": sum(r["solver_seconds"] for r in refined if r["l"] == l and r["chi"] == 32)})
    fits = {}
    for label, start in [("all_sizes", 0), ("largest_two", 1)]:
        rec = records[start:]
        slope, intercept = np.polyfit([1/r["l"] for r in rec], [r["refined32"]["t_peak"] for r in rec], 1)
        relative = abs(intercept-TC)/TC
        check(f"{method} {label} Tc", relative, .01)
        fits[label] = {"tc": float(intercept), "a": float(slope), "relative_error": float(relative)}
        x = np.linspace(0, 1/min(sizes), 100)
        ax[1].plot(x, intercept+slope*x, "--" if start else "-", label=f"{label.replace('_', ' ')}: Tc={intercept:.6f}")
    # Also report the dependence of the extrapolation itself on chi and h.
    sensitivity = {}
    for label in ["refined16", "wide32"]:
        fit = np.polyfit([1/r["l"] for r in records], [r[label]["t_peak"] for r in records], 1)
        sensitivity[label] = {"tc": float(fit[1]), "relative_shift_vs_final": float(abs(fit[1]-fits["all_sizes"]["tc"])/fits["all_sizes"]["tc"])}
    for r in records:
        ax[1].scatter(1/r["l"], r["refined32"]["t_peak"], c="black", zorder=5)
        ax[1].annotate(f"L={r['l']}", (1/r["l"], r["refined32"]["t_peak"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax[0].axvline(TC, c="black", ls=":", label="Onsager Tc")
    ax[1].axhline(TC, c="black", ls=":", label="Onsager Tc")
    ax[0].set(xlabel="Temperature T=1/K", ylabel="Specific heat per spin", title=f"(a) {'Open boundary MPS' if method=='boundary' else 'Periodic TRG'}")
    ax[1].set(xlabel="Inverse side length 1/L", ylabel="Peak temperature", title="(b) Finite-range size extrapolation")
    if method == "trg": ax[0].set_xlim(2.22, 2.33)
    for a in ax: a.legend(fontsize=7, loc="best"); a.grid(alpha=.13)
    fig.tight_layout(); fig.savefig(ART/f"{method}-critical.png"); plt.close(fig)
    # Independently replay every saved closure, including all final peak-stencil runs.
    for row in coarse+refined:
        result = replay(row["run"])
        check(f"{row['run']}/table", abs(row["g"]-result["g"]), 1e-14)
    return {"method": method, "sizes": records, "fits_chi32": fits, "fit_sensitivity": sensitivity,
            "solver_seconds_total": sum(r["solver_seconds"] for r in coarse+refined)}


def small_evidence():
    rows = json.loads((ART/"small-runs.json").read_text())
    results = []
    for method, l, chi, ks in [("boundary", 4, 8, [0., .44, .7]), ("trg", 2, 16, [.44]), ("trg", 4, 8, [.44])]:
        for k in ks:
            exact = enumeration(l, k, method == "trg")
            row = next(r for r in rows if r["method"] == method and r["l"] == l and r["k"] == k)
            tolerance = .002 if method == "trg" and l == 4 else 1e-8
            check(f"{method} L{l} K{k} enumeration g", abs(exact["g"]-row["g"]), tolerance)
            record = {"method": method, "l": l, "chi": chi, "k": k, "g_solver": row["g"], "g_enumeration": exact["g"], "g_abs_error": abs(exact["g"]-row["g"])}
            if k == .44 and (method == "boundary" or l == 2):
                triple = sorted([r for r in rows if r["method"] == method and r["l"] == l and .4374 < r["k"] < .4426], key=lambda r:r["k"])
                _, cv = curvature(triple)
                check(f"{method} L{l} heat variance", abs(cv[0]-exact["c_v"]), 1e-3)
                record.update(c_v_solver=float(cv[0]), c_v_variance=exact["c_v"], c_v_abs_error=float(abs(cv[0]-exact["c_v"])))
            results.append(record)
    for row in rows:
        result = replay(row["run"])
        if row["method"] == "trg":
            # Full independent NumPy replay of both SVD cuts, regrouping and closure
            # for small tori. Factors are checked against the incoming tensor.
            k = row["k"]
            w = np.array([[np.sqrt(np.cosh(k)), np.sqrt(np.sinh(k))], [np.sqrt(np.cosh(k)), -np.sqrt(np.sinh(k))]])
            a = np.einsum("sr,su,sl,sd->ruld", w,w,w,w)
            for step in result["evidence"]["steps"]:
                a /= step["scale"]
                d = a.shape[0]
                s1,s2,s3,s4 = [load_tensor(row["run"], desc) for desc in step["factors"]]
                for label, left,right, axes,spectrum in [("a",s1,s3,(2,1,0,3),step["singular_a"]), ("b",s2,s4,(3,2,1,0),step["singular_b"])]:
                    target = a.transpose(axes).reshape(d*d, d*d)
                    values = np.linalg.svd(target, compute_uv=False)
                    check(f"{row['run']}/step{step['step']}/spectrum{label}", np.max(np.abs(values-spectrum)), 1e-8)
                    q = left.shape[-1]
                    measured = np.linalg.norm(target-left.reshape(d*d,q)@right.reshape(q,d*d))
                    predicted = np.linalg.norm(values[q:])
                    check(f"{row['run']}/step{step['step']}/split{label}", abs(measured-predicted), 1e-8)
                a = np.einsum("war,abu,lbg,dgw->ruld", s1,s2,s3,s4,optimize=True)
            stored = load_tensor(row["run"],result["evidence"]["final_tensor"])
            check(f"{row['run']}/independent-regroup", np.max(np.abs(a-stored))/max(1.,np.max(np.abs(a))), 1e-8)
    return results


def main():
    summary = {"onsager_tc": float(TC), "svd": svd_evidence(), "small": small_evidence()}
    for method, sizes in [("boundary", [16,32,64]), ("trg", [64,256,1024])]:
        summary[method] = thermal_evidence(method, sizes)
    for directory in ["runs/boundary64", "runs/trg1024"]:
        replay(directory)
    summary["checks"] = CHECKS
    summary["environment"] = {"python":platform.python_version(), "platform":platform.platform(), "numpy":np.__version__, "matplotlib":matplotlib.__version__, "rust":subprocess.check_output(["rustc", "--version"],text=True).strip(), "workers":6, "compute":"single-threaded Rust solver per job; up to six simultaneous processes; timings are not isolated benchmarks"}
    # Hash all raw evidence so the release archive and local records can be audited.
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("runs").rglob("*")) if p.is_file()}
    (ART/"raw-sha256.json").write_text(json.dumps(hashes, indent=2))
    (ART/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"checks_passed":len(CHECKS), "svd":summary["svd"], "small":summary["small"], "fits":{m:summary[m]["fits_chi32"] for m in ["boundary","trg"]}}, indent=2))


if __name__ == "__main__":
    main()
