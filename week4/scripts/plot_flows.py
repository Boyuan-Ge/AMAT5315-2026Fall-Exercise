"""Create fluid evidence from retained artifact runs; invoke from week4."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)


def frames(name):
    return [json.loads(line) for line in (ART / name / "fields.jsonl").read_text().splitlines()]


def frame_at(name, time):
    data = frames(name)
    return min(data, key=lambda row: abs(row["t"] - time))


def diagnostic(name):
    rows = []
    stop = None
    for line in (ART / f"{name}.tsv").read_text().splitlines()[1:]:
        t, e, z = line.split("\t")
        if e == "non-finite":
            stop = float(t)
        else:
            rows.append((float(t), float(e), float(z)))
    return np.asarray(rows), stop


def field_error(first, second, keys=("u", "v")):
    a = np.concatenate([np.asarray(first[k], dtype=float) for k in keys])
    b = np.concatenate([np.asarray(second[k], dtype=float) for k in keys])
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def tg_plot():
    exact = json.loads((ART / "taylor-green" / "exact-t1.json").read_text())
    final = frame_at("taylor-green", 1)
    error = field_error(final, exact)
    print("Taylor-Green relative velocity error at t=1:", error)
    assert error < 1e-5
    data = [frame_at("taylor-green", t) for t in (0, 1)]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    n = 64
    axis = np.arange(n)*2*math.pi/n
    for ax, item in zip(axs, data):
        omega = np.asarray(item["omega"]).reshape(n, n)
        image = ax.imshow(omega, origin="lower", extent=[0, 2*math.pi, 0, 2*math.pi], cmap="RdBu_r", vmin=-2, vmax=2)
        u = np.asarray(item["u"]).reshape(n, n)
        v = np.asarray(item["v"]).reshape(n, n)
        ax.quiver(axis[::8], axis[::8], u[::8, ::8], v[::8, ::8], color="black", scale=12, width=.003)
        ax.set(xlabel="x", ylabel="y", title=f"t={item['t']:g}, max |ω|={np.max(np.abs(omega)):.3f}")
    fig.colorbar(image, ax=axs, label="vorticity ω", shrink=.8)
    fig.suptitle(f"Taylor–Green decay; relative velocity error at t=1: {error:.2e}")
    fig.savefig(OUT / "taylor-green.png", dpi=180)
    plt.close(fig)


def random_plot():
    times = [0, 2, 5, 10]
    selected = [frame_at("random", t) for t in times]
    n = 128
    vmax = max(abs(v) for v in selected[0]["omega"])
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.0), constrained_layout=True)
    diag, _ = diagnostic("random")
    for ax, item in zip(axs, selected):
        omega = np.asarray(item["omega"]).reshape(n, n)
        image = ax.imshow(omega, origin="lower", extent=[0, 2*math.pi, 0, 2*math.pi], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        row = diag[np.argmin(abs(diag[:, 0]-item["t"]))]
        ax.set(xlabel="x", ylabel="y", title=f"t={item['t']:g}  E={row[1]:.3f}  Z={row[2]:.3f}")
    fig.colorbar(image, ax=axs, label="vorticity ω (shared scale)", shrink=.8)
    fig.suptitle("Seeded random flow: fine filaments fade as viscosity acts")
    fig.savefig(OUT / "random.png", dpi=180)
    plt.close(fig)
    print("random initial/final E,Z:", diag[0, 1:], diag[-1, 1:])


def blowup_plot():
    summary = json.loads((ART / "scan" / "summary.json").read_text())
    results = {float(k): v for k, v in summary["random_rk4_survives"].items()}
    stable = max(k for k, v in results.items() if v)
    unstable = min(k for k, v in results.items() if not v and k > stable)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    inset_curves = []
    for dt in [.032, .033]:
        name = "scan/plot-tg-rk4-0.033" if dt == .033 else f"scan/tg-rk4-{dt:.3f}"
        row, stop = diagnostic(name)
        axs[0].semilogy(row[:, 0], row[:, 1], label=f"RK4 Δt={dt:.3f}")
        if stop is not None:
            axs[0].axvline(stop, color="tab:red", linestyle=":", alpha=.7)
            axs[0].text(stop, .02, f"stop {stop:.2f}", rotation=90, va="bottom", fontsize=8)
    t = np.linspace(0, 8, 300)
    axs[0].semilogy(t, .25*np.exp(-.4*t), "k--", label="exact energy")
    axs[0].set(title="Taylor–Green: predicted limit 0.0316", xlabel="time t", ylabel="energy E(t)", xlim=(0, 8), ylim=(1e-3, 1e2))
    axs[0].legend(fontsize=8)
    for dt in [stable, unstable]:
        name = f"scan/plot-random-rk4-{dt:.3f}" if dt == unstable else f"scan/random-rk4-{dt:.3f}"
        row, stop = diagnostic(name)
        axs[1].semilogy(row[:, 0], row[:, 1], label=f"RK4 Δt={dt:.3f}")
        inset_curves.append((row, f"RK4 {dt:.3f}"))
        if stop is not None:
            axs[1].axvline(stop, color="tab:red", linestyle=":", alpha=.7)
            axs[1].text(stop, 1.0, f"stop {stop:.2f}", rotation=90, va="bottom", fontsize=8)
    row, stop = diagnostic("scan/plot-random-euler-0.010")
    axs[1].semilogy(row[:, 0], row[:, 1], ":", color="black", label="Euler Δt=0.010")
    inset_curves.append((row, "Euler 0.010"))
    if stop is not None:
        axs[1].axvline(stop, color="black", linestyle=":", alpha=.6)
        axs[1].text(stop, 2.0, f"Euler stop {stop:.2f}", rotation=90, va="bottom", fontsize=8)
    axs[1].set(title=f"Random flow: advective bound {summary['advective_bound']:.4f}", xlabel="time t", ylabel="energy E(t)", xlim=(0, 10), ylim=(1e-2, 1e2))
    axs[1].legend(fontsize=8)
    inset = axs[1].inset_axes([.34, .22, .42, .45])
    for data, label in inset_curves:
        inset.semilogy(data[:, 0], data[:, 1], ":" if label.startswith("Euler") else "-", color="black" if label.startswith("Euler") else None, lw=1.2)
    inset.set(xlim=(0, 1.2), ylim=(.3, 1e3), title="onset, t ≤ 1.2")
    inset.tick_params(labelsize=7)
    fig.savefig(OUT / "blowup.png", dpi=180)
    plt.close(fig)
    print("random stable/unstable bracket:", stable, unstable)


def sensitivity_plot():
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for case, label in [("tg", "Taylor–Green"), ("random", "random flow")]:
        base = frames(f"sensitivity/{case}-base")
        changed = frames(f"sensitivity/{case}-perturbed")
        assert len(base) == len(changed)
        distance = [field_error(b, a, ("omega",)) for a, b in zip(base, changed)]
        times = np.array([x["t"] for x in base])
        if case == "tg":
            # After the first crossing, six-decimal frames cannot resolve this distance.
            threshold = 6e-7
            crossing = next((i for i, value in enumerate(distance) if value < threshold), len(distance)-1)
            ax.semilogy(times[:crossing+1], distance[:crossing+1], "o-", ms=3, label=label)
            ax.axhline(threshold, color="tab:blue", linestyle="--", alpha=.5, label="six-decimal floor")
        else:
            ax.semilogy(times, distance, "o-", ms=3, label=label)
        print(case, "relative distance start/end:", distance[0], distance[-1])
    ax.set(xlabel="time t", ylabel="relative vorticity distance", title="Same stable step; perturbed initial vorticity")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig.savefig(OUT / "sensitivity.png", dpi=180)
    plt.close(fig)


def order_plot():
    dts = np.array([.4, .25, .2])
    n = 8
    x, y = np.meshgrid(np.arange(n)*2*math.pi/n, np.arange(n)*2*math.pi/n)
    exact = {"u": (np.cos(x)*np.sin(y)*math.exp(-2*.5*2)).ravel(),
             "v": (-np.sin(x)*np.cos(y)*math.exp(-2*.5*2)).ravel()}
    errors = np.array([field_error(frames(f"order/rk4-dt{dt}")[-1], exact) for dt in dts])
    slope, intercept = np.polyfit(np.log(dts), np.log(errors), 1)
    assert abs(slope-4)/4 < .15
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.loglog(dts, errors, "o", ms=7, label="measured")
    xx = np.linspace(.19, .42, 100)
    ax.loglog(xx, np.exp(intercept)*xx**slope, label=f"RK4 fit, slope {slope:.2f}")
    ax.axhline(7e-7, color="gray", ls="--", label="six-decimal storage floor")
    ax.set(xlabel="time step Δt", ylabel="relative velocity error at t=2", title="Taylor–Green RK4 time order")
    ax.legend()
    ax.grid(True, which="both", alpha=.25)
    fig.savefig(OUT / "order.png", dpi=180)
    plt.close(fig)
    print("fluid RK4 order slope:", slope)


def convergence_plot():
    dts = [.02, .0125, .01]
    ref = frames("convergence/rk4-dt0.0025")[-1]
    fields = {dt: frames(f"convergence/rk4-dt{dt}")[-1] for dt in dts}
    errors = [field_error(fields[dt], ref, ("omega",)) for dt in dts]
    slope, intercept = np.polyfit(np.log(dts), np.log(errors), 1)
    assert 3.7 <= slope <= 4.3
    rich = field_error(fields[.02], fields[.01], ("omega",)) / 15
    predicted = {dt: rich*(dt/.01)**4 for dt in dts}
    valid = [dt for dt in dts if predicted[dt] < 5e-6]
    chosen = max(valid)
    measured = errors[dts.index(chosen)]
    assert measured < 5e-6
    report = {
        "reference": {"n":128,"dt":.0025,"t":2,"seed":2026,"nu":.004,"k_band":[2,6]},
        "runs": [{"dt":dt,"relative_error":err,"richardson_prediction":predicted[dt]} for dt,err in zip(dts,errors)],
        "log_log_slope":slope,
        "richardson_at_dt_0_01":rich,
        "threshold":5e-6,
        "chosen_dt":chosen,
        "chosen_predicted_error":predicted[chosen],
        "chosen_measured_error":measured,
    }
    (OUT / "convergence.json").write_text(json.dumps(report,indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.loglog(dts, errors, "o", ms=7, label=f"measured, slope {slope:.3f}")
    xx=np.linspace(.009,.021,150)
    ax.loglog(xx,np.exp(intercept)*xx**slope,label="log-log fit")
    ax.loglog(xx,rich*(xx/.01)**4,"--",label="Richardson prediction")
    ax.axhline(5e-6,color="gray",ls=":",label="required error < 5e-6")
    ax.scatter([chosen],[measured],s=120,marker="*",color="tab:green",zorder=5,label=f"chosen Δt={chosen}")
    ax.set(xlabel="time step Δt",ylabel="relative ω error at t=2",title="Random-flow time-step refinement")
    ax.grid(True,which="both",alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(OUT / "convergence.png",dpi=180)
    plt.close(fig)
    print("random time-order slope:",slope)
    print("chosen dt:",chosen,"predicted:",predicted[chosen],"measured:",measured)


if __name__ == "__main__":
    tg_plot()
    random_plot()
    blowup_plot()
    sensitivity_plot()
    order_plot()
    convergence_plot()
