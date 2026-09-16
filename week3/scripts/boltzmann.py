"""Check the canonical Boltzmann reweighting slope at T=3.0 and 3.1."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, configure_plotting, load_groups, save_figure


def main() -> None:
    configure_plotting()
    meta_30, group_30 = load_groups(ROOT / "runs" / "T3.0")
    meta_31, group_31 = load_groups(ROOT / "runs" / "T3.1")
    lattice_size = int(meta_30["L"])
    energy_30 = next(iter(group_30.values()))["E"] * lattice_size**2
    energy_31 = next(iter(group_31.values()))["E"] * lattice_size**2
    lower = min(np.min(energy_30), np.min(energy_31))
    upper = max(np.max(energy_30), np.max(energy_31))
    bins = np.linspace(lower, upper, 41)
    hist_30, edges = np.histogram(energy_30, bins=bins, density=True)
    hist_31, _ = np.histogram(energy_31, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = (hist_30 > 0) & (hist_31 > 0)
    log_ratio = np.log(hist_30[valid] / hist_31[valid])
    slope, intercept = np.polyfit(centers[valid], log_ratio, 1)
    expected = 1 / 3.1 - 1 / 3.0

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].step(centers, hist_30, where="mid", label="T=3.0")
    axes[0].step(centers, hist_31, where="mid", label="T=3.1")
    axes[0].set(xlabel="total energy E", ylabel="density", title="Energy histograms")
    axes[0].legend()
    axes[1].scatter(centers[valid], log_ratio, s=22, label="observed log ratio")
    axes[1].plot(centers[valid], slope * centers[valid] + intercept, label=f"fit slope={slope:.5f}")
    axes[1].plot(
        centers[valid],
        expected * centers[valid] + np.mean(log_ratio - expected * centers[valid]),
        "--",
        label=f"Boltzmann slope={expected:.5f}",
    )
    axes[1].set(xlabel="total energy E", ylabel=r"$\log[P_{3.0}(E)/P_{3.1}(E)]$", title="Boltzmann ratio test")
    axes[1].legend()
    figure.suptitle(f"Canonical energy check (L={lattice_size})")
    save_figure(figure, "boltzmann.png")
    plt.close(figure)
    print(f"observed slope={slope:.8f}; expected slope={expected:.8f}")


if __name__ == "__main__":
    main()
