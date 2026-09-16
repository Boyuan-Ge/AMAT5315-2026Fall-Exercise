"""Generate magnetization and susceptibility evidence from the Metropolis runs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, configure_plotting, curve, five_point_peak, merge_groups, save_figure


def main() -> None:
    configure_plotting()
    data = {}
    for lattice_size in (32, 64):
        actual_l, groups = merge_groups(
            ROOT / "artifacts" / f"coarse-l{lattice_size}",
            ROOT / "artifacts" / f"window-l{lattice_size}",
        )
        data[lattice_size] = (*curve(groups, actual_l),)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for lattice_size, (temperatures, mean_abs, _) in data.items():
        axis.plot(temperatures, mean_abs, "o-", ms=3.2, label=f"L={lattice_size}")
    axis.set(xlabel="temperature T", ylabel=r"$\langle |M|\rangle$", title="2D Ising magnetization")
    axis.legend()
    save_figure(figure, "magnetization.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for lattice_size, (temperatures, _, chi) in data.items():
        peak, indices = five_point_peak(temperatures, chi)
        axis.plot(temperatures, chi, "o-", ms=3.2, label=f"L={lattice_size}; peak {peak:.4f}")
        axis.scatter(temperatures[indices], chi[indices], s=38, facecolors="none", edgecolors="black")
        axis.axvline(peak, ls="--", lw=1)
    axis.set(xlabel="temperature T", ylabel=r"$\chi$", title="Susceptibility and five-point peak fits")
    axis.legend()
    save_figure(figure, "susceptibility.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
