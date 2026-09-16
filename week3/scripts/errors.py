"""Diagnose correlation, error inflation, and integrated autocorrelation time."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ROOT,
    autocorrelation,
    blocked_error,
    configure_plotting,
    load_groups,
    naive_error,
    save_figure,
    tau_int,
)


def main() -> None:
    configure_plotting()
    datasets = {}
    for lattice_size in (32, 64):
        _, groups = load_groups(ROOT / "artifacts" / f"window-l{lattice_size}")
        datasets[lattice_size] = groups

    lines = ["Naive versus blocked standard error for |M|", "block length = 2000 measurements", ""]
    for lattice_size, groups in datasets.items():
        for temperature in (2.0, 2.3, 2.6):
            values = np.abs(groups[temperature]["M"])
            naive = naive_error(values)
            blocked = blocked_error(values, 2000)
            lines.append(
                f"L={lattice_size} T={temperature:.1f}: naive={naive:.8f} "
                f"blocked={blocked:.8f} inflation={blocked / naive:.2f}x"
            )
    destination = ROOT / "evidence" / "errors.txt"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, lattice_size in zip(axes, (32, 64)):
        values = np.abs(datasets[lattice_size][2.3]["M"][:12000])
        axis.plot(values, lw=0.6)
        axis.set(ylabel=r"$|M|$", title=f"Metropolis trace at T=2.3, L={lattice_size}")
    axes[-1].set_xlabel("measurement sweep")
    save_figure(figure, "trace.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for lattice_size, groups in datasets.items():
        values = np.abs(groups[2.3]["M"])
        rho = autocorrelation(values)[:1500]
        axes[0].plot(np.arange(len(rho)), rho, label=f"L={lattice_size}")
        lengths = np.asarray([1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000])
        errors = np.asarray([blocked_error(values, int(length)) for length in lengths])
        axes[1].plot(lengths, errors, "o-", label=f"L={lattice_size}")
    axes[0].set(xlabel="lag (sweeps)", ylabel="autocorrelation of |M|", title="Autocorrelation at T=2.3", ylim=(-0.1, 1.02))
    axes[1].set(xlabel="block length (sweeps)", ylabel="blocked standard error", title="Binning convergence", xscale="log")
    for axis in axes:
        axis.legend()
    save_figure(figure, "acf-binning.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for lattice_size, groups in datasets.items():
        temperatures = np.asarray(sorted(groups))
        taus = np.asarray([tau_int(np.abs(groups[float(t)]["M"])) for t in temperatures])
        axis.plot(temperatures, taus, "o-", label=f"L={lattice_size}")
    axis.set(xlabel="temperature T", ylabel=r"$\tau_{int}$ (sweeps)", title="Metropolis critical slowing down", yscale="log")
    axis.legend()
    save_figure(figure, "tau.png")
    plt.close(figure)

    print(destination.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
