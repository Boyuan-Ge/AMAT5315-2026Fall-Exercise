"""Compare Metropolis and Wolff estimates and work-normalized correlation times."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ROOT,
    blocked_error,
    bootstrap_peaks,
    configure_plotting,
    curve,
    five_point_peak,
    load_groups,
    save_figure,
    tau_int,
)


def main() -> None:
    configure_plotting()
    rng = np.random.default_rng(5315)
    datasets = {}
    for lattice_size in (32, 64):
        _, metro = load_groups(ROOT / "artifacts" / f"window-l{lattice_size}")
        _, wolff = load_groups(ROOT / "artifacts" / f"wolff-l{lattice_size}")
        datasets[lattice_size] = {"Metropolis": metro, "Wolff": wolff}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    report = ["Metropolis versus Wolff comparison", "blocked errors use 4000 measurements", ""]
    for axis, lattice_size in zip(axes, (32, 64)):
        for method, groups in datasets[lattice_size].items():
            temperatures, mean_abs, _ = curve(groups, lattice_size)
            errors = np.asarray([blocked_error(np.abs(groups[float(t)]["M"]), 4000) for t in temperatures])
            axis.errorbar(temperatures, mean_abs, yerr=errors, fmt="o-", ms=3, capsize=2, label=method)
            _, _, chi = curve(groups, lattice_size)
            peak, _ = five_point_peak(temperatures, chi)
            boot = bootstrap_peaks(groups, lattice_size, 4000, 500, rng)
            report.append(f"L={lattice_size} {method}: T_peak={peak:.6f} +/- {np.std(boot, ddof=1):.6f}")
        metro_values = np.abs(datasets[lattice_size]["Metropolis"][2.3]["M"])
        wolff_values = np.abs(datasets[lattice_size]["Wolff"][2.3]["M"])
        metro_mean, wolff_mean = np.mean(metro_values), np.mean(wolff_values)
        metro_error = blocked_error(metro_values, 4000)
        wolff_error = blocked_error(wolff_values, 4000)
        d_value = (wolff_mean - metro_mean) / np.hypot(metro_error, wolff_error)
        report.append(f"L={lattice_size} T=2.3 standardized difference d={d_value:.3f}")
        axis.set(xlabel="temperature T", ylabel=r"$\langle |M|\rangle$", title=f"L={lattice_size}")
        axis.legend()
    figure.suptitle("Sampler agreement with blocked uncertainty")
    save_figure(figure, "magnetization-compare.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, lattice_size in zip(axes, (32, 64)):
        metro = datasets[lattice_size]["Metropolis"]
        wolff = datasets[lattice_size]["Wolff"]
        temperatures = np.asarray(sorted(metro))
        metro_tau = np.asarray([tau_int(np.abs(metro[float(t)]["M"])) for t in temperatures])
        wolff_moves = np.asarray([tau_int(np.abs(wolff[float(t)]["M"])) for t in temperatures])
        work_fraction = np.asarray(
            [np.mean(wolff[float(t)]["cluster_size"]) / lattice_size**2 for t in temperatures]
        )
        wolff_work = wolff_moves * work_fraction
        axis.plot(temperatures, metro_tau, "o-", label="Metropolis sweeps")
        axis.plot(temperatures, wolff_work, "o-", label="Wolff, work normalized")
        axis.set(xlabel="temperature T", title=f"L={lattice_size}", yscale="log")
        axis.legend()
    axes[0].set_ylabel(r"$\tau_{int}$ in lattice-sweep work units")
    figure.suptitle("Autocorrelation cost after cluster-size normalization")
    save_figure(figure, "tau-compare.png")
    plt.close(figure)

    report.extend(
        [
            "",
            "Wolff tau is converted from cluster moves to sweep-equivalent work by multiplying",
            "by mean_cluster_size / L^2 at each temperature.",
        ]
    )
    output = ROOT / "evidence" / "comparison.txt"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
