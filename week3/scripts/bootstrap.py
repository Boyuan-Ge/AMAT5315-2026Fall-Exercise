"""Block-bootstrap the susceptibility peak for three required block lengths."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, bootstrap_peaks, configure_plotting, curve, five_point_peak, load_groups, save_figure


def main() -> None:
    configure_plotting()
    rng = np.random.default_rng(2026)
    block_lengths = (2000, 4000, 8000)
    results = {}
    central = {}
    for lattice_size in (32, 64):
        metadata, groups = load_groups(ROOT / "artifacts" / f"window-l{lattice_size}")
        temperatures, _, chi = curve(groups, int(metadata["L"]))
        central[lattice_size], _ = five_point_peak(temperatures, chi)
        results[lattice_size] = {
            length: bootstrap_peaks(groups, lattice_size, length, 500, rng)
            for length in block_lengths
        }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    lines = ["Block-bootstrap susceptibility peaks (500 replicates)", ""]
    stable = True
    for axis, lattice_size in zip(axes, (32, 64)):
        means, errors = [], []
        for length in block_lengths:
            samples = results[lattice_size][length]
            mean, error = float(np.mean(samples)), float(np.std(samples, ddof=1))
            means.append(mean)
            errors.append(error)
            lines.append(f"L={lattice_size} block={length}: T_peak={mean:.6f} +/- {error:.6f}")
        axis.errorbar(block_lengths, means, yerr=errors, fmt="o-", capsize=4)
        axis.axhline(central[lattice_size], ls="--", color="black", label="full-sample fit")
        axis.set(xlabel="block length", title=f"L={lattice_size}", xscale="log")
        axis.legend()
        spread = max(means) - min(means)
        stable &= spread <= 2 * max(errors)
    axes[0].set_ylabel(r"bootstrap $T_{peak}$")
    figure.suptitle("Block-length stability of the susceptibility peak")
    save_figure(figure, "chi-bootstrap.png")
    plt.close(figure)

    lines.extend(["", f"Stability verdict: {'stable within bootstrap uncertainty' if stable else 'unresolved block-length dependence'}"])
    output = ROOT / "evidence" / "bootstrap.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
