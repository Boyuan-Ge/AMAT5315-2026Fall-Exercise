"""Write the five-point susceptibility peak estimates and finite-size extrapolation."""

from common import ROOT, curve, five_point_peak, load_groups


def main() -> None:
    lines = ["Five-point quadratic susceptibility peak fits", ""]
    peaks = {}
    for lattice_size in (32, 64):
        metadata, groups = load_groups(ROOT / "artifacts" / f"window-l{lattice_size}")
        temperatures, _, chi = curve(groups, int(metadata["L"]))
        peak, indices = five_point_peak(temperatures, chi)
        peaks[lattice_size] = peak
        points = ", ".join(f"{temperatures[i]:.2f}" for i in indices)
        lines.append(f"L={lattice_size}: T_peak={peak:.6f}; fit temperatures: {points}")
    extrapolated = 2 * peaks[64] - peaks[32]
    lines.extend(
        [
            "",
            f"Linear 1/L extrapolation Tc = 2*T_peak(64)-T_peak(32) = {extrapolated:.6f}",
            "Exact Onsager Tc = 2.269185",
            f"Absolute difference = {abs(extrapolated - 2.269185):.6f}",
        ]
    )
    destination = ROOT / "evidence" / "peaks.txt"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
