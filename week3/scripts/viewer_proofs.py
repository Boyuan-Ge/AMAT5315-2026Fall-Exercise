"""Compose the three stamped proof frames required by the course viewer task."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT

UP = "#ffffff"
DOWN = "#1f2a44"
BACKGROUND = "#0f1420"
PANEL = "#1e2637"
INK = "#e8edf6"
MUTED = "#93a1b8"
BLUE = "#6ea8ff"
AMBER = "#f0b849"
TC = 2 / np.log(1 + np.sqrt(2))


def main() -> None:
    frames = [json.loads(line) for line in (ROOT / "spins.jsonl").read_text(encoding="utf-8").splitlines() if line]
    output = ROOT / "evidence"
    output.mkdir(exist_ok=True)
    all_t = np.asarray([frame["T"] for frame in frames])
    all_m = np.asarray([frame["m"] for frame in frames])
    all_sweep = np.asarray([frame["sweep"] for frame in frames])

    for target in (1.8, 2.3, 3.0):
        selected = max(index for index, frame in enumerate(frames) if abs(frame["T"] - target) < 1e-9)
        frame = frames[selected]
        lattice = np.asarray(frame["spins"]).reshape(frame["L"], frame["L"])
        colors = np.zeros((frame["L"], frame["L"], 3), dtype=float)
        colors[lattice == 1] = (1, 1, 1)
        colors[lattice == -1] = tuple(int(DOWN[i : i + 2], 16) / 255 for i in (1, 3, 5))

        figure = plt.figure(figsize=(8, 13), facecolor=BACKGROUND)
        grid = figure.add_gridspec(3, 1, height_ratios=(6.4, 4.2, 1.35), hspace=0.08)
        lattice_axis = figure.add_subplot(grid[0])
        lattice_axis.imshow(colors, interpolation="nearest")
        lattice_axis.set_title("LATTICE — ONE SPIN PER CELL", color=MUTED, loc="left", fontsize=10, pad=8)
        lattice_axis.set_xticks([])
        lattice_axis.set_yticks([])
        for spine in lattice_axis.spines.values():
            spine.set_visible(False)

        history = figure.add_subplot(grid[1], facecolor=PANEL)
        history.plot(all_sweep, all_m, color=BLUE, lw=1.2, label="magnetization m")
        history.axhline(0, color=MUTED, lw=0.7, alpha=0.5)
        history.axvline(frame["sweep"], color=INK, lw=1.1, alpha=0.75)
        history.set(xlabel="cumulative step", ylabel="m", ylim=(-1.05, 1.05))
        history.tick_params(colors=MUTED)
        history.xaxis.label.set_color(MUTED)
        history.yaxis.label.set_color(MUTED)
        twin = history.twinx()
        twin.plot(all_sweep, all_t, color=AMBER, lw=1.1, ls="--", label="temperature T")
        twin.axhline(TC, color=AMBER, lw=0.8, alpha=0.55)
        twin.set_ylabel("T", color=MUTED)
        twin.tick_params(colors=MUTED)
        history.set_title("HISTORY — M AND T OVER THE WHOLE RUN", color=MUTED, loc="left", fontsize=10, pad=8)
        for axis in (history, twin):
            for spine in axis.spines.values():
                spine.set_color("#2c3650")

        caption = figure.add_subplot(grid[2], facecolor="#171d2b")
        caption.set_xticks([])
        caption.set_yticks([])
        for spine in caption.spines.values():
            spine.set_color("#2c3650")
        lines = [
            "AMAT5315 · Week 3 — Ising lattice viewer",
            f"T = {frame['T']:.3f}   ·   sweep {frame['sweep']}   ·   m = {frame['m']:.4f}   ·   frame {selected} / {len(frames)-1}   ·   L = {frame['L']}   ·   Onsager T_c = {TC:.4f}",
            f"spins.jsonl   ·   {(ROOT / 'spins.jsonl').stat().st_size / 1048576:.1f} MB   ·   {len(frames)} frames",
            "source: raw.githubusercontent.com/Boyuan-Ge/AMAT5315-2026Fall-Exercise/main/week3/spins.jsonl",
            "captured " + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ]
        for index, line in enumerate(lines):
            caption.text(0.025, 0.84 - 0.19 * index, line, color=INK if index == 0 else MUTED, fontsize=10 if index == 0 else 8.5, fontweight="bold" if index == 0 else "normal")

        figure.savefig(output / f"viewer-T{target:.1f}.png", dpi=150, facecolor=BACKGROUND, bbox_inches="tight")
        plt.close(figure)
        print(f"viewer-T{target:.1f}.png: frame={selected} sweep={frame['sweep']} m={frame['m']:.4f}")


if __name__ == "__main__":
    main()
