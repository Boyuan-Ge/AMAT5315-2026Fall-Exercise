"""Shared JSONL loading and statistical helpers for the Week 3 evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def load_groups(folder: Path) -> tuple[dict, dict[float, dict[str, np.ndarray]]]:
    """Load one simulator output folder, grouped by temperature."""
    with (folder / "run.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    raw: dict[float, dict[str, list[float]]] = {}
    with (folder / "series.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            temperature = round(float(row["T"]), 10)
            group = raw.setdefault(temperature, {"M": [], "E": [], "cluster_size": []})
            group["M"].append(float(row["M"]))
            group["E"].append(float(row["E"]))
            if "cluster_size" in row:
                group["cluster_size"].append(float(row["cluster_size"]))
    groups = {
        temperature: {
            name: np.asarray(values, dtype=float)
            for name, values in group.items()
            if values
        }
        for temperature, group in raw.items()
    }
    return metadata, groups


def merge_groups(*folders: Path) -> tuple[int, dict[float, dict[str, np.ndarray]]]:
    """Merge grids, allowing a later fine-window run to replace coarse overlap."""
    merged: dict[float, dict[str, np.ndarray]] = {}
    lattice_size: int | None = None
    for folder in folders:
        metadata, groups = load_groups(folder)
        current = int(metadata["L"])
        if lattice_size is not None and current != lattice_size:
            raise ValueError("cannot merge runs with different lattice sizes")
        lattice_size = current
        merged.update(groups)
    if lattice_size is None:
        raise ValueError("at least one output folder is required")
    return lattice_size, merged


def susceptibility(lattice_size: int, temperature: float, magnetization: np.ndarray) -> float:
    return float(
        lattice_size**2
        * (np.mean(magnetization**2) - np.mean(np.abs(magnetization)) ** 2)
        / temperature
    )


def curve(groups: dict[float, dict[str, np.ndarray]], lattice_size: int):
    temperatures = np.asarray(sorted(groups), dtype=float)
    mean_abs = np.asarray([np.mean(np.abs(groups[t]["M"])) for t in temperatures])
    chi = np.asarray(
        [susceptibility(lattice_size, t, groups[t]["M"]) for t in temperatures]
    )
    return temperatures, mean_abs, chi


def five_point_peak(temperatures: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit a quadratic to five points centered on the largest sampled value."""
    if len(temperatures) < 5:
        raise ValueError("five-point peak fit requires at least five temperatures")
    center = int(np.argmax(values))
    start = min(max(center - 2, 0), len(temperatures) - 5)
    indices = np.arange(start, start + 5)
    coefficients = np.polyfit(temperatures[indices], values[indices], 2)
    if coefficients[0] >= 0:
        return float(temperatures[center]), indices
    peak = -coefficients[1] / (2 * coefficients[0])
    low, high = temperatures[indices[0]], temperatures[indices[-1]]
    return float(np.clip(peak, low, high)), indices


def autocorrelation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    variance = np.dot(centered, centered) / len(centered)
    if variance <= np.finfo(float).eps:
        result = np.zeros(len(values))
        result[0] = 1.0
        return result
    fft_size = 1 << (2 * len(values) - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    covariance = np.fft.irfft(transformed * np.conjugate(transformed), n=fft_size)[: len(values)]
    covariance /= np.arange(len(values), 0, -1)
    return covariance / variance


def tau_int(values: np.ndarray) -> float:
    rho = autocorrelation(values)
    tau = 0.5
    for lag, correlation in enumerate(rho[1:], start=1):
        if not np.isfinite(correlation) or correlation <= 0:
            break
        tau += float(correlation)
        if lag >= 6 * tau:
            break
    return max(tau, 0.5)


def naive_error(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def blocked_error(values: np.ndarray, block_length: int) -> float:
    blocks = len(values) // block_length
    if blocks < 2:
        return float("nan")
    block_means = values[: blocks * block_length].reshape(blocks, block_length).mean(axis=1)
    return float(np.std(block_means, ddof=1) / np.sqrt(blocks))


def bootstrap_peaks(
    groups: dict[float, dict[str, np.ndarray]],
    lattice_size: int,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving-independent-block bootstrap of the susceptibility peak."""
    temperatures = np.asarray(sorted(groups), dtype=float)
    boot_chi = np.empty((replicates, len(temperatures)))
    for column, temperature in enumerate(temperatures):
        values = groups[float(temperature)]["M"]
        blocks = len(values) // block_length
        if blocks < 2:
            raise ValueError(f"block length {block_length} leaves fewer than two blocks")
        used = values[: blocks * block_length].reshape(blocks, block_length)
        sum_abs = np.abs(used).sum(axis=1)
        sum_square = (used**2).sum(axis=1)
        selected = rng.integers(0, blocks, size=(replicates, blocks))
        denominator = blocks * block_length
        mean_abs = sum_abs[selected].sum(axis=1) / denominator
        mean_square = sum_square[selected].sum(axis=1) / denominator
        boot_chi[:, column] = lattice_size**2 * (mean_square - mean_abs**2) / temperature
    peaks = np.empty(replicates)
    for row in range(replicates):
        peaks[row], _ = five_point_peak(temperatures, boot_chi[row])
    return peaks


def configure_plotting() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(figure, name: str) -> None:
    EVIDENCE.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(EVIDENCE / name, bbox_inches="tight")
