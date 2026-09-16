"""Validate the generated Week 3 contract outputs and committed evidence."""

import json
from pathlib import Path

from common import ROOT


def count_and_check_series(folder: Path, expected: int, wolff: bool = False) -> None:
    with (folder / "run.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required_metadata = {"L", "update", "t_grid", "discard", "measure", "seed", "sample_every", "time_unit"}
    assert required_metadata <= metadata.keys(), folder
    count = 0
    with (folder / "series.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert {"L", "T", "sweep", "M", "E"} <= row.keys(), folder
            assert ("cluster_size" in row) == wolff, folder
            count += 1
    assert count == expected, f"{folder}: expected {expected} rows, found {count}"


def main() -> None:
    for lattice_size in (32, 64):
        count_and_check_series(ROOT / "artifacts" / f"coarse-l{lattice_size}", 21 * 5000)
        count_and_check_series(ROOT / "artifacts" / f"window-l{lattice_size}", 13 * 100000)
        count_and_check_series(ROOT / "artifacts" / f"wolff-l{lattice_size}", 13 * 100000, wolff=True)

    frames = 0
    previous_sweep = -1
    with (ROOT / "spins.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            assert set(frame) == {"L", "T", "sweep", "m", "spins"}
            assert frame["L"] == 64 and len(frame["spins"]) == 64**2
            assert all(spin in (-1, 1) for spin in frame["spins"])
            assert frame["sweep"] > previous_sweep
            previous_sweep = frame["sweep"]
            frames += 1
    assert frames == 41 * 10, f"expected 410 viewer frames, found {frames}"

    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and ".venv" not in path.parts and "target" not in path.parts:
            if path.stat().st_size >= 5_000_000 and not ({"artifacts", "runs"} & set(path.parts)):
                raise AssertionError(f"committable file exceeds 5 MB: {path}")
    print("validated 5,410,000 measurement rows, 410 frames, schemas, and file sizes")


if __name__ == "__main__":
    main()
