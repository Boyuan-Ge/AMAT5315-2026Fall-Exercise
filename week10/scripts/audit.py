"""Read-only audit: requires local runs or the extracted release evidence archive."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def main():
    hashes = json.loads(Path("artifacts/raw-sha256.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    summary = json.loads(Path("artifacts/summary.json").read_text())
    assert len(summary["checks"]) >= 7000
    for check in summary["checks"]:
        assert np.isfinite(check["error"]) and check["error"] < check["tolerance"], check
    for method in ["boundary", "trg"]:
        coarse = json.loads(Path(f"artifacts/{method}-coarse.json").read_text())
        refined = json.loads(Path(f"artifacts/{method}-refined.json").read_text())
        assert len(coarse) == 3*41 and len(refined) == 3*2*9
        for size in summary[method]["sizes"]:
            c = sorted([r for r in coarse if r["l"] == size["l"]], key=lambda r:r["k"])
            assert np.max(np.abs(np.array([r["k"] for r in c])-np.linspace(.4,.5,41))) < 1e-12
            for chi in [16,32]:
                r = sorted([r for r in refined if r["l"]==size["l"] and r["chi"]==chi], key=lambda r:r["k"])
                assert np.max(np.abs(np.diff([x["k"] for x in r])-.00125)) < 1e-12
            assert size["relative_rank_shift"] < .01 and size["relative_step_shift"] < .01
        for fit in summary[method]["fits_chi32"].values():
            assert abs(fit["tc"]-summary["onsager_tc"])/summary["onsager_tc"] < .01
    tracked = subprocess.check_output(["git","ls-files","--", "."], text=True).splitlines()
    assert not any(p.startswith(("artifacts/","runs/","target/",".venv/")) for p in tracked)
    readme = Path("README.md").read_text()
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)",readme)
    assert len(images) == 3 and all("/releases/download/week10-results/" in x for x in images)
    for name in ["svd.png","boundary-critical.png","trg-critical.png"]:
        assert (Path("artifacts")/name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    print(f"PASS: {len(hashes)} raw-file hashes, {len(summary['checks'])} numeric checks, full study grids, three README figure embeds, source-only Git policy")


if __name__ == "__main__":
    main()
