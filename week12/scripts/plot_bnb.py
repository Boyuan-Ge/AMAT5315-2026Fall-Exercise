"""Draw sorted benchmark curves from the two Rust runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [json.loads(Path(f"artifacts/{d}/run.json").read_text()) for d in ("bnb_most","bnb_strong")]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
for ax, key, ylabel in zip(axs, ["nodes","lp_solves"], ["Explored nodes","LP solves"]):
    for run in runs:
        values = sorted(r[key] for r in run["instances"])
        ax.step(range(1,len(values)+1), values, where="mid", label=run["parameters"]["policy"])
    ax.set(xlabel="Instance rank (sorted separately)", ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=.25)
fig.savefig("artifacts/bnb.png", dpi=150)
