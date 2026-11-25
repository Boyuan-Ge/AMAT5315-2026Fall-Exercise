"""Plot saved HiGHS evidence; run from week12."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw = json.loads(Path("artifacts/gap/run.json").read_text())
rows = raw["instances"]
i = [v["id"] for v in rows]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
ax[0].plot(i, [v["lp_bound"] for v in rows], "o-", label="LP lower bound")
ax[0].plot(i, [v["integer_optimum"] for v in rows], "s-", label="Exact MILP optimum")
ax[0].set(xlabel="Instance", ylabel="Energy", title="Relaxed versus exact energy")
ax[0].legend()
ax[1].bar([e-.19 for e in i], [v["gap"] for v in rows], width=.38, color="#3c799d", label="Integrality gap")
ax[1].bar([e+.19 for e in i], [v["rounding_excess"] for v in rows], width=.38, color="#be6e21", label="Rounding excess")
ax[1].set(xlabel="Instance", ylabel="Energy above reference", title="Relaxation gap and rounding error")
ax[1].legend()
for a in ax:
    a.set_xticks(i)
    a.grid(axis="y", alpha=.25)
fig.savefig("artifacts/gap.png", dpi=150)
