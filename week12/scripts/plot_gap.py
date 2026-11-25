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
ax[1].bar(i, [v["rounding_excess"] for v in rows], color="#be6e21")
ax[1].set(xlabel="Instance", ylabel="Rounded energy − optimum", title="Cost of nearest rounding")
for a in ax:
    a.set_xticks(i)
    a.grid(axis="y", alpha=.25)
fig.savefig("artifacts/gap.png", dpi=150)
