"""Draw the neural branching and transfer plots from the persisted evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("artifacts")
model = json.loads((P/"model.json").read_text())
test = json.loads((P/"eval_test"/"run.json").read_text())
large = json.loads((P/"eval_transfer"/"run.json").read_text())
rules = ("most-frac", "strong", "neural", "root-neural")
colors = ("#798b9b", "#c67d4a", "#248775", "#4875ad")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
h = model["history"]
for key, label in [("train_ce","Training"),("val_ce","Validation")]:
    ax[0].plot([r["epoch"] for r in h], [r[key] for r in h], label=label)
ax[0].scatter(model["best"]["epoch"], model["best"]["val_ce"], c="#222", s=30, zorder=3, label="Kept checkpoint")
ax[0].set(xlabel="Epoch", ylabel="Cross-entropy (nats)", title="Imitate strong branching")
for rule, c in zip(rules[:3], colors[:3]):
    values = sorted(r["rules"][rule]["nodes"] for r in test["instances"])
    ax[1].step(range(1, len(values)+1), values, where="mid", color=c, label=rule)
ax[1].set(xlabel="Instance rank (sorted independently)", ylabel="Explored nodes", title="16 unseen 22-spin graphs")
for a in ax:
    a.legend()
    a.grid(alpha=.2)
fig.savefig(P/"neural_bnb.png",dpi=155)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.3, 4.3), layout="constrained")
for j, run in enumerate([test,large]):
    for k,(rule,c) in enumerate(zip(rules,colors)):
        value=run["summary"][rule]["mean_nodes"]
        x=j+(k-1.5)*.19
        bar=ax[0].bar(x,value,width=.17,color=c,label=rule if j==0 else None)
        ax[0].bar_label(bar,fmt="%.1f",padding=2,fontsize=8)
ax[0].set(xticks=[0,1],xticklabels=["22 spins / 52 bonds","30 spins / 72 bonds"],
          ylabel="Mean explored nodes",title="Training at 22; transfer to 30",ylim=(0,205))
ax[0].legend(ncol=2,fontsize=8)
summary=large["summary"]
xs=range(len(rules))
lp=[summary[r]["mean_lp_ms"] for r in rules]
inf=[summary[r]["mean_inference_ms"] for r in rules]
tot=[summary[r]["mean_total_ms"] for r in rules]
ax[1].bar(xs,lp,label="LP model + solve",color="#4875ad")
ax[1].bar(xs,inf,bottom=lp,label="Features + MLP",color="#248775")
bars=ax[1].bar(xs,[max(0,t-l-i) for t,l,i in zip(tot,lp,inf)],bottom=[l+i for l,i in zip(lp,inf)],
               label="Other search work",color="#b8c5ce")
ax[1].bar_label(bars,labels=[f"{v:.1f}" for v in tot],padding=2,fontsize=9)
ax[1].set(xticks=list(xs),xticklabels=rules,ylabel="Mean search wall time (ms)",title="30-spin timing breakdown",
          ylim=(0,1.2*max(tot)))
ax[1].set_xlabel(f"Inference: neural {inf[2]:.3f} ms; root-neural {inf[3]:.3f} ms")
ax[1].legend(fontsize=8)
for a in ax:
    a.grid(axis="y",alpha=.2)
fig.savefig(P/"eval.png",dpi=155)
