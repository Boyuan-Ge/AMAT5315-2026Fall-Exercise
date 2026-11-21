"""Build figures and audit saved evidence without rerunning the Rust solver."""
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("runs")
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def load(name):
    return json.loads((RUN / name / "run.json").read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


def gates():
    d = load("gadgets")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (name, rows) in zip(axes.flat, [(g["gate"], g["rows"]) for g in d["gates"]] + [("AND–NOT–SET1", d["circuit"])]):
        floor = min(row["energy"] for row in rows)
        assert all((row["energy"] == floor) == row["allowed"] for row in rows)
        ax.bar(["".join(map(str, row["bits"])) for row in rows], [row["energy"] for row in rows],
               color=["#238364" if row["allowed"] else "#c45d45" for row in rows])
        ax.set(title=name, ylabel="Energy")
        ax.tick_params(axis="x", labelrotation=65)
    assert sum(x["energy"] == -5 for x in d["circuit"]) == 3
    fig.suptitle("Every truth-table row (green = allowed)")
    save(fig, "gates.png")


def queens():
    ns = list(range(4, 11))
    actual = [load(f"count_nq{n}")["board_count"] for n in ns]
    expected = [2, 10, 4, 40, 92, 352, 724]
    assert actual == expected
    frequencies = {}
    for n in (6, 8):
        draws = load(f"sample_nq{n}")["draws"]
        assert len(draws) == 200
        for row in draws:
            assert sorted(row) == list(range(n))
            assert len({i+x for i,x in enumerate(row)}) == len({i-x for i,x in enumerate(row)}) == n
        frequencies[n] = Counter(map(tuple, draws))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ns, actual, "o-", label="Sparse contraction")
    ax[0].scatter(ns, expected, marker="x", label="Known counts")
    ax[0].set(xlabel="Board size N", ylabel="Exact board count", yscale="log")
    ax[0].legend()
    f = frequencies[6]
    ax[1].bar(range(len(f)), [f[b] for b in sorted(f)], color="#238364")
    ax[1].axhline(50, ls="--", color="black", label="Uniform expectation")
    ax[1].set(xlabel="Six-queen solution (sorted order)", ylabel="Frequency in 200 draws", xticks=range(4), xticklabels=range(1,5))
    ax[1].legend()
    save(fig, "nqueens.png")
    return {"counts": actual, "six_frequencies": [f[b] for b in sorted(f)], "eight_unique":len(frequencies[8]),"eight_example": list(next(iter(frequencies[8])))}


def factoring():
    ferro, count, sample = map(load, ("count_p0","count_f15","sample_f15"))
    assert (ferro["e_min"], int(ferro["g_0"])) == (-180, 2)
    assert (count["e_min"], int(count["g_0"]),count["variables"]) == (-299, 1,104)
    assert abs(count["log_z"]-598.3179)<1e-4
    draws = sample["draws"]
    assert len(draws) == 2000
    for s in draws:
        assert s["energy"] == sum(c*math.prod(2*s["bits"][i]-1 for i in ix) for ix,c in count["terms"])
    freq = Counter(s["energy"] for s in draws)
    e = count["e_min"]
    p = int(count["g_0"])*math.exp(-2*e-count["log_z"])
    assert abs(freq[e]-len(draws)*p) < 4*math.sqrt(len(draws)*p*(1-p))
    bits = next(s["bits"] for s in draws if s["energy"] == e)
    extraction=subprocess.check_output(["pred","extract","runs/pred/factor_15.bundle.json","--config",",".join(map(str,bits))],text=True)
    (OUT/"extract_15.txt").write_text(extraction)
    print(extraction)
    import re
    assert re.search(r"\[\s*1,\s*1,\s*1,\s*0,\s*1\s*\]", extraction)
    assert "Min(0)" in (RUN/"pred/evaluate_15.txt").read_text()
    assert "Min(0)" in (RUN/"pred/solve_6.txt").read_text()
    assert "Min(0)" not in (RUN/"pred/solve_7.txt").read_text()
    fig, ax = plt.subplots(1,2,figsize=(11.2,4.2))
    costs={}
    for order in ("greedy","treesa"):
        costs[order]=[(load(f"plan_k{k}_{order}")["sc"],load(f"plan_k{k}_{order}")["tc"]) for k in range(2,9)]
        ax[0].plot(range(2,9),[p[0] for p in costs[order]],"o-",label=order)
    ax[0].axhline(30,ls=":",color="gray",label="8 GiB for float64 alone")
    ax[0].set(xlabel="Bits per factor k",ylabel="Space cost sc (log₂ entries)")
    ax[0].legend()
    ax[1].bar(sorted(freq),[freq[i] for i in sorted(freq)],color="#238364",label="Sampled energies")
    ax[1].scatter([e],[2000*p],color="#b53b31",marker="x",s=70,zorder=4,label=f"Expected ground-state draws: {2000*p:.1f}")
    ax[1].set(xlabel="Energy",ylabel="Frequency",title="Factoring 15 at β=2")
    ax[1].legend()
    save(fig,"factoring.png")
    return {"log_z":count["log_z"],"sc":count["sc"],"observed_ground":freq[e],"expected_ground":2000*p,"costs":costs}


gates()
summary = {"queens":queens(), "factoring":factoring()}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
