"""Read-only audit of the saved Week 12 run artifacts."""
from itertools import combinations, product
import json
import math
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[1]/"artifacts"
RULES = ("most-frac","strong","neural","root-neural")
MASK = (1<<64)-1


def read(name):
    return json.loads((BASE/name).read_text())


def make_graph(n,m,seed,i):
    # Independent implementation of the handout generator; compare actual graph
    # data as well as recalculating energies from reported spin certificates.
    state=(seed ^ (i*0xD1B54A32D192ED03))&MASK

    def next_int():
        nonlocal state
        state=(state+0x9E3779B97F4A7C15)&MASK
        z=(state^(state>>30))*0xBF58476D1CE4E5B9&MASK
        z=(z^(z>>27))*0x94D049BB133111EB&MASK
        return z^(z>>31)

    state=next_int()
    edges=set()
    for u in range(n):
        while sum(u in e for e in edges)<2:
            v=next_int()%n
            if u!=v:
                edges.add(tuple(sorted((u,v))))
    while len(edges)<m:
        u,v=next_int()%n,next_int()%n
        if u!=v:
            edges.add(tuple(sorted((u,v))))
    return [[u,v,1 if next_int()%2==0 else -1] for u,v in sorted(edges)], [next_int()%3-1 for _ in range(n)]


def energy(bits,bonds,fields):
    assert len(bits)==len(fields) and set(bits)<={0,1}
    x=[1-2*b for b in bits]
    return sum(w*x[u]*x[v] for u,v,w in bonds)+sum(h*v for h,v in zip(fields,x))


for name, exact in [("square",-2),("r100",-138)]:
    r=read(f"pred/{name}.solve.txt")
    assert r["solver"]=="ilp" and r["evaluation"]==f"Min({exact})"
gap=read("gap/run.json")
square=gap["square"]
assert math.isclose(square["lp_bound"],-4,abs_tol=1e-9) and square["lp_spins"]==[.5]*4
assert square["integer_optimum"]==-2 and square["rounded_energy"]==2
assert min(energy(x,square["bonds"],square["fields"]) for x in product((0,1),repeat=4))==-2
assert math.isclose(gap["diet"]["cost"],323/14,abs_tol=1e-9)
assert len(gap["instances"])==16 and gap["seed"]==2026
for r in gap["instances"]:
    b,h=make_graph(22,52,2026,r["id"])
    assert b==r["bonds"] and h==r["fields"]
    assert abs(r["lp_bound"]+52)<1e-6 and r["fractional_count"]==22
    assert energy(r["integer_spins"],b,h)==r["integer_optimum"]
    assert energy(r["rounded_spins"],b,h)==r["rounded_energy"]
    assert r["rounding_excess"]==r["rounded_energy"]-r["integer_optimum"]>0
    assert abs(r["gap"]-r["integer_optimum"]+r["lp_bound"])<1e-8
metrics={k:mean(r[k] for r in gap["instances"]) for k in ["lp_bound","integer_optimum","gap","fractional_count","rounding_excess"]}
assert all(abs(metrics[k]-v)<1e-6 for k,v in zip(metrics,[-52,-38.25,13.75,22,34.25]))
print("Square: LP -4, optimum -2, rounded +2; diet LP:",gap["diet"])
print("Gap means:",metrics,"; 16/16 suboptimal nearest roundings")

most,strong=(read(f"{d}/run.json") for d in ("bnb_most","bnb_strong"))
tetra=strong["tetrahedron"]
assert tetra["nodes"]==5 and tetra["energy"]==-3
assert min(energy(x,[(u,v,1) for u,v in combinations(range(4),2)],[1,0,0,0]) for x in product((0,1),repeat=4))==-3
assert [r["status"] for r in tetra["trace"]]==["branch","branch","incumbent","pruned","pruned"]
assert [round(r["bound"]) for r in tetra["trace"]]==[-6,-4,-3,-3,-2]
print("Tetrahedron trace:")
for r in tetra["trace"]:
    print(f"  node {r['node']}: fix={r['fix']} bound={r['bound']:.6f} status={r['status']} branch={r.get('branch','—')}")
for run in [most,strong]:
    assert run["seed"]==2026 and len(run["instances"])==48
    for r in run["instances"]:
        assert r["error"]<1e-9
        assert energy(r["witness"],*make_graph(22,52,2026,r["id"]))==r["energy"]
        if run is most: assert r["nodes"]==r["lp_solves"]
    print(run["parameters"]["policy"], {k:mean(r[k] for r in run["instances"]) for k in ["nodes","lp_solves","elapsed_ms"]},
          "max energy error", max(r["error"] for r in run["instances"]))

data=read("bnb_strong/dataset.json")
model=read("model.json")
assert data["seed"]==model["seed"]==2026 and model["jax"]=="0.12.1" and model["backend"]=="cpu"
assert model["precision"]=="float64" and model["settings"]=={"epochs":120,"batch_size":64,"l2":2e-4,"learning_rate":4e-3}
assert model["split"]["overlap"]==[]
assert model["split"]["train_ids"]==list(range(36)) and model["split"]["validation_ids"]==list(range(36,48))
for key,nodes in [("train",276),("validation",84)]:
    assert sum(len(x["nodes"]) for x in data[key])==nodes
    for r in data[key]:
        assert (r["bonds"],r["fields"])==make_graph(22,52,2026,r["id"])
        for node in r["nodes"]:
            assert all(len(c["features"])==12 and all(math.isfinite(x) for x in c["features"]) for c in node["candidates"])
assert len(model["history"])==120 and len(model["probes"])==6
assert model["best"]==min((r for r in model["history"] if r["epoch"]==1 or r["epoch"]%5==0),key=lambda r:r["val_ce"])
print("Disjoint splits: 276 training nodes / 36 graphs; 84 validation nodes / 12 graphs; overlap 0")
print("Best checkpoint:",model["best"])

for name,n,m,count,seed in [("eval_test",22,52,16,3026),("eval_transfer",30,72,8,4026)]:
    r=read(f"{name}/run.json")
    assert r["parameters"]=={"spins":n,"edges":m,"count":count} and len(r["instances"])==count and r["seed"]==seed
    assert r["max_probe_probability_error"]<1e-4
    for row in r["instances"]:
        b,h=make_graph(n,m,seed,row["id"])
        for rule in RULES:
            x=row["rules"][rule]
            assert x["error"]<1e-6 and energy(x["witness"],b,h)==x["energy"]
            if rule!="strong": assert x["lp_solves"]==x["nodes"]
            if rule=="root-neural": assert x["inference_calls"]<=1
            assert x["elapsed_ms"]+1e-5>=x["lp_ms"]+x["inference_ms"]
    s=r["summary"]
    for rule in ("neural","root-neural"):
        assert s[rule]["mean_nodes"]<s["most-frac"]["mean_nodes"]
        assert s[rule]["mean_total_ms"]*5<s["strong"]["mean_total_ms"]
    print(name,f"max probability error {r['max_probe_probability_error']:.3e}; max optimum error {max(t['max_error'] for t in s.values()):.3e}; LP solves = nodes for all non-Strong rules")
    print("rule\tnodes\twall_ms\tlp_ms\tinference_ms")
    for rule in RULES:
        print(rule,*(f"{s[rule][k]:.5f}" for k in ["mean_nodes","mean_total_ms","mean_lp_ms","mean_inference_ms"]),sep="\t")
    example=r["largest_neural_saving"]
    print(f"Biggest neural saving: instance {example['id']}, energy {example['highs_energy']:.0f}")
    for rule in ("most-frac","neural"):
        x=example["rules"][rule]
        print(f"  {rule}: first two (spin, first-bit) splits {x['first_splits']}, {x['nodes']} nodes, {x['elapsed_ms']:.3f} ms")

for name in ("gap","bnb","neural_bnb","eval"):
    assert (BASE/f"{name}.png").read_bytes()[:8]==b"\x89PNG\r\n\x1a\n"
print("PASS: all four Week 12 local benchmark checks and PNGs")
