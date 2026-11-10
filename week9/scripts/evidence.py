"""Independent NumPy checks, plan recounts, and figures from the actual Rust runs."""
import hashlib
import itertools
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)
CASES = ("prior", "AD", "ADX")
NAMES = "ATSLBEXD"
IDS = (59, 102, 139)
BLUE, ORANGE, GREEN = "#17668b", "#b95819", "#29734c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "svg.hashsalt": "week9-asia"})


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    (OUT / path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save(fig, name):
    path=OUT/name
    if path.suffix==".svg":
        fig.savefig(path,dpi=160,facecolor="white",bbox_inches="tight",metadata={"Date":None})
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    else:
        fig.savefig(path,dpi=160,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def uai(path):
    tokens = iter((ROOT / path).read_text().split())
    assert next(tokens) == "MARKOV"
    d = [int(next(tokens)) for _ in range(int(next(tokens)))]
    scopes = [tuple(int(next(tokens)) for _ in range(int(next(tokens))))
              for _ in range(int(next(tokens)))]
    tables = [np.array([float(next(tokens)) for _ in range(int(next(tokens)))])
              .reshape([d[v] for v in s]) for s in scopes]
    assert next(tokens, None) is None
    return d, scopes, tables


def independent_joint(scopes, tables, obs):
    # Broadcast each original UAI table onto the full labelled cube; sum only afterwards.
    grid = np.indices((2,) * 8)
    weight = np.ones((2,) * 8)
    for scope, table in zip(scopes, tables):
        weight *= table[tuple(grid[v] for v in scope)]
    for v, state in obs.items():
        weight *= grid[int(v)] == state
    return weight


def draw_diagram(scopes):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(-.9, 10.9), ylim=(-1.0, 6.0), aspect="equal")
    ax.axis("off")
    vx = [(0,3.6),(2.2,3.6),(2.2,1),(4.35,1),(6.5,1),(6.5,3.6),(9.1,4.65),(9.1,1.8)]
    fx = [(0,4.9),(1.1,3.6),(1.0,1),(3.25,1),(4.5,0),(4.5,3.6),(7.85,4.65),(7.85,1.8)]
    labels = ["p(A)","p(T|A)","p(S)","p(L|S)","p(B|S)","p(E|L,T)","p(X|E)","p(D|E,B)"]
    for i, (x,y) in enumerate(fx):
        for v in scopes[i]:
            u,w = vx[v]
            ax.plot([x,u],[y,w],color="#6f8794",lw=1.5,zorder=1)
        ax.add_patch(FancyBboxPatch((x-.43,y-.22),.86,.44,boxstyle="round,pad=.04",
                                   fc="#dfedf5",ec=BLUE,zorder=3))
        ax.text(x,y,labels[i],ha="center",va="center",zorder=4,fontsize=9)
    for (x,y), label in zip(vx,NAMES):
        ax.add_patch(Circle((x,y),.19,fc="white",ec=BLUE,lw=1.5,zorder=4))
        ax.text(x,y,label,ha="center",va="center",weight="bold",zorder=5)
    for v, end, text in [(0,(-.60,2.55),"(1, 0)"),(7,(10.4,1.8),"(1, 0)"),(6,(10.4,4.65),"(1, 1)")]:
        x,y=vx[v]; u,w=end
        color=ORANGE if v!=6 else "#657680"
        ax.plot([x,u],[y,w],color=color,lw=1.5,zorder=1)
        ax.text(u,w,text,ha="center",va="center",color=color,
                bbox=dict(boxstyle="round,pad=.2",fc="white",ec=color),zorder=4)
    x,y=vx[1]
    ax.annotate("",(x,5.05),(x,y+.19),arrowprops=dict(arrowstyle="->",color=GREEN,lw=2))
    ax.text(x,5.18,"Open T",ha="center",color=GREEN,weight="bold")
    ax.text(5.1,5.75,"ASIA: evidence A=0, D=0; retain tuberculosis T",ha="center",weight="bold",fontsize=14)
    ax.text(5.1,-.63,"Each circle is one shared index. Orange: evidence selectors. Grey X cap: sum both X-ray states.",
            ha="center",color="#385261")
    fig.savefig(ROOT / "artifacts" / "graph-preview.png",dpi=130,bbox_inches="tight")
    save(fig,"graphs.svg")


def recount(plan, dims, report):
    """Rebuild the tree from leaf subsets, not Rust's live-list algorithm."""
    scopes = [set(s) for s in plan["inputs"]]
    n = len(scopes)
    all_leaves = set(range(n))
    leaves = [{i} for i in range(n)]
    opened = set(plan["output"])
    work = 0
    generated = 0
    for i, step in enumerate(plan["steps"]):
        a,b=step["left"],step["right"]
        assert leaves[a].isdisjoint(leaves[b])
        subset=leaves[a] | leaves[b]
        seen=set().union(*(scopes[j] for j in subset))
        outside=set().union(*(scopes[j] for j in all_leaves-subset))
        retain=seen & (outside | opened)
        union=scopes[a] | scopes[b]
        assert set(step["retain"])==retain
        term=math.prod(dims[j] for j in union)
        entries=math.prod(dims[j] for j in retain)
        row=report["steps"][i]
        assert (row["result"],row["left"],row["right"])==(n+i,a,b)
        assert set(row["union"])==union and set(row["retain"])==retain
        assert (row["work"],row["entries"])==(term,entries)
        work+=term; generated=max(generated,entries)
        leaves.append(subset);scopes.append(retain)
    assert leaves[-1]==all_leaves
    largest=max(math.prod(dims[j] for j in s) for s in scopes)
    assert (work,largest,generated)==(report["work"],report["largest_tensor"],report["largest_generated"])
    return {"work":work,"largest_tensor":largest,"largest_generated":generated}


def minimum_work(scopes, dims):
    """Exhaustive subset DP over all binary trees, counting raw input scopes at leaves."""
    n=len(scopes)
    incident={}
    for i,s in enumerate(scopes):
        for v in s: incident[v]=incident.get(v,0)|(1<<i)
    live=[set() for _ in range(1<<n)]
    cost=[0]*(1<<n)
    full=(1<<n)-1
    for bits in range(1,1<<n):
        if bits.bit_count()==1:
            live[bits]=set(scopes[bits.bit_length()-1])
            continue
        live[bits]={v for v, m in incident.items() if m&bits and m&(full^bits)}
        best=math.inf
        a=(bits-1)&bits
        while a:
            b=bits^a
            if a<b: best=min(best,cost[a]+cost[b]+math.prod(dims[v] for v in live[a]|live[b]))
            a=(a-1)&bits
        cost[bits]=best
    return cost[full]


def order_picture(plans,costs):
    fig, axes=plt.subplots(2,1,figsize=(13,8),layout="constrained")
    for ax,(name,p) in zip(axes,plans.items()):
        c=costs[name];n=len(p["inputs"]);children={n+i:(s["left"],s["right"]) for i,s in enumerate(p["steps"])}
        positions={}; leaves=[]
        def place(node):
            if node<n:
                xy=(len(leaves),0);leaves.append(node)
            else:
                a,b=children[node];x1,y1=place(a);x2,y2=place(b)
                xy=((x1+x2)/2,max(y1,y2)+1)
            positions[node]=xy
            return xy
        place(n+len(p["steps"])-1)
        for node,(a,b) in children.items():
            for child in (a,b):
                x,y=positions[node];u,v=positions[child]
                ax.plot([x,u],[y,v],color="#8ca2ad",lw=1)
        peak=max(c["steps"],key=lambda s:s["entries"])
        for node,(x,y) in positions.items():
            label=f"f{node}: {''.join(NAMES[j] for j in p['inputs'][node])}" if node<n else (
                f"{''.join(NAMES[j] for j in p['steps'][node-n]['retain']) or 'scalar'} · "
                f"{c['steps'][node-n]['entries']} entries")
            highlight=node==peak["result"]
            ax.text(x,y,label,ha="center",va="center",fontsize=8.5,zorder=3,
                    bbox=dict(boxstyle="round,pad=.22",fc="#fbe4d0" if highlight else "#edf4f7",
                              ec=ORANGE if highlight else "#b2c8d2"))
        ax.set(xlim=(-.6,n-.4),ylim=(-.65,max(y for x,y in positions.values())+.6))
        ax.axis("off")
        ax.set_title(f"{name.capitalize()}: {c['work']:,} work terms, largest tensor {c['largest_tensor']} entries",
                     loc="left",weight="bold")
    fig.suptitle("Same AD scalar, different binary contraction trees\nOrange highlights a largest generated tensor")
    save(fig,"orders.png")


def bp_trace(bp,target,select):
    h=bp["history"]
    assert len(h)==bp["sweeps"]+1 and h[-1]["beliefs"]==bp["marginals"]
    assert h[-1]["residual"]==bp["residual"]
    assert bp["converged"]==(bp["residual"]<=bp["tolerance"])
    assert bp["converged"] or bp["sweeps"]==bp["max_sweeps"]
    trace=[]
    for i,s in enumerate(h):
        assert s["sweep"]==i and s["cumulative_work"]==(i+1)*bp["work_per_proposal"]
        assert np.max(np.abs(np.asarray(s["beliefs"]).sum(axis=1)-1))<1e-12
        err=float(np.max(np.abs(np.asarray(s["beliefs"])[list(select)]-target)))
        trace.append([i,s["residual"],err,s["cumulative_work"]])
    return trace


def trace_axes(axes, trace, label, color):
    x=np.array(trace)
    axes[0].semilogy(x[:,0],np.maximum(x[:,1],1e-17),label=label,color=color)
    axes[1].semilogy(x[:,0],np.maximum(x[:,2],1e-17),label=label,color=color)


def main():
    dims, scopes, tables=uai("asia.uai")
    assert scopes==[(0,),(1,0),(2,),(3,2),(4,2),(5,3,1),(6,5),(7,5,4)]
    np.testing.assert_array_equal(tables[1],[[.05,.01],[.95,.99]])
    draw_diagram(scopes)
    evidence={}
    for case,mass in zip(CASES,[1,.004501375,.00098822675]):
        run=read(f"artifacts/{case}/run.json"); e=run["examples"]
        joint=independent_joint(scopes,tables,run["observations"])
        expected_mass=joint.sum()
        assert abs(expected_mass-mass)<1e-14
        expected=np.array([joint.sum(axis=tuple(j for j in range(8) if j!=i))/expected_mass for i in range(8)])
        expected_tl=joint.sum(axis=(0,2,4,5,6,7))/expected_mass
        for which in ("enumeration","tenferro"):
            result=run[which]
            assert abs(result["evidence_mass"]/expected_mass-1)<1e-10
            np.testing.assert_allclose([result["marginals"][str(i)] for i in range(8)],expected,atol=1e-10,rtol=0)
            np.testing.assert_allclose(np.reshape(result["joint_TL_row_major"],(2,2)),expected_tl,atol=1e-10,rtol=0)
        a,b=np.array([[1,2],[3,4]]),np.array([[2,0],[1,2]])
        ab=np.einsum("ij,jk->ik",a,b)
        tr=np.einsum("ij,ji->",a,b)
        z=sum(math.exp(.5*(x*y+y*w+w*x)) for x,y,w in itertools.product([-1,1],repeat=3))
        matrix_err=float(np.max(np.abs(ab.ravel()-e["AB_row_major"])))
        trace_err=float(abs(tr-e["trace_AB"])); z_err=abs(z-e["ising_tenferro"])
        assert max(matrix_err,trace_err,z_err)<1e-12
        run["numpy_check"]={"AB":ab.tolist(),"trace":float(tr),"ising_enumeration":z,
                            "matrix_max_error":matrix_err,"trace_error":trace_err,"ising_error":z_err}
        bp=read(f"artifacts/bp-{case}/run.json")["bp"]
        trace=bp_trace(bp,expected,range(8))
        evidence[case]={"exact":run,"bp":{k:v for k,v in bp.items() if k!="history"},"trace":trace}
        print(case,"mass",mass,"marginal error",run["marginal_max_absolute_error"],
              "relative evidence error",run["evidence_relative_error"],"BP",trace[-1],
              "AB NumPy/Tenferro",ab.tolist(),e["AB_row_major"],"traces",tr,e["trace_AB"],
              "Ising",z,e["ising_tenferro"],"errors",matrix_err,trace_err,z_err)
    d0,d1,d2=[np.array([evidence[c]["exact"]["tenferro"]["marginals"][v][0] for v in ("1","3","4")]) for c in CASES]
    assert np.all(d2[:2]>d1[:2]) and d2[2]<d1[2]
    adx=evidence["ADX"]["exact"]
    assert abs(adx["tenferro"]["joint_TL_row_major"][0]-.022214)<1e-6
    assert abs(adx["product_TL_row_major"][0]-.174026)<1e-6
    write("asia.json",evidence)

    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    for i,label in enumerate(["Tuberculosis","Lung cancer","Bronchitis"]):
        color=[BLUE,ORANGE,GREEN][i]
        ax.plot(CASES,[x[i] for x in (d0,d1,d2)],"-o",color=color,label=label)
        ax.scatter(CASES,[evidence[c]["exact"]["enumeration"]["marginals"][str((1,3,4)[i])][0] for c in CASES],
                   marker="x",s=70,c="#16181c",zorder=5,label="Enumeration (overlaid)" if i==0 else None)
    ax.set(ylabel="P(disease present | observations)",ylim=(0,1),title="ASIA: posterior changes with evidence")
    ax.legend();ax.grid(axis="y",alpha=.22)
    save(fig,"posteriors.png")

    fig,axes=plt.subplots(3,3,figsize=(10,9),layout="constrained")
    for i,c in enumerate(CASES):
        e=evidence[c]["exact"]
        for j,(title,data) in enumerate(zip(("Joint","Product of marginals","Dependence: joint − product"),
            [e["tenferro"]["joint_TL_row_major"],e["product_TL_row_major"],e["dependence_TL_row_major"]])):
            ax=axes[i,j];v=np.reshape(data,(2,2));limit=.18 if j==2 else 1
            ax.imshow(v,cmap="RdBu_r" if j==2 else "Blues",vmin=-limit if j==2 else 0,vmax=limit)
            for x,y in itertools.product(range(2),repeat=2):
                label=f"{v[y,x]:+.4f}" if j==2 and abs(v[y,x])>=.00005 else f"{abs(v[y,x]):.4f}"
                ax.text(x,y,label,ha="center",va="center",
                        color="white" if abs(v[y,x])>.68*limit else "#152531",fontsize=11)
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["L=0","L=1"],yticklabels=["T=0","T=1"])
            if i==0:ax.set_title(title,fontsize=10)
            if j==0:ax.set_ylabel(c,weight="bold")
    fig.suptitle("ASIA joint vs. independence assumption (0 = present)")
    save(fig,"joint.png")

    plans={name:read(f"runs/{name}/order.json") for name in ("random","greedy")}
    costs={name:read(f"runs/{name}/complexity.json") for name in plans}
    sizes={name:recount(plans[name],dims,costs[name]) for name in plans}
    assert costs["random"]==read("runs/analysis/complexity.json")
    assert not (ROOT/"runs/analysis/run.json").exists()
    assert plans["greedy"]==read("runs/replay/order.json")
    assert costs["greedy"]==read("runs/replay/complexity.json")
    for name in ("random","greedy","replay"):
        x=read(f"runs/{name}/run.json")
        assert abs(x["tenferro"]["evidence_mass"]/.004501375-1)<1e-10
    minima={c:minimum_work(read(f"artifacts/{c}/order.json")["inputs"],dims) for c in CASES}
    assert minima=={"prior":44,"AD":50,"ADX":52}
    write("orders.json",{"plans":plans,"complexities":costs,"recounted":sizes,"exhaustive_minimum_work":minima})
    for name in plans:
        shutil.copyfile(ROOT/f"runs/{name}/order.json",OUT/f"ad-{name}-plan.json")
    order_picture(plans,costs)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for c,color in zip(CASES,[BLUE,ORANGE,GREEN]):
        trace_axes(axes,evidence[c]["trace"],c,color)
    axes[0].axhline(1e-10,color="0.5",ls=":",label="Stopping tolerance")
    for ax,title in zip(axes,["Undamped fixed-point residual","Maximum all-node error vs. exact"]):
        ax.set(xlabel="Applied sweeps",ylabel="Maximum absolute difference",title=title);ax.grid(alpha=.22);ax.legend()
    save(fig,"asia-bp.png")

    bench_dims,_,_=uai("promedus/Promedus_14.uai")
    pr_file=(ROOT/"promedus/Promedus_14.uai.PR").read_text().split()
    assert pr_file[0]=="PR"
    pr=float(pr_file[1]); f=iter((ROOT/"promedus/Promedus_14.uai.MAR").read_text().split())
    assert next(f)=="MAR"
    mar=[np.array([float(next(f)) for _ in range(int(next(f)))]) for _ in range(int(next(f)))]
    assert len(mar)==len(bench_dims)==414 and next(f,None) is None
    target=np.array([mar[v] for v in IDS])
    benchmark={}
    for order in ("greedy","treesa"):
        base=f"artifacts/promedus-{order}"
        run=read(f"{base}/run.json");p=read(f"{base}/order.json")
        costs=read(f"{base}/complexity.json");metrics=recount(p,bench_dims,costs)
        qs=read(f"{base}/query-complexities.json");qp=read(f"{base}/query-plans.json")
        for q in qs:recount(qp[q],bench_dims,qs[q])
        run["mar_max_abs"]=float(np.max(np.abs(np.array([run["marginals"][str(v)] for v in IDS])-target)))
        run["pr_abs"]=abs(run["log10_partition"]-pr)
        assert run["mar_max_abs"]<1e-5 and run["pr_abs"]<1e-4 and len(run["observations"])==9
        benchmark[order]={"run":run,"scalar_cost":metrics,
                          "sum_query_work":sum(q["work"] for q in qs.values()),
                          "query_cost":{v:{k:val for k,val in row.items() if k!="steps"} for v,row in qs.items()}}
        shutil.copyfile(ROOT/f"{base}/order.json",OUT/f"promedus-{order}-plan.json")
        shutil.copyfile(ROOT/f"{base}/complexity.json",OUT/f"promedus-{order}-complexity.json")
    a,b=(benchmark[o]["run"] for o in ("greedy","treesa"))
    assert abs(a["log10_partition"]-b["log10_partition"])<1e-10
    for v in IDS:
        np.testing.assert_allclose(a["marginals"][str(v)],b["marginals"][str(v)],atol=1e-10,rtol=0)
    bp=read("artifacts/promedus-bp/run.json")["bp"]
    trace=bp_trace(bp,target,IDS)
    assert not bp["converged"] and trace[-1][2]>.1
    benchmark.update({"published_PR":pr,"published_MAR":{v:mar[v].tolist() for v in IDS},
                      "bp":{k:v for k,v in bp.items() if k!="history"},"bp_trace":trace})
    write("promedus.json",benchmark)
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.2),layout="constrained")
    for ax,key,title in zip(axes,["work","largest_generated"],["Scalar contraction work","Largest generated tensor"]):
        vals=[benchmark[o]["scalar_cost"][key] for o in ("greedy","treesa")]
        bars=ax.bar(["Greedy","Time-only TreeSA"],vals,color=[ORANGE,BLUE])
        ax.set_yscale("log");ax.bar_label(bars,labels=[f"{x:,}" for x in vals],padding=4)
        ax.set_ylim(min(vals)*.45,max(vals)*2)
        ax.set(title=title,ylabel="Terms" if key=="work" else "Entries")
    fig.suptitle("Promedus_14: orders recounted from the saved plans")
    save(fig,"promedus-cost.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    trace_axes(axes,trace,"BP",BLUE)
    axes[0].axhline(1e-10,ls=":",color=ORANGE,label="Stopping tolerance")
    for ax,title in zip(axes,["Undamped residual","Maximum selected-query error vs. MAR"]):
        ax.set(xlabel="Applied sweeps",ylabel="Maximum absolute difference",title=title)
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Promedus_14: BP is inaccurate and still above tolerance at the sweep limit")
    save(fig,"promedus-bp.png")

    inputs=["asia.uai"]+[f"promedus/Promedus_14.{s}" for s in ("uai","uai.evid","uai.PR","uai.MAR")]
    write("input-hashes.json",{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs})
    import os
    cargo=Path(os.environ.get("CARGO",Path.home()/".cargo"/"bin"/"cargo"))
    rustc=cargo.with_name("rustc")
    environment={"rustc":subprocess.check_output([rustc,"+1.96.0","--version"],text=True).strip(),
                 "cargo":subprocess.check_output([cargo,"+1.96.0","--version"],text=True).strip(),
                 "python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,
                 "system":platform.platform(),"tenferro":"0.11.0 native CPU without autodiff or CUDA","omeco":"0.2.6"}
    write("environment.json",environment)
    print("ASIA minimum work",minima,"AD plan costs",sizes)
    print("Promedus",[(o,benchmark[o]) for o in ("greedy","treesa")])
    print("Promedus BP final [sweep,residual,error,work]:",trace[-1])
    print("All independent numerical and plan checks passed.")


if __name__=="__main__":
    main()
