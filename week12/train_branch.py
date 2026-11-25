"""Train the shared per-candidate branch predictor specified in train_branch.design.toml."""
import argparse
import json
import math
import time
from pathlib import Path

import jax
from jax import numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

SEED, EPOCHS, BATCH = 2026, 120, 64
DECAY, RATE = 2e-4, 4e-3


def gather(instances):
    rows = []
    for graph in instances:
        assert len(graph["nodes"]) <= 80
        for row in graph["nodes"]:
            c = row["candidates"]
            assert len(c) >= 2
            assert len({p["spin"] for p in c}) == len(c)
            score = np.array([p["score"] for p in c], dtype=np.float64)
            for p in c:
                assert math.isclose(math.prod(max(v, 1e-6) for v in p["child_gains"]),
                                    p["score"], rel_tol=1e-12)
                assert p["pruned_children"] == p["child_gains"].count(1e6)
            top = score.max()
            hot = top-score <= 1e-9*max(1, abs(top))
            assert row["selected_spin"] in {p["spin"] for p, selected in zip(c, hot) if selected}
            features = np.array([p["features"] for p in c], dtype=np.float64)
            assert features.shape == (len(c), 12) and np.isfinite(features).all()
            rows.append((graph["id"], row["node"], c, features, hot/hot.sum()))
    return rows


def batch_table(rows, width, mean, std):
    x = np.zeros((len(rows), width, 12))
    y = np.zeros((len(rows), width))
    mask = np.zeros((len(rows), width), dtype=bool)
    for i, (_, _, _, f, t) in enumerate(rows):
        n = len(f)
        x[i, :n] = (f-mean)/std
        y[i, :n] = t
        mask[i, :n] = True
    return tuple(jnp.asarray(v) for v in (x, y, mask))


def logits(params, x):
    w1, b1, w2, b2 = params
    return jnp.tanh(x @ w1 + b1) @ w2 + b2


@jax.jit
def cross_entropy(params, x, y, mask):
    scores = jnp.where(mask, logits(params, x), -jnp.inf)
    log_p = jax.nn.log_softmax(scores, axis=-1)
    return -(jnp.where(mask, log_p, 0.0)*y).sum(-1).mean()


def loss(params, *batch):
    return cross_entropy(params, *batch) + DECAY*(jnp.square(params[0]).sum()+jnp.square(params[2]).sum())


@jax.jit
def update(params, m, v, step, batch):
    grad = jax.grad(loss)(params, *batch)
    m = jax.tree.map(lambda a,g: .9*a+.1*g, m, grad)
    v = jax.tree.map(lambda a,g: .999*a+.001*g*g, v, grad)
    params = jax.tree.map(
        lambda w,a,b: w-RATE*(a/(1-.9**step))/(jnp.sqrt(b/(1-.999**step))+1e-8),
        params, m, v)
    return params, m, v


def train(data):
    train_ids = {g["id"] for g in data["train"]}
    val_ids = {g["id"] for g in data["validation"]}
    fingerprint = lambda g: json.dumps([g["fields"], g["bonds"]], separators=(",", ":"))
    assert train_ids == set(range(36)) and val_ids == set(range(36,48))
    assert len(data["train"]) == 36 and len(data["validation"]) == 12
    assert not train_ids & val_ids
    assert not {fingerprint(g) for g in data["train"]} & {fingerprint(g) for g in data["validation"]}
    tr, va = gather(data["train"]), gather(data["validation"])
    features = np.concatenate([r[3] for r in tr])
    mean, std = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-6)
    width = max(len(r[3]) for r in tr+va)
    train_batch = batch_table(tr, width, mean, std)
    val_batch = batch_table(va, width, mean, std)
    k1, k2 = jax.random.split(jax.random.key(SEED))
    params = (jax.random.normal(k1,(12,48),dtype=jnp.float64)/math.sqrt(12),
              jnp.zeros(48,dtype=jnp.float64),
              jax.random.normal(k2,(48,),dtype=jnp.float64)/math.sqrt(48),
              jnp.array(0.,dtype=jnp.float64))
    m = v = jax.tree.map(jnp.zeros_like,params)
    rng = np.random.default_rng(SEED)
    history, best, opt, step = [], None, None, 0
    for epoch in range(1, EPOCHS+1):
        order = rng.permutation(len(tr))
        for start in range(0,len(order),BATCH):
            ids = order[start:start+BATCH]
            step += 1
            params,m,v=update(params,m,v,step,tuple(x[ids] for x in train_batch))
        t = float(cross_entropy(params,*train_batch))
        z = float(cross_entropy(params,*val_batch))
        assert math.isfinite(t) and math.isfinite(z)
        history.append({"epoch":epoch,"train_ce":t,"val_ce":z})
        if (epoch==1 or epoch%5==0) and (best is None or z<best["val_ce"]):
            opt = params
            best = history[-1]
    probes = []
    # Six distinct, fixed validation nodes spread across the saved sequence.
    for idx in np.linspace(0,len(va)-1,6,dtype=int):
        graph,node,candidates,f,_ = va[idx]
        scores = logits(opt,jnp.asarray((f-mean)/std))
        probes.append({"instance":graph,"node":node,"spins":[c["spin"] for c in candidates],
                       "features":f.tolist(),"logits":scores.tolist(),
                       "probabilities":jax.nn.softmax(scores).tolist()})
    w1,b1,w2,b2 = opt
    return {"schema":1,"jax":jax.__version__,"backend":jax.default_backend(),"precision":"float64",
            "seed":SEED,"settings":{"epochs":EPOCHS,"batch_size":BATCH,"l2":DECAY,"learning_rate":RATE},
            "mean":mean.tolist(),"std":std.tolist(),"w1":w1.tolist(),"b1":b1.tolist(),
            "w2":w2.tolist(),"b2":float(b2),"history":history,"best":best,"probes":probes,
            "split":{"train_nodes":len(tr),"validation_nodes":len(va),"train_ids":sorted(train_ids),
                     "validation_ids":sorted(val_ids),"overlap":[]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    start = time.perf_counter()
    result = train(json.loads(args.dataset.read_text()))
    result["training_seconds"] = time.perf_counter()-start
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("jax","backend","split","best","training_seconds")},indent=2))


if __name__ == "__main__":
    main()
