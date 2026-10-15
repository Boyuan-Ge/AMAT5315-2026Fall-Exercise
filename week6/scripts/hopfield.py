"""Classical asynchronous Hopfield dynamics and modern attention retrieval in JAX."""
import argparse
import functools
import time

from common import *
import jax
import jax.numpy as jnp
from jax import lax, random

jax.config.update("jax_default_matmul_precision", "highest")


def hebbian_numerator(patterns):
    """N*J has integer-valued float32 entries, preserving exact zero-field ties."""
    k = patterns.T@patterns
    return k.at[jnp.diag_indices(k.shape[0])].set(0.)


@jax.jit
def recall(couplings, starts, key):
    """Random-order single-spin updates, batched over independent starting states."""
    batch, n = starts.shape
    fields = starts@couplings
    initial = (starts, fields, key, jnp.int32(0), jnp.ones(batch, bool),
               jnp.zeros(batch,jnp.int32), jnp.zeros(batch,jnp.int32))

    def condition(carry):
        return jnp.any(carry[4]) & (carry[3] < 10000)

    def sweep(carry):
        states, h, rng, sweeps, _, flips_total, first_stable = carry
        rng, order_key = random.split(rng)
        keys = random.split(order_key, batch)
        orders = jax.vmap(lambda k: random.permutation(k, n))(keys)
        row = jnp.arange(batch)
        def update(i, shf):
            s, field, flips = shf
            index = orders[:,i]
            old = s[row,index]
            local = field[row,index]
            new = jnp.where(local>0,1.,jnp.where(local<0,-1.,old))
            delta = new-old
            s = s.at[row,index].set(new)
            field = field + delta[:,None]*couplings[index,:]
            return s,field,flips+(delta!=0)
        states,h,flips = lax.fori_loop(0,n,update,(states,h,jnp.zeros(batch,jnp.int32)))
        first_stable = jnp.where((first_stable==0)&(flips==0),sweeps+1,first_stable)
        return states,h,rng,sweeps+1,flips>0,flips_total+flips,first_stable

    states, fields, _, sweeps, changing, flips, first_stable = lax.while_loop(condition,sweep,initial)
    return states, first_stable, jnp.any(changing), flips


@functools.partial(jax.jit, static_argnames=("n",))
def random_trial(key, p, n):
    key, pattern_key, recall_key = random.split(key,3)
    maximum = n//4
    patterns = 2*random.bernoulli(pattern_key,.5,(maximum,n)).astype(jnp.float32)-1
    mask = jnp.arange(maximum)<p
    k = hebbian_numerator(patterns*mask[:,None])
    local = patterns@k
    unstable = jnp.sum((patterns*local<0)&mask[:,None])/(p*n)
    indices = jnp.arange(10)%p
    states,sweeps,not_converged,flips = recall(k,patterns[indices],recall_key)
    overlap = jnp.mean(states*patterns[indices],axis=1)
    return unstable,overlap,sweeps,not_converged,flips


def random_capacity(selected=None):
    sizes = [100,200,400,800] if selected is None else [selected]
    out = ART/"hopfield"
    out.mkdir(exist_ok=True)
    for n in sizes:
        rows=[]
        start=time.perf_counter()
        for tick in range(5,26):
            p=round(n*tick/100)
            trials=[]
            for repeat in range(20):
                seed=6_603_000+n*10000+tick*100+repeat
                unstable,overlap,sweeps,not_converged,flips=random_trial(random.PRNGKey(seed),p,n)
                q=min(10,p)
                assert not bool(not_converged)
                trials.append({"seed":seed,"unstable_fraction":float(unstable),
                               "overlaps":np.asarray(overlap[:q]).tolist(),
                               "sweeps":np.asarray(sweeps[:q]).tolist(),
                               "flips":np.asarray(flips[:q]).tolist()})
            unstable=np.array([t["unstable_fraction"] for t in trials])
            set_mean=np.array([np.mean(t["overlaps"]) for t in trials])
            success=np.array([np.mean(np.array(t["overlaps"])>=.9) for t in trials])
            rows.append({"n":n,"p":p,"alpha":p/n,"trials":trials,
                         "unstable_mean":float(unstable.mean()),
                         "unstable_se":float(unstable.std(ddof=1)/np.sqrt(20)),
                         "overlap_mean":float(set_mean.mean()),
                         "overlap_se":float(set_mean.std(ddof=1)/np.sqrt(20)),
                         "success_mean":float(success.mean()),
                         "success_se":float(success.std(ddof=1)/np.sqrt(20))})
            print(f"Random Hopfield N={n} alpha={p/n:.2f}: unstable={unstable.mean():.5f} overlap={set_mean.mean():.4f}",flush=True)
            write_json(out/f"random-N{n}.json",{"n":n,"pattern_sets":20,"rows":rows,
                       "seconds":time.perf_counter()-start,"seed_rule":"6603000+N*10000+alpha_tick*100+repeat",
                       "environment":environment(),
                       "zero_field":"unchanged","updates":"random-order asynchronous; until a zero-flip sweep"})
        summarize_random([n])


def summarize_random(sizes=(100,200,400,800)):
    for n in sizes:
        report=json.loads((ART/"hopfield"/f"random-N{n}.json").read_text())
        for row in report["rows"]:
            row.pop("trials")
        write_json(EVIDENCE/f"hopfield-N{n}.json",report)


@jax.jit
def modern_update(patterns, queries, beta):
    weights = jax.nn.softmax(beta*(queries@patterns.T),axis=-1)
    return weights@patterns,weights


def attention_update(patterns,queries,beta):
    return jax.nn.dot_product_attention(queries[:,None,:],patterns[:,None,:],
                                       patterns[:,None,:],scale=beta,implementation="xla")[:,0,:]


def modern_experiment():
    x,_,_,_=load_mnist()
    spins=jnp.asarray(2*(x[:1000].reshape(-1,784)>=128).astype(np.float32)-1)
    out=ART/"modern"
    out.mkdir(exist_ok=True)
    rows=[]
    max_error=0.
    start=time.perf_counter()
    for p in (3,10,30,100,300,1000):
        patterns=spins[:p]
        target_indices=np.arange(100)%p
        targets=patterns[target_indices]
        rng=random.fold_in(random.PRNGKey(6604),p)
        lower_key,recall_key=random.split(rng)
        random_lower=2*random.bernoulli(lower_key,.5,(100,392)).astype(jnp.float32)-1
        queries=targets.at[:,392:].set(random_lower)
        couplings=hebbian_numerator(patterns)
        classical,sweeps,bad,_=recall(couplings,queries,recall_key)
        assert not bool(bad)
        def score(values):
            signed=jnp.where(values>=0,1.,-1.)
            overlaps=jnp.mean(signed*targets,axis=1)
            nearest=jnp.argmax(signed@patterns.T,axis=1)
            success=(overlaps>=.95)&(nearest==target_indices)
            return float(jnp.mean(success)),np.asarray(overlaps),np.asarray(nearest)
        classical_success,classical_overlap,_=score(classical)
        modern_values=[]
        weights_values=[]
        betas=(.001,.003,.01,.03,.1,.3,1.)
        for beta in betas:
            modern,weights=modern_update(patterns,queries,beta)
            attended=attention_update(patterns,queries,beta)
            error=float(jnp.max(jnp.abs(modern-attended)))
            max_error=max(error,max_error)
            success,overlap,nearest=score(modern)
            modern_values.append(np.asarray(modern))
            weights_values.append(np.asarray(weights))
            rows.append({"p":p,"beta":beta,"modern_success":success,
                         "classical_success":classical_success,"attention_max_error":error,
                         "mean_modern_overlap":float(overlap.mean()),
                         "mean_classical_overlap":float(classical_overlap.mean())})
            print(f"Modern P={p:4d} beta={beta:g}: success={success:.2f} classical={classical_success:.2f} attention_error={error:.2g}",flush=True)
            if beta==.3:assert success>=.9,(p,success)
        assert classical_success==0,(p,classical_success)
        unstable=float(jnp.mean(patterns*(patterns@couplings)<0))
        np.savez_compressed(out/f"P{p}.npz",targets=np.asarray(targets),queries=np.asarray(queries),
                            classical=np.asarray(classical),modern=np.stack(modern_values),
                            weights=np.stack(weights_values),target_indices=target_indices,
                            sweeps=np.asarray(sweeps),betas=np.array(betas))
        print(f"P={p} uncorrupted stored-digit unstable fraction={unstable:.4f}",flush=True)
    assert max_error<1e-5,max_error
    report={"seed":6604,"queries_per_load":100,"stored_order":"first P training images",
            "lower_half":"independent fair random spins","betas":list(betas),
            "rows":rows,"attention_max_error":max_error,"seconds":time.perf_counter()-start,
            "success":"signed overlap >= 0.95 and target is nearest stored digit",
            "environment":environment()}
    write_json(out/"metrics.json",report)
    write_json(EVIDENCE/"modern-metrics.json",report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=["random","modern","all"],default="all")
    parser.add_argument("--n",type=int,choices=[100,200,400,800])
    args=parser.parse_args()
    if args.part in ("random","all"):random_capacity(args.n)
    if args.part in ("modern","all"):modern_experiment()


if __name__=="__main__":main()
