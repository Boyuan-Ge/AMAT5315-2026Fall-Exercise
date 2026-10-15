"""JAX binary RBM: CD-k via differentiated stopped-endpoint free energies."""
import argparse
import functools
import hashlib
import time
import tomllib

from common import *
import jax
import jax.numpy as jnp
from jax import lax, random


def energy(params, v, h):
    w, a, b = params
    return -jnp.sum(v*a, axis=-1) - jnp.sum(h*b, axis=-1) - jnp.sum((v@w)*h, axis=-1)


def free_energy(params, v):
    w, a, b = params
    return -(v@a) - jnp.sum(jax.nn.softplus(v@w+b), axis=-1)


def draw_step(params, v, key):
    w, a, b = params
    key, kh, kv = random.split(key, 3)
    h = random.bernoulli(kh, jax.nn.sigmoid(v@w+b)).astype(jnp.float32)
    next_v = random.bernoulli(kv, jax.nn.sigmoid(h@w.T+a)).astype(jnp.float32)
    return next_v, key


@functools.partial(jax.jit, static_argnames=("steps",))
def sample(params, v, key, steps, observed, mask):
    def body(_, carry):
        bits, rng = draw_step(params, *carry)
        return jnp.where(mask, observed, bits), rng
    return lax.fori_loop(0, steps, body, (v, key))


def cd_surrogate(params, positive, negative):
    negative = lax.stop_gradient(negative)
    return jnp.mean(free_energy(params, positive)-free_energy(params, negative))


@functools.partial(jax.jit, static_argnames=("k",))
def train_epoch(params, batches, key, learning_rate, k):
    def body(carry, v):
        par, rng = carry
        rng, chain_key = random.split(rng)
        vk, _ = sample(par, v, chain_key, k, v, jnp.zeros(v.shape[-1], bool))
        value, grad = jax.value_and_grad(cd_surrogate)(par, v, vk)
        par = jax.tree.map(lambda x, g: x-learning_rate*g, par, grad)
        return (par, rng), value
    (params, _), values = lax.scan(body, (params, key), batches)
    return params, jnp.mean(values)


def initialize(train, config):
    init = config["initialization"]
    key = random.PRNGKey(init["seed"])
    frequency = jnp.clip(jnp.mean(train, axis=0), *init["pixel_frequency_clip"])
    return (random.normal(key, (784, 128), dtype=jnp.float32)*init["weight_std"],
            jnp.log(frequency/(1-frequency)), jnp.zeros(128, jnp.float32))


def save_params(path, params):
    np.savez(path, w=np.asarray(params[0]), a=np.asarray(params[1]), b=np.asarray(params[2]))


def load_params(path):
    with np.load(path) as d:
        return tuple(jnp.asarray(d[name]) for name in ("w", "a", "b"))


def reconstruction(params, test, key):
    v, _ = sample(params, test, key, 1, test, jnp.zeros(784, bool))
    return float(jnp.mean(v != test)), np.asarray(v)


def train_models(config, train, test):
    out = ART / "rbm"
    out.mkdir(exist_ok=True)
    settings = {**config, "hidden_units": 128, "binarization": "uint8 >= 128",
                "backend": "CPU", "environment": environment(),
                "random_schedule": "fold_in(seed, epoch); independent split for permutation and batch-Gibbs key"}
    write_json(out / "settings.json", settings)
    initial = initialize(train, config)
    save_params(out / "initial.npz", initial)
    initial_hash = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in initial)).hexdigest()
    eval_key = random.PRNGKey(config["evaluation"]["seed"])
    train_key = random.PRNGKey(config["training"]["seed"])
    histories = {}
    for k in (1, 10):
        params = initial
        initial_error, _ = reconstruction(params, test, eval_key)
        rows = [{"epoch": 0, "reconstruction_error": initial_error}]
        shuffle_hashes = []
        start = time.perf_counter()
        for epoch in range(config["training"]["epochs"]):
            shuffle_key, sampling_key = random.split(random.fold_in(train_key, epoch))
            permutation = random.permutation(shuffle_key, len(train))
            shuffle_hashes.append(hashlib.sha256(np.asarray(permutation).tobytes()).hexdigest())
            batches = train[permutation].reshape(-1, config["training"]["batch_size"], 784)
            params, surrogate = train_epoch(params, batches, sampling_key,
                                            config["training"]["learning_rate"], k)
            error, _ = reconstruction(params, test, eval_key)
            row = {"epoch": epoch+1, "cd_surrogate": float(surrogate),
                   "reconstruction_error": error}
            rows.append(row)
            print(f"CD-{k}: epoch {epoch+1:2d}/20 surrogate={float(surrogate):.5f} reconstruction={error:.5f}", flush=True)
        runtime = time.perf_counter()-start
        save_params(out / f"cd{k}.npz", params)
        histories[f"cd{k}"] = {"history": rows, "seconds": runtime, "initial_hash": initial_hash,
                                "shuffle_hashes": shuffle_hashes}
        assert rows[-1]["reconstruction_error"] < .5*initial_error
        write_json(out / "training.json", histories)
    assert histories["cd1"]["shuffle_hashes"] == histories["cd10"]["shuffle_hashes"]
    write_json(EVIDENCE / "rbm-training.json", {"settings": settings, "models": histories})
    return histories


def wrong_indices(labels):
    n = len(labels)
    return np.array([next((i+j)%n for j in range(1, n) if labels[(i+j)%n]!=labels[i])
                     for i in range(n)])


def snapshot_chain(params, initial, key, observed, mask):
    snaps = [np.asarray(initial)]
    current = initial
    previous = 0
    for step in (1, 10, 100, 1000):
        current, key = sample(params, current, key, step-previous, observed, mask)
        snaps.append(np.asarray(current))
        previous = step
    return np.stack(snaps)


def sample_models(config, test, labels):
    out = ART / "rbm"
    test = jnp.asarray(test[:64])
    generation_seed = 6605
    start_key, generation_key = random.split(random.PRNGKey(generation_seed))
    generation_start = random.bernoulli(start_key, .5, (64, 784)).astype(jnp.float32)
    comp = config["completion"]
    start_key, comp_key = random.split(random.PRNGKey(comp["seed"]))
    comp_start = random.bernoulli(start_key, .5, (64, comp["chains"], 784)).astype(jnp.float32)
    mask = jnp.arange(784)<392
    wrong = wrong_indices(labels[:64])
    np.savez(out / "matched-starts.npz", generation=np.asarray(generation_start),
             completion=np.asarray(comp_start), wrong_upper_indices=wrong)
    summary = {"generation_seed": generation_seed, "completion_seed": comp["seed"],
               "steps": [0,1,10,100,1000], "wrong_upper_indices": wrong.tolist(),
               "models": {}}
    for k in (1,10):
        params = load_params(out / f"cd{k}.npz")
        start = time.perf_counter()
        _, reconstructed = reconstruction(params, test, random.PRNGKey(config["evaluation"]["seed"]))
        generated = snapshot_chain(params, generation_start, generation_key, generation_start, jnp.zeros(784,bool))
        captures = {}
        model_summary = {}
        for condition, indices in (("true", np.arange(64)), ("wrong", wrong)):
            observed = jnp.repeat(test[indices, None, :], comp["chains"], axis=1).reshape(-1,784)
            initial = jnp.where(mask, observed, comp_start.reshape(-1,784))
            frames = snapshot_chain(params, initial, comp_key, observed, mask)
            assert np.array_equal(frames[..., :392], np.broadcast_to(np.asarray(observed[:,:392]), frames[...,:392].shape))
            frames = frames.reshape(5,64,comp["chains"],784)
            captures[condition] = frames
            truth = np.asarray(test)[None,:,None,392:]
            pred = frames[...,392:]
            intersection = np.sum(pred*truth, axis=-1)
            denominator = np.sum(pred,axis=-1)+np.sum(truth,axis=-1)
            dice = np.divide(2*intersection, denominator, out=np.zeros_like(intersection), where=denominator>0)
            per_digit = dice.mean(axis=2)
            model_summary[condition] = {"mean": per_digit.mean(axis=1).tolist(),
                                       "standard_error": (per_digit.std(axis=1,ddof=1)/8).tolist(),
                                       "per_digit": per_digit.tolist()}
        gap = model_summary["true"]["mean"][-1]-model_summary["wrong"]["mean"][-1]
        model_summary["true_wrong_gap_1000"] = gap
        model_summary["seconds"] = time.perf_counter()-start
        np.savez_compressed(out / f"samples-cd{k}.npz", generation=generated,
                            reconstruction=reconstructed, completion_true=captures["true"],
                            completion_wrong=captures["wrong"])
        summary["models"][f"cd{k}"] = model_summary
        print(f"CD-{k} completion: true={model_summary['true']['mean'][-1]:.4f} wrong={model_summary['wrong']['mean'][-1]:.4f} gap={gap:.4f}", flush=True)
        write_json(out / "sampling.json", summary)
    summarize_sampling()
    for model in summary["models"].values():
        assert model["true_wrong_gap_1000"]>=.1, model
    return summary


def summarize_sampling():
    compact=json.loads((ART/"rbm/sampling.json").read_text())
    for model in compact["models"].values():
        for condition in ("true", "wrong"):
            model[condition].pop("per_digit")
    write_json(EVIDENCE/"rbm-sampling.json",compact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "rbm.toml").read_text())
    x, _, tx, ty = load_mnist()
    train = jnp.asarray((x.reshape(-1,784)>=128).astype(np.float32))
    test = (tx[:64].reshape(-1,784)>=128).astype(np.float32)
    if not args.sample_only:
        train_models(config, train, jnp.asarray(test))
    sample_models(config, test, ty)


if __name__=="__main__":
    main()
