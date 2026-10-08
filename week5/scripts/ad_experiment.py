"""Reproduce the Week 5 pair-potential AD checks and scaling figures."""

import json
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ad"
OUT.mkdir(parents=True, exist_ok=True)


def pair(r):
    a = r ** -6
    b = a * a
    c = b - a
    return 4 * c


def analytic(r):
    return 24 * (r ** -7 - 2 * r ** -13)


def handwritten(r):
    a = r ** -6
    b = a * a
    c = b - a
    energy = 4 * c
    tangent = {"r": 1.0, "a": -6 * r ** -7}
    tangent["b"] = 2 * a * tangent["a"]
    tangent["c"] = tangent["b"] - tangent["a"]
    tangent["U"] = 4 * tangent["c"]
    adjoint = {"U": 1.0, "c": 4.0, "b": 4.0}
    adjoint["a"] = -adjoint["c"] + 2 * a * adjoint["b"]
    adjoint["r"] = adjoint["a"] * (-6 * r ** -7)
    return energy, tangent, adjoint


def cluster_energy(x):
    separation = x[:, None, :] - x[None, :, :]
    r2 = jnp.sum(separation * separation, axis=-1) + jnp.eye(x.shape[0])
    a = r2 ** -3
    u = 4 * (a * a - a)
    return jnp.sum(jnp.triu(u, k=1))


def graph_png(name, jaxpr):
    # Build edges from the actual JAX jaxpr rather than drawing the intended formula.
    graph = jaxpr.jaxpr
    producer = {v: -1 for v in graph.invars}
    depth = {-1: 0}
    edges = set()
    for i, eqn in enumerate(graph.eqns):
        parents = set()
        for v in eqn.invars:
            try:
                if v in producer:
                    parents.add(producer[v])
            except TypeError:  # JAX literals do not have a hashable producer.
                pass
        depth[i] = 1 + max((depth[p] for p in parents), default=0)
        edges.update((p, i) for p in parents)
        for v in eqn.outvars:
            producer[v] = i
    end = len(graph.eqns)
    parents = {producer[v] for v in graph.outvars if v in producer}
    depth[end] = 1 + max(depth[p] for p in parents)
    edges.update((p, end) for p in parents)
    columns = defaultdict(list)
    for node, x in depth.items():
        columns[x].append(node)
    positions = {}
    for x, nodes in columns.items():
        for y, node in zip(np.linspace(-max(2, len(nodes)-1), max(2, len(nodes)-1), len(nodes)), nodes):
            positions[node] = (x * 2.6, y * 1.35)
    fig, ax = plt.subplots(figsize=(max(9, max(depth.values()) * 2.5),
                                    max(5, max(map(len, columns.values())) * 1.2)))
    for start, finish in sorted(edges):
        ax.add_patch(FancyArrowPatch(positions[start], positions[finish],
                                    arrowstyle="-|>", mutation_scale=10, lw=1,
                                    color="#74879d", connectionstyle="arc3,rad=.03"))
    for node, (x, y) in positions.items():
        if node == -1:
            label, color = "input r", "#d6eaff"
        elif node == end:
            label, color = ("dU/dr" if "grad" in name else "energy U"), "#dff5de"
        else:
            eqn = graph.eqns[node]
            label = eqn.primitive.name
            if "y" in eqn.params:
                label += f"\ny={eqn.params['y']}"
            color = "#ffd4d6" if label == "add_any" else "#eff2f6"
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=.4", "facecolor": color,
                      "edgecolor": "#53677d"}, zorder=3)
    ax.set_xlim(-1.5, max(x for x, _ in positions.values()) + 1.5)
    ax.set_ylim(min(y for _, y in positions.values()) - 1.5,
                max(y for _, y in positions.values()) + 1.5)
    ax.axis("off")
    ax.set_title("JAX gradient jaxpr" if "grad" in name else "JAX primal jaxpr",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def main():
    r = 1.3
    energy, tangent, adjoint = handwritten(r)
    grad = float(jax.grad(pair)(r))
    record = {"r": r, "energy": energy, "tangents": tangent,
              "adjoints": adjoint, "jax_grad": grad}
    (OUT / "derivatives.json").write_text(json.dumps(record, indent=2) + "\n")
    assert abs(energy - (-0.6570169144600471)) < 1e-12
    assert abs(adjoint["a"] - (-2.3425903117359734)) < 1e-12
    assert abs(grad - 2.239979929791143) < 1e-12
    print("node check:", record)

    rs = np.linspace(.95, 2.5, 601)
    exact = analytic(rs)
    forward = np.array([handwritten(float(v))[1]["U"] for v in rs])
    reverse = np.array([handwritten(float(v))[2]["r"] for v in rs])
    fd = (np.array([pair(float(v) + 1e-6) for v in rs]) -
          np.array([pair(float(v) - 1e-6) for v in rs])) / (2e-6)
    errs = {"forward": float(np.max(abs(forward-exact))),
            "reverse": float(np.max(abs(reverse-exact))),
            "finite_difference": float(np.max(abs(fd-exact)))}
    assert errs["forward"] < 1e-12 and errs["reverse"] < 1e-12
    assert errs["finite_difference"] > max(errs["forward"], errs["reverse"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for label, values in [("analytic", exact), ("forward", forward),
                          ("reverse", reverse), ("centered FD", fd)]:
        ax[0].plot(rs, values, label=label)
    for label, values in [("forward", forward), ("reverse", reverse), ("FD", fd)]:
        ax[1].semilogy(rs, np.maximum(abs(values-exact), 1e-17), label=label)
    ax[0].set(xlabel="separation r", ylabel="dU/dr", title="Lennard-Jones derivative")
    ax[1].set(xlabel="separation r", ylabel="absolute error", title="Error against analytic derivative")
    for a in ax:
        a.grid(alpha=.2)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "modes.png", dpi=170)
    plt.close(fig)
    graph_png("graph.png", jax.make_jaxpr(pair)(1.3))
    graph_png("grad-graph.png", jax.make_jaxpr(jax.grad(pair))(1.3))

    rows = []
    rng = np.random.default_rng(2026)
    for n in [64, 128, 256, 512, 1024]:
        side = int(np.ceil(n ** (1/3)))
        grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), -1).reshape(-1, 3)[:n]
        coords = jnp.asarray(grid * 2 ** (1/6) + rng.normal(0, .05, (n, 3)))
        flat = coords.reshape(-1)
        f = jax.jit(lambda v: cluster_energy(v.reshape(n, 3)))
        forward_one = jax.jit(lambda v, t: jax.jvp(f, (v,), (t,))[1])
        reverse = jax.jit(jax.grad(f))
        f(flat).block_until_ready()
        reverse(flat).block_until_ready()
        direction = jnp.zeros_like(flat).at[0].set(1.)
        forward_one(flat, direction).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(4):
            f(flat).block_until_ready()
        energy_time = (time.perf_counter() - t0) / 4
        t0 = time.perf_counter()
        rev = reverse(flat).block_until_ready()
        reverse_time = time.perf_counter() - t0
        # Every coordinate is measured, not just a sample direction.
        t0 = time.perf_counter()
        fwd = np.empty(3*n)
        for k in range(3*n):
            direction = jnp.zeros_like(flat).at[k].set(1.)
            fwd[k] = float(forward_one(flat, direction))
        forward_time = time.perf_counter() - t0
        error = float(np.max(abs(fwd - np.asarray(rev)))) / float(np.max(abs(np.asarray(rev))))
        assert error < 1e-12
        rows.append({"n": n, "p": 3*n, "energy_seconds": energy_time,
                     "forward_seconds": forward_time, "reverse_seconds": reverse_time,
                     "forward_ratio": forward_time/energy_time,
                     "reverse_ratio": reverse_time/energy_time, "relative_error": error})
        print("cluster:", rows[-1], flush=True)
    (OUT / "scaling.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert rows[-1]["forward_ratio"] > 100 * rows[-1]["reverse_ratio"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    p = np.array([row["p"] for row in rows])
    ax.loglog(p, [row["forward_ratio"] for row in rows], "o-", label="P JVPs (forward)")
    ax.loglog(p, [row["reverse_ratio"] for row in rows], "s-", label="one VJP (reverse)")
    ax.loglog(p, p/p[0]*rows[0]["forward_ratio"], "k:", label="proportional to P")
    ax.set(xlabel="coordinate inputs P = 3N", ylabel="gradient time / energy time",
           title="Cluster gradient scaling, JAX 64-bit")
    ax.grid(alpha=.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=170)
    plt.close(fig)
    print("maximum derivative errors", errs)


if __name__ == "__main__":
    main()
