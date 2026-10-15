"""Shared data readers and plotting helpers. Original split/order are preserved."""
import gzip
import hashlib
import json
import os
import platform
import struct
import tarfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
ART = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"
for path in (ART, EVIDENCE):
    path.mkdir(parents=True, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_mnist():
    result = []
    for split in ("train", "t10k"):
        with gzip.open(INPUT / "mnist" / f"{split}-images-idx3-ubyte.gz", "rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051 and (rows, cols) == (28, 28)
            images = np.frombuffer(f.read(), np.uint8).reshape(count, rows, cols)
        with gzip.open(INPUT / "mnist" / f"{split}-labels-idx1-ubyte.gz", "rb") as f:
            magic, count_y = struct.unpack(">II", f.read(8))
            assert magic == 2049 and count_y == count
            labels = np.frombuffer(f.read(), np.uint8)
            assert len(labels) == count and labels.max() == 9
        result.extend((images, labels))
    assert len(result[0]) == 60000 and len(result[2]) == 10000
    return tuple(result)


def load_cifar():
    with tarfile.open(INPUT / "cifar10" / "cifar-10-binary.tar.gz", "r:gz") as tar:
        arrays = []
        for name in [*(f"data_batch_{i}.bin" for i in range(1, 6)), "test_batch.bin"]:
            f = tar.extractfile(f"cifar-10-batches-bin/{name}")
            records = np.frombuffer(f.read(), np.uint8).reshape(-1, 3073)
            images = records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            arrays.append((images, records[:, 0]))
    train_x = np.concatenate([a[0] for a in arrays[:5]])
    train_y = np.concatenate([a[1] for a in arrays[:5]])
    test_x, test_y = arrays[5]
    assert len(train_x) == 50000 and len(test_x) == 10000
    return train_x, train_y, test_x, test_y


def show_image(ax, image, title=None, spin=False, continuous=False):
    kwargs = {"interpolation": "nearest"}
    if image.ndim == 2:
        kwargs.update(cmap="gray_r", vmin=-1 if spin else 0, vmax=1 if spin or continuous else 255)
    ax.imshow(image, **kwargs)
    if title is not None:
        ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def savefig(fig, name):
    fig.savefig(EVIDENCE / name, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def environment():
    import jax
    import scipy
    return {"python": platform.python_version(), "platform": platform.platform(),
            "jax": jax.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "devices": [str(x) for x in jax.devices()], "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
