"""Download authoritative binary records, validate splits, and plot native pixels."""
import hashlib
import time
import urllib.request

from common import *

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
CIFAR_URL = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"


def download():
    sources = [(f"{MNIST_URL}/{split}-{kind}.gz",
                INPUT / "mnist" / f"{split}-{kind}.gz")
               for split in ("train", "t10k")
               for kind in ("images-idx3-ubyte", "labels-idx1-ubyte")]
    sources.append((CIFAR_URL, INPUT / "cifar10/cifar-10-binary.tar.gz"))
    provenance = []
    for url, dest in sources:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print("Downloading", url, flush=True)
            temp = dest.with_suffix(dest.suffix + ".partial")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response, temp.open("wb") as f:
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
                    temp.replace(dest)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        digest = hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()
        provenance.append({"url": url, "file": str(dest.relative_to(ROOT)),
                           "bytes": dest.stat().st_size, "sha256": digest})
    write_json(EVIDENCE / "dataset-provenance.json", provenance)


def distribution_figure(name, train_x, train_y, test_x, test_y, labels):
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(5, 10, height_ratios=[1, 1, 1, .25, 1.8], hspace=.15)
    for c in range(10):
        indices = np.flatnonzero(train_y == c)[:3]
        for row, index in enumerate(indices):
            show_image(fig.add_subplot(grid[row, c]), train_x[index], labels[c] if row == 0 else None)
    ax = fig.add_subplot(grid[4, :])
    counts = {}
    for offset, split, values in [(-.18, "training", train_y), (.18, "test", test_y)]:
        count = np.bincount(values, minlength=10)
        counts[split] = count.tolist()
        ax.bar(np.arange(10) + offset, count / len(values), width=.36, label=f"{split} (N={len(values):,})")
    ax.set(xticks=np.arange(10), xticklabels=labels, ylabel="relative class frequency",
           title="Every original label is counted; galleries show first 3 stored examples per class")
    ax.legend()
    ax.set_ylim(0, .13)
    fig.suptitle(f"{name}: native-pixel labeled examples and empirical class frequencies", fontsize=15)
    savefig(fig, f"{name.lower()}-distribution.png")
    return counts


def main():
    download()
    x, y, tx, ty = load_mnist()
    mnist = distribution_figure("MNIST", x, y, tx, ty, [str(i) for i in range(10)])
    assert np.argmax(mnist["training"]) == np.argmax(mnist["test"]) == 1
    assert np.argmin(mnist["training"]) == np.argmin(mnist["test"]) == 5
    x, y, tx, ty = load_cifar()
    cifar = distribution_figure("CIFAR10", x, y, tx, ty, CLASSES)
    assert cifar["training"] == [5000]*10 and cifar["test"] == [1000]*10
    write_json(EVIDENCE / "dataset-counts.json",
               {"mnist": mnist, "cifar10": cifar, "pixel_dtype": "uint8",
                "pixel_range": [0, 255], "mnist_shape": [28, 28],
                "cifar10_shape": [32, 32, 3], "original_order_and_splits": True})
    print("Dataset validation passed.", flush=True)


if __name__ == "__main__":
    main()
