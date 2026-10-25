"""Shared CPU defaults, paths, reproducible metadata, and evidence rendering."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax

jax.config.update("jax_default_matmul_precision","highest")
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"
INPUT=ROOT/"inputs"
EVIDENCE=ROOT/"evidence"
for p in (ART,INPUT,EVIDENCE):p.mkdir(parents=True,exist_ok=True)

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")

def environment():
    import scipy
    return {"python":platform.python_version(),"platform":platform.platform(),
            "jax":jax.__version__,"numpy":np.__version__,"scipy":scipy.__version__,
            "devices":[str(d) for d in jax.devices()],"cpu_count":os.cpu_count(),
            "cpu_affinity":sorted(os.sched_getaffinity(0))}

def savefig(fig,name):
    fig.savefig(EVIDENCE/name,dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def sha256(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def show_rgb(ax,image,title=None):
    if image.shape[0]==3:image=image.transpose(1,2,0)
    ax.imshow(np.clip((image+1)/2,0,1),interpolation="nearest")
    if title:ax.set_title(title,fontsize=9)
    ax.set_xticks([]);ax.set_yticks([])
