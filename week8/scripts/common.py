"""CPU defaults, explicit provenance, figures, and reproducible statistical helpers."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random,lax
from scipy.special import softmax,logsumexp
from scipy.stats import chi2

jax.config.update('jax_default_matmul_precision','highest')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';ART=ROOT/'artifacts';EVIDENCE=ROOT/'evidence'
for directory in (DATA,ART,EVIDENCE):directory.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def sha256(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def environment():
    import scipy,sentencepiece,tokenizers
    return {'python':platform.python_version(),'platform':platform.platform(),'jax':jax.__version__,
            'numpy':np.__version__,'scipy':scipy.__version__,'sentencepiece':sentencepiece.__version__,
            'tokenizers':tokenizers.__version__,'devices':[str(d) for d in jax.devices()],
            'cpu_affinity':sorted(os.sched_getaffinity(0))}

def savefig(fig,name):
    fig.savefig(EVIDENCE/name,dpi=155,bbox_inches='tight',facecolor='white');plt.close(fig)

def pooled_chisquare(observed,probabilities):
    """Pool rare expected categories; preserve all mass and sum conditional dfs."""
    observed=np.atleast_2d(observed);probabilities=np.atleast_2d(probabilities)
    statistic=0.;df=0;minimum=[]
    for counts,prob in zip(observed,probabilities):
        assert abs(prob.sum()-1)<1e-10
        expected=counts.sum()*prob
        positive=expected>0
        assert np.sum(counts[~positive])==0
        common=expected>=5
        obs=counts[common].astype(float).tolist();exp=expected[common].tolist()
        rare=positive&~common
        if np.any(rare):
            o=float(counts[rare].sum());e=float(expected[rare].sum())
            if e>=5:obs.append(o);exp.append(e)
            elif exp:
                i=int(np.argmin(exp));obs[i]+=o;exp[i]+=e
            else:raise ValueError('Insufficient observations for Pearson test')
        obs=np.asarray(obs);exp=np.asarray(exp)
        statistic+=float(np.sum((obs-exp)**2/exp));df+=len(exp)-1
        minimum.append(float(exp.min()))
    return {'statistic':statistic,'degrees_of_freedom':df,'p_value':float(chi2.sf(statistic,df)),
            'minimum_expected_bin':min(minimum),'rare_bins':'pooled; an undersized pooled bin joins the smallest adequate bin'}
