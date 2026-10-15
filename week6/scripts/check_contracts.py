"""Independent small-system checks before full data experiments."""
import itertools

from common import *
import jax
import jax.numpy as jnp
from jax import random
from scipy.special import expit, logsumexp
from rbm import energy, free_energy, cd_surrogate, sample
from hopfield import hebbian_numerator, recall, modern_update, attention_update


def main():
    rng=np.random.default_rng(6610)
    w=rng.normal(0,.2,(4,3)).astype(np.float32)
    a=rng.normal(0,.1,4).astype(np.float32)
    b=rng.normal(0,.1,3).astype(np.float32)
    params=tuple(jnp.asarray(x) for x in (w,a,b))
    v=np.array(list(itertools.product((0.,1.),repeat=4)),np.float32)
    h=np.array(list(itertools.product((0.,1.),repeat=3)),np.float32)
    explicit=-logsumexp(-np.asarray(energy(params,v[:,None,:],h[None,:,:])),axis=1)
    free_error=float(np.max(abs(np.asarray(free_energy(params,v))-explicit)))
    assert free_error<1e-6
    pos=v[:8]
    neg=v[8:]
    gradient=jax.grad(cd_surrogate)(params,jnp.asarray(pos),jnp.asarray(neg))
    ph=expit(pos@w+b);nh=expit(neg@w+b)
    analytic=(-(pos.T@ph-neg.T@nh)/len(pos),-np.mean(pos-neg,axis=0),-np.mean(ph-nh,axis=0))
    gradient_error=max(float(np.max(abs(np.asarray(x)-y))) for x,y in zip(gradient,analytic))
    assert gradient_error<1e-6
    sampled,_=sample(params,jnp.asarray(v),random.PRNGKey(17),10,jnp.asarray(v),jnp.array([True,True,False,False]))
    assert set(np.unique(sampled))<={0.,1.}
    assert np.array_equal(np.asarray(sampled)[:,:2],v[:,:2])

    patterns=jnp.asarray(2*rng.integers(0,2,(3,20)).astype(np.float32)-1)
    couplings=hebbian_numerator(patterns)
    assert np.all(np.diag(couplings)==0)
    starts=jnp.asarray(2*rng.integers(0,2,(5,20)).astype(np.float32)-1)
    final,_,bad,_=recall(couplings,starts,random.PRNGKey(28))
    assert not bool(bad)
    assert np.all(np.asarray(final)*(np.asarray(final)@np.asarray(couplings))>=0)
    start_energy=-.5*np.sum(np.asarray(starts)*(np.asarray(starts)@np.asarray(couplings)),axis=1)
    final_energy=-.5*np.sum(np.asarray(final)*(np.asarray(final)@np.asarray(couplings)),axis=1)
    assert np.all(final_energy<=start_energy+1e-6)
    zeros=jnp.zeros((20,20),jnp.float32)
    tie_result=recall(zeros,starts,random.PRNGKey(1))[0]
    assert np.array_equal(tie_result,starts)
    modern,_=modern_update(patterns,starts,.03)
    attention=attention_update(patterns,starts,.03)
    attention_error=float(jnp.max(abs(modern-attention)))
    assert attention_error<1e-5
    wrong_scale=jax.nn.dot_product_attention(starts[:,None,:],patterns[:,None,:],patterns[:,None,:])[:,0,:]
    negative_control=float(jnp.max(abs(wrong_scale-modern)))
    assert negative_control>.01
    report={"free_energy_enumeration_max_error":free_error,
            "cd_gradient_closed_form_max_error":gradient_error,
            "gibbs_binary_and_clamp":True,"hopfield_zero_diagonal":True,
            "recall_final_state_stable":True,"recall_energy_nonincreasing":True,
            "zero_field_unchanged":True,"attention_max_error":attention_error,
            "wrong_attention_scale_negative_control":negative_control}
    write_json(EVIDENCE/"contract-checks.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
