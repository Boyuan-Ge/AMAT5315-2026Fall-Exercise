"""The sheet's unclipped DDPM, deterministic DDIM, and backward Euler updates."""
import functools
from network import *
from scipy.spatial.distance import cdist,pdist

@functools.partial(jax.jit,static_argnames=("method","exact"))
def diffusion_sample(params,initial,key,levels,method="ddpm",exact=False):
    def body(carry,pair):
        x,rng=carry
        t,next_t=pair
        eps=exact_noise(x,t) if exact else predict(params,x,t-1)
        if method=="ddim":
            clean=(x-jnp.sqrt(1-ABAR[t])*eps)/jnp.sqrt(ABAR[t])
            nxt=jnp.sqrt(ABAR[next_t])*clean+jnp.sqrt(1-ABAR[next_t])*eps
        else:
            ratio=ABAR[t]/ABAR[next_t]
            mean=(x-(1-ratio)/jnp.sqrt(1-ABAR[t])*eps)/jnp.sqrt(ratio)
            rng,noise_key=random.split(rng)
            z=random.normal(noise_key,x.shape,dtype=jnp.float32)
            nxt=mean+jnp.where(next_t>0,jnp.sqrt(1-ratio),0)*z
        return (nxt,rng),nxt
    (_, _),states=lax.scan(body,(initial,key),(levels[:-1],levels[1:]))
    return jnp.concatenate((initial[None],states),axis=0)

@functools.partial(jax.jit,static_argnames=("exact",))
def flow_sample(params,initial,times,exact=False):
    def body(x,pair):
        t,next_t=pair
        velocity=exact_velocity(x,t) if exact else predict(params,x,1000*t)
        nxt=x-(t-next_t)*velocity
        return nxt,nxt
    _,states=lax.scan(body,initial,(times[:-1],times[1:]))
    return jnp.concatenate((initial[None],states),axis=0)

@jax.jit
def forward_process(initial,key):
    def body(carry,beta):
        x,rng=carry
        rng,k=random.split(rng)
        x=jnp.sqrt(1-beta)*x+jnp.sqrt(beta)*random.normal(k,x.shape)
        return (x,rng),x
    _,states=lax.scan(body,(initial,key),BETA)
    return jnp.concatenate((initial[None],states),axis=0)

def levels_for(steps):
    return jnp.rint(jnp.linspace(1000,0,steps+1)).astype(jnp.int32)

def cluster_stats(points):
    points=np.asarray(points,dtype=np.float64)
    centers=np.asarray(CENTERS,dtype=np.float64)
    assignments=np.argmin(cdist(points,centers),axis=1)
    populations=[];means=[];widths=[]
    for k in range(4):
        group=points[assignments==k]
        populations.append(len(group)/len(points))
        means.append(group.mean(axis=0).tolist() if len(group) else [None,None])
        widths.append(group.std(axis=0).tolist() if len(group) else [None,None])
    valid=all(v is not None for row in means for v in row)
    return {"populations":populations,"centers":means,"widths":widths,
            "max_population_error":float(np.max(abs(np.array(populations)-.25))),
            "max_center_error":float(np.max(np.linalg.norm(np.array(means)-centers,axis=1))) if valid else None,
            "max_width_error":float(np.max(abs(np.array(widths)-.3))) if valid else None}

def energy_distance(x,y,y_within=None):
    # U-statistic: self-distances excluded. It may be slightly negative at the null.
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    within_y=float(pdist(y).mean()) if y_within is None else y_within
    return float(2*cdist(x,y).mean()-pdist(x).mean()-within_y)
