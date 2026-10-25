"""Three-hidden-layer time-conditioned SiLU MLP and the specified Adam optimizer."""
import functools
from common import *
import jax.numpy as jnp
from jax import random,lax

CENTERS=jnp.array([[3.,0.],[0.,3.],[-3.,0.],[0.,-3.]],jnp.float32)
BETA=jnp.linspace(.0001,.02,1000,dtype=jnp.float32)
ALPHA=1-BETA
ABAR=jnp.concatenate((jnp.ones(1,jnp.float32),jnp.cumprod(ALPHA)))

def target(key,n):
    component_key,noise_key=random.split(key)
    components=random.randint(component_key,(n,),0,4)
    return CENTERS[components]+.3*random.normal(noise_key,(n,2)),components

def initialize(key):
    widths=[34,128,128,128,2]
    keys=random.split(key,4)
    return tuple((random.normal(k,(a,b),dtype=jnp.float32)/jnp.sqrt(float(a)),
                  jnp.zeros(b,jnp.float32)) for k,a,b in zip(keys,widths[:-1],widths[1:]))

def features(time):
    frequency=10000.**(-jnp.arange(16,dtype=jnp.float32)/16)
    phase=time[...,None]*frequency
    return jnp.concatenate((jnp.sin(phase),jnp.cos(phase)),axis=-1)

def predict(params,x,time):
    time=jnp.broadcast_to(jnp.asarray(time,jnp.float32),x.shape[:-1])
    h=jnp.concatenate((x,features(time)),axis=-1)
    for w,b in params[:-1]:h=jax.nn.silu(h@w+b)
    w,b=params[-1]
    return h@w+b

def training_batch(key,kind,n=512):
    data_key,noise_key,time_key=random.split(key,3)
    clean,_=target(data_key,n)
    noise=random.normal(noise_key,(n,2),dtype=jnp.float32)
    if kind=="diffusion":
        level=random.randint(time_key,(n,),1,1001)
        a=ABAR[level,None]
        x=jnp.sqrt(a)*clean+jnp.sqrt(1-a)*noise
        return x,level.astype(jnp.float32)-1,noise
    time=random.uniform(time_key,(n,),dtype=jnp.float32)
    x=(1-time[:,None])*clean+time[:,None]*noise
    return x,1000*time,noise-clean

def loss(params,batch):
    x,time,truth=batch
    return jnp.mean((predict(params,x,time)-truth)**2)

@functools.partial(jax.jit,static_argnames=("kind",))
def train_chunk(params,m,v,key,step,kind):
    def body(carry,_):
        par,first,second,rng,count=carry
        rng,batch_key=random.split(rng)
        value,grad=jax.value_and_grad(loss)(par,training_batch(batch_key,kind))
        count=count+1
        first=jax.tree.map(lambda a,g:.9*a+.1*g,first,grad)
        second=jax.tree.map(lambda a,g:.999*a+.001*g*g,second,grad)
        lr=.0005*(1+jnp.cos(jnp.pi*(count-1)/24999))
        par=jax.tree.map(lambda w,a,b:w-lr*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),
                         par,first,second)
        return (par,first,second,rng,count),value
    (params,m,v,key,step),values=lax.scan(body,(params,m,v,key,step),None,length=50)
    return params,m,v,key,step,values

def save_params(path,params):
    np.savez(path,**{f"w{i}":np.asarray(w) for i,(w,_) in enumerate(params)},
             **{f"b{i}":np.asarray(b) for i,(_,b) in enumerate(params)})

def load_params(path):
    with np.load(path) as f:return tuple((jnp.asarray(f[f"w{i}"]),jnp.asarray(f[f"b{i}"])) for i in range(4))

def exact_noise(x,level):
    a=ABAR[level]
    mean=jnp.sqrt(a)*CENTERS
    variance=.09*a+1-a
    weights=jax.nn.softmax(-jnp.sum((x[:,None,:]-mean)**2,axis=-1)/(2*variance),axis=-1)
    return jnp.sqrt(1-a)/variance*(x-weights@mean)

def exact_velocity(x,time):
    mean=(1-time)*CENTERS
    variance=.09*(1-time)**2+time*time
    weights=jax.nn.softmax(-jnp.sum((x[:,None,:]-mean)**2,axis=-1)/(2*variance),axis=-1)
    component=-CENTERS+(time-.09*(1-time))/variance*(x[:,None,:]-mean)
    return jnp.sum(weights[...,None]*component,axis=1)
