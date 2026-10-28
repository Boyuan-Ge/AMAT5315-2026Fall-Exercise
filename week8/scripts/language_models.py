"""One-token tanh MLP and a biased, pre-norm causal transformer in CPU JAX."""
import functools
from common import *

V=512;WIDTH=96;HEADS=4;CONTEXT=64;LAYERS=2

def initialize(kind,key):
    def normal(shape,scale=.02):
        nonlocal key
        key,k=random.split(key);return scale*random.normal(k,shape,dtype=jnp.float32)
    def linear(a,b,scale=.02):return {'w':normal((a,b),scale),'b':jnp.zeros(b,jnp.float32)}
    def ln():return {'g':jnp.ones(WIDTH,jnp.float32),'b':jnp.zeros(WIDTH,jnp.float32)}
    if kind=='mlp':
        return {'embedding':normal((V,64),.1),'hidden':linear(64,512,64**-.5),'readout':linear(512,V,512**-.5)}
    blocks=[]
    for _ in range(LAYERS):
        blocks.append({'ln1':ln(),'qkv':linear(WIDTH,3*WIDTH),'attn_out':linear(WIDTH,WIDTH,.02/np.sqrt(2*LAYERS)),
                       'ln2':ln(),'expand':linear(WIDTH,4*WIDTH),'contract':linear(4*WIDTH,WIDTH,.02/np.sqrt(2*LAYERS))})
    return {'embedding':normal((V,WIDTH)),'position':normal((CONTEXT,WIDTH)),
            'blocks':tuple(blocks),'final_ln':ln(),'readout':linear(WIDTH,V)}

def linear(p,x):return x@p['w']+p['b']

def layer_norm(p,x):
    mean=jnp.mean(x,axis=-1,keepdims=True)
    variance=jnp.mean((x-mean)**2,axis=-1,keepdims=True)
    return (x-mean)*lax.rsqrt(variance+1e-5)*p['g']+p['b']

def mlp(p,previous):return linear(p['readout'],jnp.tanh(linear(p['hidden'],p['embedding'][previous])))

def transformer(p,tokens,return_attention=False):
    batch,length=tokens.shape
    x=p['embedding'][tokens]+p['position'][None,:length]
    attention=[]
    causal=jnp.arange(length)[:,None]>=jnp.arange(length)[None,:]
    for block in p['blocks']:
        h=layer_norm(block['ln1'],x)
        q,k,v=jnp.split(linear(block['qkv'],h),3,axis=-1)
        reshape=lambda z:z.reshape(batch,length,HEADS,WIDTH//HEADS).transpose(0,2,1,3)
        q,k,v=map(reshape,(q,k,v))
        scores=(q@jnp.swapaxes(k,-1,-2))/jnp.sqrt(float(WIDTH//HEADS))
        a=jax.nn.softmax(jnp.where(causal[None,None],scores,-jnp.inf),axis=-1)
        out=(a@v).transpose(0,2,1,3).reshape(batch,length,WIDTH)
        x=x+linear(block['attn_out'],out)
        h=layer_norm(block['ln2'],x)
        x=x+linear(block['contract'],jax.nn.gelu(linear(block['expand'],h),approximate=True))
        if return_attention:attention.append(a)
    logits=linear(p['readout'],layer_norm(p['final_ln'],x))
    return (logits,jnp.stack(attention)) if return_attention else logits

def save_model(kind,params):
    leaves,_=jax.tree.flatten(params)
    np.savez(ART/f'{kind}-final.npz',**{f'p{i}':np.asarray(v) for i,v in enumerate(leaves)})

def load_model(kind):
    _,structure=jax.tree.flatten(initialize('mlp' if kind=='mlp-initial' else kind,random.PRNGKey(0)))
    with np.load(ART/f'{kind}-final.npz') as f:return jax.tree.unflatten(structure,[jnp.asarray(f[f'p{i}']) for i in range(len(f.files))])

def schedule(step,kind):
    total,warmup,peak=(4000,200,.006) if kind=='mlp' else (6000,300,.003)
    progress=jnp.clip((step-warmup)/(total-warmup-1),0,1)
    decay=.1+.9*.5*(1+jnp.cos(jnp.pi*progress))
    return peak*jnp.where(step<warmup,(step+1)/warmup,decay)

def mlp_batch_loss(p,indices,tokens):
    previous=jnp.where(indices>0,tokens[jnp.maximum(indices-1,0)],2)
    targets=tokens[indices]
    # EXACT aggregation of the actual 16,384 sampled examples, not full-data
    # training or a smaller batch. A repeated input has identical logits;
    # weighting its log-probabilities by pair counts gives the same gradients.
    counts=jnp.bincount(previous*V+targets,length=V*V).reshape(V,V)
    logp=jax.nn.log_softmax(mlp(p,jnp.arange(V)),axis=-1)
    return -jnp.sum(counts*logp)/len(indices)

@functools.partial(jax.jit,static_argnames=('kind',))
def train_chunk(p,m,v,key,step,tokens,kind):
    def body(carry,_):
        params,first,second,rng,count=carry;rng,batch_key=random.split(rng)
        if kind=='mlp':
            indices=random.randint(batch_key,(16384,),0,len(tokens))
            value,gradient=jax.value_and_grad(mlp_batch_loss)(params,indices,tokens)
        else:
            starts=random.randint(batch_key,(32,),0,len(tokens)-CONTEXT)
            block=tokens[starts[:,None]+jnp.arange(CONTEXT+1)[None]]
            def objective(par):
                logp=jax.nn.log_softmax(transformer(par,block[:,:-1]),axis=-1)
                return -jnp.take_along_axis(logp,block[:,1:,None],axis=-1).mean()
            value,gradient=jax.value_and_grad(objective)(params)
        norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(gradient)))
        gradient=jax.tree.map(lambda g:g*jnp.minimum(1,1/(norm+1e-12)),gradient)
        first=jax.tree.map(lambda a,g:.9*a+.1*g,first,gradient)
        second=jax.tree.map(lambda a,g:.999*a+.001*g*g,second,gradient)
        next_count=count+1;lr=schedule(count,kind)
        def update(w,a,b):
            adam=(a/(1-.9**next_count))/(jnp.sqrt(b/(1-.999**next_count))+1e-8)
            return w-lr*(adam+(.01*w if w.ndim>=2 else 0))
        params=jax.tree.map(update,params,first,second)
        return (params,first,second,rng,next_count),value
    (p,m,v,key,step),losses=lax.scan(body,(p,m,v,key,step),None,length=50)
    return p,m,v,key,step,losses

@jax.jit
def score_transformer_batch(p,inputs,targets):
    logits=transformer(p,inputs)
    return -jnp.take_along_axis(jax.nn.log_softmax(logits,axis=-1),targets[...,None],axis=-1)[...,0]

def evaluate(kind,p,validation):
    if kind=='mlp':
        logits=np.asarray(mlp(p,jnp.arange(V)),dtype=np.float64)
        logp=logits-logsumexp(logits,axis=-1,keepdims=True)
        previous=np.r_[2,validation[:-1]]
        return -logp[previous,validation]
    length=len(validation);rows=(length+CONTEXT-1)//CONTEXT
    previous=np.r_[2,validation[:-1]]
    x=np.pad(previous,(0,rows*CONTEXT-length),constant_values=2).reshape(rows,CONTEXT)
    y=np.pad(validation,(0,rows*CONTEXT-length),constant_values=2).reshape(rows,CONTEXT)
    losses=[]
    for start in range(0,rows,128):
        xb=x[start:start+128];yb=y[start:start+128];size=len(xb)
        if size<128:
            xb=np.pad(xb,((0,128-size),(0,0)),constant_values=2);yb=np.pad(yb,((0,128-size),(0,0)),constant_values=2)
        losses.append(np.asarray(score_transformer_batch(p,jnp.asarray(xb),jnp.asarray(yb)))[:size])
    return np.concatenate(losses).ravel()[:length]
