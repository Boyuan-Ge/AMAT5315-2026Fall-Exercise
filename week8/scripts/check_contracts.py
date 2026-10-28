"""Independent numerical contracts and deliberate bad-model controls."""
from scipy.optimize import minimize_scalar
from language_models import *
from language_analysis import exact_distribution,sampling_distribution,inverse_cdf
from ngrams import Ngram,codes
from memory import evaluate_stores,glyph,capacity
from scaling import objective,optimum,predict

def main():
    results={}
    def check(name,condition):
        assert bool(condition),name
        results[name]=True
    rng=np.random.default_rng(1890)
    tokens=np.array([2,3,4,3,5,2,4],np.int32)
    keys=codes(tokens,3)
    reference=[]
    padded=[2,2]+tokens.tolist()
    for i in range(len(tokens)):reference.append(padded[i]*512**2+padded[i+1]*512+padded[i+2])
    check('packed n-grams equal independent tuple encoding',np.array_equal(keys,reference))
    model=Ngram.fit(tokens,2,.1)
    for previous in [2,3,4,5,99]:
        probability=model.probabilities([previous]);check(f'bigram normalized even for context {previous}',abs(probability.sum()-1)<1e-12 and np.all(probability>0))
    check('unseen context reduces to uniform',np.array_equal(model.probabilities([99]),np.ones(512)/512))
    check('deliberately unnormalized score fails the normalization check',abs(np.exp(np.arange(512)/512).sum()-1)>100)
    params=initialize('mlp',random.PRNGKey(1891));sequence=jnp.array(rng.integers(0,512,200),jnp.int32)
    indices=jnp.array(rng.integers(0,200,128),jnp.int32)
    def direct_loss(p):
        previous=jnp.where(indices>0,sequence[jnp.maximum(indices-1,0)],2)
        logp=jax.nn.log_softmax(mlp(p,previous),axis=-1)
        return -jnp.take_along_axis(logp,sequence[indices,None],axis=-1).mean()
    aggregated,gradient=jax.value_and_grad(mlp_batch_loss)(params,indices,sequence)
    direct,direct_gradient=jax.value_and_grad(direct_loss)(params)
    gradient_difference=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree.leaves(gradient),jax.tree.leaves(direct_gradient)))
    check('MLP aggregated loss equals actual per-example batch loss',abs(float(aggregated-direct))<2e-6)
    check('MLP aggregation preserves EVERY parameter gradient',gradient_difference<2e-6)
    p=initialize('transformer',random.PRNGKey(1892));x=jnp.array(rng.integers(0,512,(2,16)),jnp.int32)
    changed=x.at[:,8:].set(jnp.array(rng.integers(0,512,(2,8)),jnp.int32))
    logits,weights=transformer(p,x,return_attention=True);other=transformer(p,changed)
    check('future-token changes cannot affect earlier logits',np.max(abs(np.asarray(logits[:,:8]-other[:,:8])))<1e-6)
    check('every future attention weight is exactly zero',np.max(np.triu(np.asarray(weights),k=1))==0)
    check('every causal attention row sums to one',np.max(abs(np.asarray(weights).sum(axis=-1)-1))<3e-7)
    check('all eight heads present',weights.shape==(2,2,4,16,16))
    check('transformer parameter count includes learned positions and all biases',sum(v.size for v in jax.tree.leaves(p))==328832)
    def linear_biases(tree):
        if isinstance(tree,dict):
            if 'w' in tree:assert 'b' in tree and tree['b'].shape==(tree['w'].shape[1],)
            for value in tree.values():linear_biases(value)
        elif isinstance(tree,tuple):
            for value in tree:linear_biases(value)
    linear_biases(p);check('every linear map has its required bias',True)
    # A deliberately unmasked attention example demonstrably leaks future data.
    q=np.ones((4,2));k=np.arange(8).reshape(4,2)/10;v=np.eye(4);v2=v.copy();v2[-1]*=10
    check('unmasked negative control changes the first-position output',np.max(abs((softmax(q@k.T,axis=-1)@v)[0]-(softmax(q@k.T,axis=-1)@v2)[0]))>1)
    for kind,total,warmup,peak in [('mlp',4000,200,.006),('transformer',6000,300,.003)]:
        check(f'{kind}: linear warmup and exact 10%-peak endpoint',np.allclose([schedule(0,kind),schedule(warmup-1,kind),schedule(total-1,kind)],[peak/warmup,peak,peak*.1],rtol=1e-6,atol=1e-9))
    logits=np.full(512,-1000.);logits[:3]=np.log([.6,.3,.1])
    pnp=exact_distribution(logits,1.,'top-p',.8)
    pj=np.asarray(sampling_distribution(jnp.asarray(logits),1.,'top-p',.8))
    check('top-p INCLUDES the threshold-crossing token',np.allclose(pnp[:3],[2/3,1/3,0]) and np.allclose(pj[:3],pnp[:3],atol=1e-6))
    bad=np.r_[1.,np.zeros(511)]
    check('dropping the crossing token is rejected by the independent target',np.max(abs(bad-pnp))>.3)
    for mode,cutoff in [('temperature',0.),('top-k',2.),('top-p',.8),('min-p',.2)]:
        independent=exact_distribution(logits,.8,mode,cutoff)
        actual=np.asarray(sampling_distribution(jnp.asarray(logits),.8,mode,cutoff))
        check(f'{mode}: JAX probabilities match independent NumPy',np.max(abs(actual-independent))<1e-6)
    probability=jnp.array([.2,.3,.5]);uniforms=jnp.array([0.,.199,.2,.499,.5,.999])
    check('inverse CDF boundary convention is correct',np.array_equal(inverse_cdf(probability,uniforms),[0,0,1,1,2,2]))
    orthogonal=evaluate_stores(np.eye(4),np.eye(4))
    check('all stores recall orthogonal keys without interference',np.max(abs(orthogonal))<1e-12)
    key=np.array([[1.,0.],[0.,1.],[2**-.5,2**-.5]]);value=rng.normal(size=(3,2));value/=np.linalg.norm(value,axis=1,keepdims=True)
    hat=key@np.linalg.pinv(key);answer=hat@value
    expected=np.linalg.norm(answer-np.diag(hat)[:,None]*value,axis=1)/np.diag(hat)
    measured=evaluate_stores(key,value)[1]
    check('least-squares interference divides by H_jj own-signal weight',np.allclose(measured,expected,atol=1e-12))
    check('log-log capacity interpolation recovers an exact power law',abs(capacity(np.array([16,64]),np.array([.5,1.]))-64)<1e-10)
    check('picture fixture is 32x32 binary ink/background',glyph().shape==(32,32) and set(np.unique(glyph()))=={-1,1})
    data=np.load(ART/'scaling-fit.npz');theta=data['theta']+np.array([.01,-.02,.02,.01,-.01])
    value,analytic=objective(theta,data['n'],data['d'],data['loss']);finite=[]
    for i in range(5):
        offset=np.eye(5)[i]*1e-6
        finite.append((objective(theta+offset,data['n'],data['d'],data['loss'])[0]-objective(theta-offset,data['n'],data['d'],data['loss'])[0])/2e-6)
    check('Huber scaling-law gradient matches independent finite differences',np.max(abs(analytic-finite))<1e-4)
    parameters=data['parameters']
    for compute in [1e18,5.76e23,1e25]:
        n,d=optimum(parameters,compute);a,b,alpha,beta=parameters[1:]
        check(f'compute budget {compute:.2g}: 6*N*D=C',abs(6*n*d/compute-1)<1e-12)
        check(f'compute budget {compute:.2g}: marginal gains balance',abs((alpha*a/n**alpha)/(beta*b/d**beta)-1)<1e-12)
        numerical=minimize_scalar(lambda logn:predict(parameters,np.exp(logn),compute/(6*np.exp(logn))),bounds=(np.log(1e3),np.log(1e18)),method='bounded')
        check(f'compute budget {compute:.2g}: analytic optimum matches scalar minimization',abs(np.log(n)-numerical.x)<1e-5)
    write_json(EVIDENCE/'contract-checks.json',{'passed':len(results),'checks':results,'mlp_max_gradient_difference':gradient_difference,'scaling_max_gradient_difference':float(np.max(abs(analytic-finite))),'environment':environment()})
    print(f'{len(results)} independent contract checks passed.',flush=True)

if __name__=='__main__':main()
