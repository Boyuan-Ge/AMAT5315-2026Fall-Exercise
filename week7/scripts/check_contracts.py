"""Independent numerical and interface controls, including an instrumented RePaint."""
import inspect
import tomllib
from transport import *
import photo_experiments as photos

def main():
    results={}
    def record(name,value):
        assert bool(value),name
        results[name]=True
    beta=np.linspace(.0001,.02,1000,dtype=np.float64)
    abar=np.r_[1,np.cumprod(1-beta)]
    record('1000-step linear schedule and cumulative product',np.max(abs(np.asarray(ABAR)-abar))<4e-7)
    record('time features: zero phase = 16 zeros then 16 ones',np.array_equal(np.asarray(features(jnp.array(0.))),np.r_[np.zeros(16),np.ones(16)]))
    frequencies=10000.**(-np.arange(16)/16)
    record('diffusion t-1 and flow 1000*s feature conventions',np.allclose(features(jnp.array(499.)),np.r_[np.sin(499*frequencies),np.cos(499*frequencies)],atol=2e-5))
    rng=np.random.default_rng(76509);points=rng.normal(size=(17,2)).astype(np.float32)
    centres=np.array([[3.,0.],[0.,3.],[-3.,0.],[0.,-3.]])
    noise_errors=[];velocity_errors=[]
    for t in [1,100,500,1000]:
        a=float(ABAR[t]);mean=np.sqrt(a)*centres;v=.09*a+1-a
        def logp(x):
            energies=-np.sum((x[:,None]-mean)**2,axis=-1)/(2*v)
            maximum=energies.max(axis=1)
            return maximum+np.log(np.exp(energies-maximum[:,None]).mean(axis=1))
        h=1e-4;score=np.empty_like(points,dtype=np.float64)
        for d in range(2):
            offset=np.eye(2)[d]*h
            score[:,d]=(logp(points.astype(float)+offset)-logp(points.astype(float)-offset))/(2*h)
        truth=-np.sqrt(1-a)*score
        noise_errors.append(float(np.max(abs(truth-np.asarray(exact_noise(jnp.asarray(points),t))))))
    record('exact noise agrees with independent finite-difference log-density score',max(noise_errors)<1e-5)
    for s in [.1,.5,.9,1.]:
        v=.09*(1-s)**2+s*s;res=points[:,None]-(1-s)*centres
        logweight=-np.sum(res**2,axis=-1)/(2*v);weight=np.exp(logweight-logweight.max(axis=1,keepdims=True));weight/=weight.sum(axis=1,keepdims=True)
        eclean=centres+(1-s)*.09/v*res;enoise=s/v*res
        truth=np.sum(weight[...,None]*(enoise-eclean),axis=1)
        velocity_errors.append(float(np.max(abs(truth-np.asarray(exact_velocity(jnp.asarray(points),s))))))
    record('exact velocity agrees with independent endpoint posterior conditioning',max(velocity_errors)<1e-5)
    record('velocity at s=1 equals x',np.allclose(exact_velocity(jnp.asarray(points),1.),points,atol=1e-6))
    p=initialize(random.PRNGKey(2026))
    record('MLP shape, zero biases, float32',[(w.shape,b.shape) for w,b in p]==[((34,128),(128,)),((128,128),(128,)),((128,128),(128,)),((128,2),(2,))] and all(np.all(np.asarray(b)==0) and w.dtype==jnp.float32 for w,b in p))
    initial=jnp.asarray(points);grid=jnp.array([1000,500,0])
    a=np.asarray(diffusion_sample(p,initial,random.PRNGKey(1),grid,method='ddim',exact=True))
    b=np.asarray(diffusion_sample(p,initial,random.PRNGKey(2),grid,method='ddim',exact=True))
    record('DDIM ignores reverse RNG key',np.array_equal(a,b))
    expected=points.astype(float)
    for t,tprime in [(1000,500),(500,0)]:
        eps=np.asarray(exact_noise(jnp.asarray(expected,dtype=jnp.float32),t))
        clean=(expected-np.sqrt(1-float(ABAR[t]))*eps)/np.sqrt(float(ABAR[t]))
        expected=np.sqrt(float(ABAR[tprime]))*clean+np.sqrt(1-float(ABAR[tprime]))*eps
    record('coarse DDIM uses schedule levels, not loop indices',np.allclose(a[-1],expected,atol=1e-4))
    one_a=diffusion_sample(p,initial,random.PRNGKey(1),jnp.array([1,0]),exact=True)
    one_b=diffusion_sample(p,initial,random.PRNGKey(2),jnp.array([1,0]),exact=True)
    record('DDPM final jump adds no noise',np.array_equal(one_a,one_b))
    x=points[:5].astype(float);y=points[5:12].astype(float)
    between=np.mean([np.linalg.norm(i-j) for i in x for j in y])
    within_x=np.mean([np.linalg.norm(x[i]-x[j]) for i in range(5) for j in range(5) if i!=j])
    within_y=np.mean([np.linalg.norm(y[i]-y[j]) for i in range(7) for j in range(7) if i!=j])
    record('energy distance excludes self-pairs and has no square root',abs(energy_distance(x,y)-(2*between-within_x-within_y))<1e-12)
    record('energy U-statistic is not forcibly clipped to nonnegative',energy_distance(x,x)<0)
    record('RePaint API has no hidden-truth argument',list(inspect.signature(photos.repaint).parameters)==['p','observed','mask','key'])
    calls=[]
    original=photos.unet
    def instrumented_predictor(params,x,index):
        jax.debug.callback(lambda t:calls.append(int(t)),index,ordered=True)
        return jnp.zeros_like(x)
    photos.unet=instrumented_predictor
    observed=jnp.asarray(rng.uniform(-1,1,(1,3,32,32)),dtype=jnp.float32)
    mask=jnp.ones_like(observed).at[:,:,6:26,6:26].set(0)
    observed=observed*mask
    sampled=photos.repaint((),observed,mask,random.PRNGKey(1));sampled.block_until_ready();jax.effects_barrier()
    expected_indices=[t-1 for t in range(1000,0,-20) for _ in range(1 if t==20 else 2)]
    record('RePaint performs exactly 99 predictor calls at t-1',calls==expected_indices)
    record('RePaint final observed pixels are exactly unchanged',np.max(abs(np.asarray((sampled[-1]-observed)*mask)))==0)
    record('RePaint clean-estimate clipping bounds final pixels',np.max(abs(np.asarray(sampled[-1])))<=1)
    photos.unet=original
    settings={kind:tomllib.loads((ROOT/f'{kind}.toml').read_text()) for kind in ['diffusion','flow']}
    record('both TOMLs contain required optimizer and update settings',all(c['training']['updates']==25000 and c['training']['batch_size']==512 and c['training']['optimizer']=='Adam' and c['model']['hidden_widths']==[128,128,128] for c in settings.values()))
    write_json(EVIDENCE/'contract-checks.json',{'checks':results,'passed':len(results),'noise_score_max_absolute_errors':noise_errors,'velocity_conditioning_max_absolute_errors':velocity_errors,'repaint_instrumented_indices':calls,'environment':environment()})
    print(f'{len(results)} independent contract checks passed.',flush=True)

if __name__=='__main__':main()
