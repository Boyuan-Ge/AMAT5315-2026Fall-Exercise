"""Unclipped 1000-step DDPM and observation-only RePaint, both executed in JAX."""
import argparse
import time
from unet import *
from network import BETA,ALPHA,ABAR
from jax import random

@jax.jit
def generate(p,key):
    key,start_key=random.split(key)
    initial=random.normal(start_key,(4,3,32,32),dtype=jnp.float32)
    def body(carry,t):
        x,rng=carry
        eps=unet(p,x,t-1)
        mean=(x-BETA[t-1]/jnp.sqrt(1-ABAR[t])*eps)/jnp.sqrt(ALPHA[t-1])
        rng,noise_key=random.split(rng)
        z=random.normal(noise_key,x.shape,dtype=jnp.float32)
        x=mean+jnp.where(t>1,jnp.sqrt(BETA[t-1]),0)*z
        return (x,rng),x
    _,states=lax.scan(body,(initial,key),jnp.arange(1000,0,-1))
    return jnp.concatenate((initial[None],states),axis=0)

@jax.jit
def repaint(p,observed,mask,key):
    # The withheld truth is not an argument and is never read by this sampler.
    key,start_key=random.split(key)
    initial=random.normal(start_key,observed.shape,dtype=jnp.float32)
    levels=jnp.arange(1000,-1,-20)
    def jump(carry,pair):
        x,rng=carry
        t,next_t=pair
        ratio=ABAR[t]/ABAR[next_t]
        attempts=jnp.where(next_t==0,1,2)
        def attempt(r,state):
            current,rng=state
            def forward_again(state):
                current,rng=state
                rng,k=random.split(rng)
                current=jnp.sqrt(ratio)*current+jnp.sqrt(1-ratio)*random.normal(k,current.shape)
                return current,rng
            current,rng=lax.cond(r>0,forward_again,lambda v:v,(current,rng))
            eps=unet(p,current,t-1)
            clean=jnp.clip((current-jnp.sqrt(1-ABAR[t])*eps)/jnp.sqrt(ABAR[t]),-1,1)
            mean=(jnp.sqrt(ABAR[next_t])*(1-ratio)/(1-ABAR[t])*clean+
                  jnp.sqrt(ratio)*(1-ABAR[next_t])/(1-ABAR[t])*current)
            variance=(1-ABAR[next_t])*(1-ratio)/(1-ABAR[t])
            rng,proposal_key,observation_key=random.split(rng,3)
            proposal=mean+jnp.sqrt(variance)*random.normal(proposal_key,current.shape)
            known=jnp.sqrt(ABAR[next_t])*observed+jnp.sqrt(1-ABAR[next_t])*random.normal(observation_key,current.shape)
            return mask*known+(1-mask)*proposal,rng
        x,rng=lax.fori_loop(0,attempts,attempt,(x,rng))
        return (x,rng),x
    _,states=lax.scan(jump,(initial,key),(levels[:-1],levels[1:]))
    return jnp.concatenate((initial[None],states),axis=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=["generation","inpainting","all"],default="all")
    args=parser.parse_args()
    p,config=load_model()
    out=ART/"photographs"
    out.mkdir(exist_ok=True)
    report=json.loads((out/"metrics.json").read_text()) if (out/"metrics.json").exists() else {}
    before=sha256(INPUT/"cifar-model/diffusion_pytorch_model.safetensors")
    if args.part in ("generation","all"):
        start=time.perf_counter()
        states=np.asarray(generate(p,random.PRNGKey(7200)))
        np.save(out/"generation.npy",states)
        report["generation"]={"seed":7200,"steps":1000,"images":4,"model_index":"mathematical t minus 1",
            "state_clipping":False,"display":"(x+1)/2 clipped only at rendering",
            "seconds":time.perf_counter()-start,"terminal_min":float(states[-1].min()),
            "terminal_max":float(states[-1].max())}
        write_json(out/"metrics.json",report)
        print("Generated four photographs.",report["generation"],flush=True)
    if args.part in ("inpainting","all"):
        with np.load(INPUT/"images.npz") as data:
            print("Prepared image arrays:",{k:data[k].shape for k in data.files},flush=True)
            name="images" if "images" in data.files else data.files[0]
            truth=data[name].astype(np.float32)
        if truth.shape==(4,32,32,3):truth=truth.transpose(0,3,1,2)
        assert truth.shape==(4,3,32,32)
        mask=np.ones_like(truth)
        mask[:,:,6:26,6:26]=0
        observed=truth*mask
        all_states=[]
        mse=[]
        changes=[]
        start=time.perf_counter()
        for seed in (7201,7202,7203,7204):
            states=np.asarray(repaint(p,jnp.asarray(observed),jnp.asarray(mask),random.PRNGKey(seed)))
            all_states.append(states)
            hidden=np.mean((states[-1,:, :,6:26,6:26]-truth[:,:,6:26,6:26])**2,axis=(1,2,3))
            difference=np.max(abs((states[-1]-observed)*mask),axis=(1,2,3))
            mse.append(hidden.tolist());changes.append(difference.tolist())
            print(f"RePaint seed {seed}: hidden MSE={hidden}; observed max change={difference}",flush=True)
        all_states=np.stack(all_states)
        variance=np.var(all_states[:,-1],axis=0)
        np.savez_compressed(out/"inpainting.npz",states=all_states,truth=truth,observed=observed,
                            mask=mask,variance=variance,seeds=np.arange(7201,7205))
        report["inpainting"]={"seeds":[7201,7202,7203,7204],"images":4,
            "hidden_rows_columns":"6:26, all channels","reverse_jumps":50,
            "attempts":"two per jump except final jump once","network_evaluations_per_completion":99,
            "hidden_mse_by_seed_then_image":mse,"observed_max_change_by_seed_then_image":changes,
            "mean_hidden_mse":float(np.mean(mse)),"max_observed_change":float(np.max(changes)),
            "mean_variance_inside_hole":float(np.mean(variance[:,:,6:26,6:26])),
            "max_variance_observed":float(np.max(variance*mask)),
            "seconds":time.perf_counter()-start,
            "sampler_inputs":"fixed predictor, observed image with hidden entries zero, mask, RNG key"}
        assert report["inpainting"]["max_observed_change"]<1e-5
        assert report["inpainting"]["max_variance_observed"]==0
        assert report["inpainting"]["mean_variance_inside_hole"]>0
        write_json(out/"metrics.json",report)
    after=sha256(INPUT/"cifar-model/diffusion_pytorch_model.safetensors")
    assert before==after
    report.update({"revision":REVISION,"weights_sha256":before,"weights_unchanged":True,
                   "predictor_implementation":"pure JAX","environment":environment()})
    write_json(out/"metrics.json",report)
    write_json(EVIDENCE/"photograph-metrics.json",report)

if __name__=="__main__":main()
