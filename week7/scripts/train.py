"""Fit either required MLP from fresh target samples; retain prescribed stages."""
import argparse
import time
import tomllib
from network import *

def train(kind):
    config=tomllib.loads((ROOT/f"{kind if kind=='diffusion' else 'flow'}.toml").read_text())
    # These kernels deliberately implement the sheet's fixed experiment, not
    # arbitrary architectures. Reject edited settings instead of silently
    # recording settings that the compiled kernels do not actually use.
    model=config['model'];settings=config['training']
    expected={'updates':25000,'batch_size':512,'optimizer':'Adam','beta1':.9,
              'beta2':.999,'epsilon':1e-8,'learning_rate':.001,
              'learning_rate_decay':'cosine','final_learning_rate':0.,'precision':'float32'}
    assert all(settings[k]==v for k,v in expected.items()),'Unsupported training configuration'
    assert model['hidden_widths']==[128]*3 and model['activation']=='SiLU'
    assert model['time_features']==32 and model['time_frequency_base']==10000.
    assert model['bias_initialization']==0. and model['weight_initialization']=='normal with std 1/sqrt(input width)'
    assert model['time_offset' if kind=='diffusion' else 'time_scale']==(-1 if kind=='diffusion' else 1000.)
    seed=config["training"]["seed"]
    init_key,key=random.split(random.PRNGKey(seed))
    params=initialize(init_key)
    m=jax.tree.map(jnp.zeros_like,params)
    v=jax.tree.map(jnp.zeros_like,params)
    out=ART/kind
    out.mkdir(exist_ok=True)
    checkpoints=config["training"].get("checkpoint_updates",[0,25000])
    save_params(out/"update-0.npz",params)
    values=[]
    count=jnp.int32(0)
    start=time.perf_counter()
    for chunk in range(500):
        params,m,v,key,count,losses=train_chunk(params,m,v,key,count,kind)
        values.extend(np.asarray(losses).tolist())
        step=(chunk+1)*50
        if step in checkpoints:save_params(out/f"update-{step}.npz",params)
        if step%1000==0:print(f"{kind}: {step}/25000; recent loss={np.mean(values[-500:]):.5f}",flush=True)
    elapsed=time.perf_counter()-start
    np.save(out/"loss.npy",np.array(values,np.float32))
    report={"kind":kind,"settings":config,"seconds":elapsed,"environment":environment(),
            "saved_updates":checkpoints,"first_500_mean_loss":float(np.mean(values[:500])),
            "last_1000_mean_loss":float(np.mean(values[-1000:])),
            "loss_by_100_updates":[float(np.mean(values[i:i+100])) for i in range(0,25000,100)],
            "optimizer":"Adam, bias-corrected moments; cosine starts at 0.001 and ends at 0"}
    report['config_sha256']=sha256(ROOT/f'{kind}.toml')
    write_json(out/"settings.json",report)
    write_json(EVIDENCE/f"{kind}-training.json",report)
    print(f"{kind}: finished in {elapsed:.2f}s",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("kind",choices=["diffusion","flow"])
    args=parser.parse_args()
    train(args.kind)
