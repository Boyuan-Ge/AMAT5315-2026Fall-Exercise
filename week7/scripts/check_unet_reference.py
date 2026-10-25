"""Independent oracle: compare pure-JAX U-Net with diffusers v0.40.0 on CPU."""
import time
from unet import *
import torch
import diffusers
from diffusers import UNet2DModel

def main():
    torch.set_num_threads(4)
    p,config=load_model()
    raw=load_file(INPUT/"cifar-model/diffusion_pytorch_model.safetensors")
    state={}
    for name,array in raw.items():
        if ".attentions." in name:
            for old,new in ((".query.",".to_q."),(".key.",".to_k."),
                            (".value.",".to_v."),(".proj_attn.",".to_out.0.")):
                name=name.replace(old,new)
        state[name]=torch.from_numpy(array.copy())
    reference=UNet2DModel.from_config(config).eval()
    reference.load_state_dict(state,strict=True)
    rng=np.random.default_rng(7200)
    x=rng.normal(size=(2,3,32,32)).astype(np.float32)
    records=[]
    for index in (0,499,999):
        with torch.no_grad():
            expected=reference(torch.from_numpy(x),index).sample.numpy()
        actual=np.asarray(unet(p,jnp.asarray(x),index))
        error=float(np.max(abs(actual-expected)))
        rms=float(np.sqrt(np.sum((actual-expected)**2)/np.sum(expected**2)))
        records.append({"model_index":index,"max_absolute_error":error,"relative_rms_error":rms})
        print(records[-1],flush=True)
        assert error<5e-4 and rms<1e-4,records
    batch=jnp.asarray(rng.normal(size=(4,3,32,32)).astype(np.float32))
    unet(p,batch,499).block_until_ready()
    start=time.perf_counter()
    for _ in range(5):unet(p,batch,499).block_until_ready()
    report={"revision":REVISION,"reference":"diffusers.UNet2DModel",
            "diffusers_version":diffusers.__version__,"torch_version":torch.__version__,
            "tensor_count":len(p),"parameter_count":sum(v.size for v in p.values()),
            "weight_sha256":sha256(INPUT/"cifar-model/diffusion_pytorch_model.safetensors"),
            "input_seed":7200,"input_shape":[2,3,32,32],"comparisons":records,
            "jax_seconds_per_batch4":(time.perf_counter()-start)/5,
            "all_reference_checks_passed":True,"environment":environment()}
    write_json(EVIDENCE/"unet-reference.json",report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()
