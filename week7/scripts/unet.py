"""Fixed-checkpoint UNet2DModel in pure JAX, matching diffusers v0.40.0.

Public tensors are NCHW. Internally convolutions use NHWC/HWIO. No weights train.
The network receives checkpoint index t-1; its input is NOT clipped.
"""
from common import *
import jax.numpy as jnp
from jax import lax
from safetensors.numpy import load_file

REVISION="267b167dc01f0e4e61923ea244e8b988f84deb80"

def load_model():
    folder=INPUT/"cifar-model"
    config=json.loads((folder/"config.json").read_text())
    assert config["block_out_channels"]==[128,256,256,256]
    assert config["layers_per_block"]==2 and config["norm_num_groups"]==32
    assert config["norm_eps"]==1e-6 and config["downsample_padding"]==0
    assert config["flip_sin_to_cos"] is False and config["freq_shift"]==1
    weights=load_file(folder/"diffusion_pytorch_model.safetensors")
    assert len(weights)==328
    converted={}
    for name,value in weights.items():
        if value.ndim==4:value=value.transpose(2,3,1,0)
        elif value.ndim==2:value=value.T
        converted[name]=jnp.asarray(value,dtype=jnp.float32)
    return converted,config

def linear(p,x,name):
    return x@p[name+".weight"]+p[name+".bias"]

def conv(p,x,name,stride=1,padding="SAME"):
    h=lax.conv_general_dilated(x,p[name+".weight"],(stride,stride),padding,
        dimension_numbers=("NHWC","HWIO","NHWC"),precision=lax.Precision.HIGHEST)
    return h+p[name+".bias"]

def group_norm(p,x,name):
    b,h,w,c=x.shape
    grouped=x.reshape(b,h,w,32,c//32)
    mean=jnp.mean(grouped,axis=(1,2,4),keepdims=True)
    variance=jnp.mean((grouped-mean)**2,axis=(1,2,4),keepdims=True)
    normalized=((grouped-mean)*lax.rsqrt(variance+1e-6)).reshape(x.shape)
    return normalized*p[name+".weight"]+p[name+".bias"]

def resnet(p,x,temb,name):
    h=conv(p,jax.nn.silu(group_norm(p,x,name+".norm1")),name+".conv1")
    h=h+linear(p,jax.nn.silu(temb),name+".time_emb_proj")[:,None,None,:]
    h=conv(p,jax.nn.silu(group_norm(p,h,name+".norm2")),name+".conv2")
    if name+".conv_shortcut.weight" in p:x=conv(p,x,name+".conv_shortcut")
    return x+h

def attention(p,x,name):
    b,h,w,c=x.shape
    normalized=group_norm(p,x,name+".group_norm").reshape(b,h*w,c)
    q=linear(p,normalized,name+".query")
    k=linear(p,normalized,name+".key")
    v=linear(p,normalized,name+".value")
    scores=(q@jnp.swapaxes(k,-1,-2))*(c**-.5)
    weights=jax.nn.softmax(scores,axis=-1)
    result=linear(p,weights@v,name+".proj_attn").reshape(b,h,w,c)
    return x+result

@jax.jit
def unet(p,sample,model_index):
    assert sample.ndim==4 and sample.shape[1:]==(3,32,32)
    t=jnp.broadcast_to(jnp.asarray(model_index,jnp.float32),(sample.shape[0],))
    # diffusers get_timestep_embedding: 128 channels, downscale_freq_shift=1.
    frequencies=jnp.exp(-jnp.log(10000.)*jnp.arange(64,dtype=jnp.float32)/63)
    phase=t[:,None]*frequencies
    embedding=jnp.concatenate((jnp.sin(phase),jnp.cos(phase)),axis=-1)
    temb=linear(p,jax.nn.silu(linear(p,embedding,"time_embedding.linear_1")),
                "time_embedding.linear_2")
    x=conv(p,sample.transpose(0,2,3,1),"conv_in")
    saved=[x]
    for block in range(4):
        for layer in range(2):
            name=f"down_blocks.{block}.resnets.{layer}"
            x=resnet(p,x,temb,name)
            if block==1:x=attention(p,x,f"down_blocks.{block}.attentions.{layer}")
            saved.append(x)
        if block<3:
            # PyTorch Downsample2D(padding=0) pads only the right/bottom edges.
            x=jnp.pad(x,((0,0),(0,1),(0,1),(0,0)))
            x=conv(p,x,f"down_blocks.{block}.downsamplers.0.conv",stride=2,padding="VALID")
            saved.append(x)
    x=resnet(p,x,temb,"mid_block.resnets.0")
    x=attention(p,x,"mid_block.attentions.0")
    x=resnet(p,x,temb,"mid_block.resnets.1")
    for block in range(4):
        for layer in range(3):
            x=jnp.concatenate((x,saved.pop()),axis=-1)
            x=resnet(p,x,temb,f"up_blocks.{block}.resnets.{layer}")
            if block==2:x=attention(p,x,f"up_blocks.{block}.attentions.{layer}")
        if block<3:
            x=jnp.repeat(jnp.repeat(x,2,axis=1),2,axis=2)
            x=conv(p,x,f"up_blocks.{block}.upsamplers.0.conv")
    assert not saved
    x=conv(p,jax.nn.silu(group_norm(p,x,"conv_norm_out")),"conv_out")
    return x.transpose(0,3,1,2)
