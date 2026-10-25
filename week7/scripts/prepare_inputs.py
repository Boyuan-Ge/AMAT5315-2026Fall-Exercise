"""Pinned model and observed-image provenance; no downloaded files enter git."""
import shutil
import urllib.request
from common import *
from huggingface_hub import hf_hub_download

REVISION="267b167dc01f0e4e61923ea244e8b988f84deb80"
IMAGE_URL="https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz"
CIFAR_URL="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
WEIGHT_SHA="1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f"

def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix(path.suffix+".partial")
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open("wb") as f:
            shutil.copyfileobj(response,f)
        temporary.replace(path)

def main():
    model_dir=INPUT/"cifar-model"
    model_dir.mkdir(exist_ok=True)
    records=[]
    for name in ("config.json","diffusion_pytorch_model.safetensors"):
        path=model_dir/name
        if not path.exists():
            hf_hub_download("google/ddpm-cifar10-32",name,revision=REVISION,local_dir=model_dir)
        records.append({"file":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,
                        "sha256":sha256(path),
                        "url":f"https://huggingface.co/google/ddpm-cifar10-32/resolve/{REVISION}/{name}"})
    assert records[1]["sha256"]==WEIGHT_SHA
    fetch(IMAGE_URL,INPUT/"images.npz")
    records.append({"file":"inputs/images.npz","url":IMAGE_URL,
                    "bytes":(INPUT/"images.npz").stat().st_size,"sha256":sha256(INPUT/"images.npz")})
    existing=ROOT.parent/"week6/inputs/cifar10/cifar-10-binary.tar.gz"
    if existing.exists():real_archive=existing
    else:
        real_archive=INPUT/"cifar10/cifar-10-binary.tar.gz"
        fetch(CIFAR_URL,real_archive)
    records.append({"file":str(real_archive.relative_to(ROOT.parent)),"url":CIFAR_URL,
                    "bytes":real_archive.stat().st_size,"sha256":sha256(real_archive),
                    "role":"real photographs read by the Week 6 gallery loader"})
    write_json(EVIDENCE/"input-provenance.json",{"model_repository":"google/ddpm-cifar10-32",
               "revision":REVISION,"files":records})
    print("Pinned inputs and hashes verified.",flush=True)

if __name__=="__main__":main()
