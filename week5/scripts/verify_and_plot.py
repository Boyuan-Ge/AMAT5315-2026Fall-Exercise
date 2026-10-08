"""Audit numerical evidence and render the Week 5 figures from saved arrays."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
reflector = json.loads((ROOT / "inputs/reflector.json").read_text())
marmousi = json.loads((ROOT / "inputs/marmousi.json").read_text())


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def field(ax, values, title, extent, cmap="seismic", symmetric=True):
    vmax = np.max(np.abs(values)) if symmetric else None
    kw = {"vmin": -vmax, "vmax": vmax} if symmetric else {}
    im = ax.imshow(values, extent=extent, origin="upper", cmap=cmap, aspect="auto", **kw)
    ax.set(xlabel="horizontal position (km)", ylabel="depth (km)", title=title)
    plt.colorbar(im, ax=ax, shrink=.85)


def plot_inputs():
    e = reflector
    speed = np.array(e["background"]) + np.array(e["perturbation"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    field(ax[0], speed, "Seismic acquisition and thin reflector", [0,4,4,0], "viridis", False)
    shots = np.array(e["shots"])
    rec = np.array(e["receivers"])
    ax[0].scatter(rec[:,0]*.1,rec[:,1]*.1,s=18,c="white",edgecolor="black",label="receivers")
    ax[0].scatter(shots[:,0]*.1,shots[:,1]*.1,marker="*",s=120,c="red",label="shots")
    ax[0].legend(loc="lower right")
    t = np.arange(e["steps"])*e["dt"]*.1
    theta=np.pi*e["source_frequency"]*(np.arange(e["steps"])*e["dt"]-e["source_peak_time"])
    pulse=(1-2*theta**2)*np.exp(-theta**2)*e["source_amplitude"]
    ax[1].plot(t,pulse)
    ax[1].set(xlabel="time (s)",ylabel="source pulse",title="Ricker source; peak at 1.5 s")
    ax[1].grid(alpha=.2)
    save(fig,A/"inputs.png")


def plot_gathers():
    e=reflector
    traces=np.load(A/"forward/traces.npy")
    assert traces.shape==(3,240,14)
    l2=float(np.linalg.norm(traces))
    assert abs(l2/11.574770-1)<1e-4, l2
    extrema=[]
    for shot in range(3):
        k,r=np.unravel_index(np.argmax(abs(traces[shot])),traces[shot].shape)
        amp=float(abs(traces[shot,k,r]))
        expected=.59271397 if shot==1 else .60809514
        assert k==83 and abs(amp/expected-1)<1e-4,(shot,k,r,amp)
        extrema.append({"shot":shot,"trace_index":int(k),"receiver":int(r),"pressure":amp})
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    vmax=np.max(abs(traces))
    rx=np.array(e["receivers"])[:,0]*.1
    for i,ax in enumerate(axs):
        im=ax.imshow(traces[i],extent=[rx.min(),rx.max(),4.8,0],origin="upper",aspect="auto",
                     cmap="seismic",vmin=-vmax,vmax=vmax)
        ax.set(xlabel="receiver position (km)",title=f"Shot {i}; source x={e['shots'][i][0]*.1:g} km")
    axs[0].set_ylabel("time (s)")
    fig.colorbar(im,ax=axs,shrink=.8,label="pressure")
    fig.savefig(A/"forward/gathers.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    return l2,extrema


def plot_frame(path,step,title,out,mark_reflector=False):
    run=json.loads((path.parent/"run.json").read_text())
    steps=run["recording"]["steps"]
    frame=np.load(path)[steps.index(step)]
    fig,ax=plt.subplots(figsize=(6,5))
    field(ax,frame,title,[0,4,4,0])
    rec=np.array(reflector["receivers"])
    ax.scatter(rec[:,0]*.1,rec[:,1]*.1,s=8,c="black")
    if mark_reflector:ax.axhline(2.1,color="yellow",ls="--",lw=1)
    save(fig,out)
    return float(np.max(abs(frame)))


def plot_image():
    m=np.array(reflector["perturbation"])
    im=np.load(A/"adjoint/image.npy")
    crop=np.s_[10:34,7:34]
    profile=np.linalg.norm(im[crop],axis=1)
    peak=int(np.argmax(profile)+10)
    assert abs(peak-21)<=1,peak
    fig,axs=plt.subplots(1,3,figsize=(13,5),sharey=True)
    field(axs[0],m[crop],"1. Known reflector",[.7,3.3,3.3,1.0])
    field(axs[1],im[crop],"2. Raw signed RTM image",[.7,3.3,3.3,1.0])
    depth=np.arange(10,34)*.1
    axs[2].plot(profile,depth)
    axs[2].axhline(2.1,color="black",ls="--",label="true depth")
    axs[2].axhline(peak*.1,color="red",ls=":",label="image peak")
    axs[2].set(xlabel="row L2 norm",title="3. Depth profile")
    axs[2].legend()
    save(fig,A/"adjoint/image.png")
    return peak


def audit(actions,n,budget):
    saved={0}; working=0; grad=[]; invalid_restores=0; budget_overruns=0
    for a in actions:
        op=a["action"];step=a["step"]
        if op=="restore":
            if step not in saved: invalid_restores+=1
            working=step
        elif op=="call":
            if step!=working:invalid_restores+=1
            working=step+1
        elif op=="store":
            if working!=step or step in saved:invalid_restores+=1
            saved.add(step)
        elif op=="grad":
            if step not in saved:invalid_restores+=1
            grad.append(step)
        elif op=="fetch":
            if step not in saved or step==0:invalid_restores+=1
            else:saved.remove(step)
        else:invalid_restores+=1
        budget_overruns+=int(len(saved)>budget+1 or a["saved_states"]!=len(saved))
    bad_grads=sum(a!=b for a,b in zip(grad,range(n-1,-1,-1)))+abs(len(grad)-n)
    if saved!={0}:invalid_restores+=1
    return {"bad_grads":bad_grads,"invalid_restores":invalid_restores,
            "budget_overruns":budget_overruns}


def plot_checkpoint():
    full=np.load(A/"adjoint/image.npy")
    rows=[]
    for b in (1,3,5,10):
        result=json.loads((A/f"checkpoint-{b}/result.json").read_text())["statistics"]
        image=np.load(A/f"checkpoint-{b}/image.npy")
        err=float(np.linalg.norm(image-full)/np.linalg.norm(full))
        audits=[]
        for shot in range(3):
            actions=json.loads((A/f"checkpoint-{b}/actions-{shot}.json").read_text())
            audits.append(audit(actions,240,b))
        assert err<1e-9 and all(not any(v.values()) for v in audits),(b,err,audits)
        reference={1:28680,3:1695,5:990,10:642}[b]
        assert result["per_shot"][0]["scheduler_forward_calls"]==reference
        assert result["peak_saved_states"]==b+1
        rows.append({"budget":b,"image_relative_error":err,"audit":audits,
                     "forward_calls_per_shot":reference,
                     "peak_saved_states":result["peak_saved_states"],
                     "peak_saved_bytes":result["peak_saved_bytes"]})
    actions=json.loads((A/"checkpoint-5/actions-0.json").read_text())
    fig,ax=plt.subplots(figsize=(11,4))
    colors={"store":"purple","restore":"green","call":"#6495ed","grad":"orange","fetch":"gray"}
    for op,color in colors.items():
        ii=[i for i,a in enumerate(actions) if a["action"]==op]
        ax.scatter(ii,[actions[i]["step"] for i in ii],s=3 if op=="call" else 8,
                   color=color,label=op,rasterized=True)
    ax.set(xlabel="operation index",ylabel="time step",title="Treeverse actions; first shot, budget 5")
    ax.legend(ncol=5)
    save(fig,A/"checkpoint-actions.png")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    b=[r["budget"] for r in rows]
    axs[0].semilogy(b,[r["forward_calls_per_shot"] for r in rows],"o-",label="Treeverse")
    axs[0].axhline(240,ls="--",c="black",label="full history")
    axs[0].set(xlabel="additional checkpoint slots",ylabel="forward steps per shot",
                title="Recomputation work")
    axs[0].legend()
    axs[1].plot(b,[r["peak_saved_bytes"] for r in rows],"o-",label="Treeverse")
    axs[1].set_ylim(0,400_000)
    axs[1].text(.03,.96,"Full history: 6,481,936 bytes (241 states; off scale)",
                transform=axs[1].transAxes,va="top",fontsize=9)
    for row in rows:
        axs[1].annotate(f"{row['peak_saved_states']} states",
                        (row["budget"],row["peak_saved_bytes"]),
                        xytext=(4 if row["budget"]==1 else 0,8),
                        textcoords="offset points",
                        ha="left" if row["budget"]==1 else "center",fontsize=8)
    axs[1].set(xlabel="additional checkpoint slots",ylabel="peak saved bytes",
                title="Saved-state storage")
    save(fig,A/"checkpoint-work.png")
    return rows


def plot_marmousi():
    e=marmousi
    bg=np.array(e["background"])
    m=np.array(e["perturbation"])
    data=np.load(A/"marmousi-born/born_data.npy")
    im=np.load(A/"marmousi-image/image.npy")
    stats=json.loads((A/"marmousi-image/result.json").read_text())["statistics"]
    norm=float(np.linalg.norm(im))
    assert abs(norm/6.7037741e-4-1)<1e-4,norm
    assert stats["peak_saved_states"]==6 and stats["peak_saved_bytes"]==20_788_320
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    width=e["nx"]*e["dx"]*.1
    depth=e["nz"]*e["dx"]*.1
    field(axs[0,0],bg,"Smoothed Marmousi background",[0,width,depth,0],"viridis",False)
    field(axs[0,1],m,"Short-wavelength perturbation",[0,width,depth,0])
    center=min(range(len(e["shots"])),key=lambda i:abs(e["shots"][i][0]*e["dx"]*.1-10))
    recv=np.array(e["receivers"])[:,0]*e["dx"]*.1
    vmax=np.max(abs(data[center]))
    g=axs[1,0].imshow(data[center],extent=[recv.min(),recv.max(),e["steps"]*e["dt"]*.1,0],
                       origin="upper",aspect="auto",cmap="seismic",vmin=-vmax,vmax=vmax)
    axs[1,0].set(xlabel="receiver position (km)",ylabel="time (s)",
                 title=f"Born gather; source x={e['shots'][center][0]*e['dx']*.1:.1f} km")
    plt.colorbar(g,ax=axs[1,0],shrink=.85)
    field(axs[1,1],im,"Checkpointed raw RTM image",[0,width,depth,0])
    save(fig,A/"marmousi.png")
    return norm,stats["peak_saved_bytes"]


def main():
    plot_inputs()
    l2,extrema=plot_gathers()
    fullmax=plot_frame(A/"forward/wavefield.npy",150,"Forward wavefield; 3.00 s",
                       A/"forward/wavefield.png")
    echomax=plot_frame(A/"forward/echo.npy",150,"Reflector echo; 3.00 s",
                       A/"forward/echo.png",True)
    d=np.load(A/"born/born_data.npy")
    im=np.load(A/"adjoint/image.npy")
    m=np.array(reflector["perturbation"])
    left=float(np.sum(d*d));right=float(np.sum(m*im))
    transpose_error=abs(left-right)/max(abs(left),abs(right))
    assert transpose_error<1e-9 and abs(left/.0348479-1)<1e-4
    peak=plot_image()
    plot_frame(A/"adjoint/wavefield.npy",132,"Adjoint field; 2.64 s",
               A/"adjoint/wavefield.png",True)
    rows=plot_checkpoint()
    marm_norm,marm_bytes=plot_marmousi()
    report={"trace_l2":l2,"shot_extrema":extrema,"wavefield_step150_max":fullmax,
            "echo_step150_max":echomax,"transpose_left":left,"transpose_right":right,
            "transpose_relative_error":transpose_error,"image_peak_row":peak,
            "checkpoint_runs":rows,"marmousi_image_l2":marm_norm,
            "marmousi_peak_saved_bytes":marm_bytes}
    (A/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
