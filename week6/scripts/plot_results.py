"""Render all Week 6 evidence from retained raw arrays; no sample selection."""
from common import *
from scipy.special import ndtr


def load_json(name):
    return json.loads((EVIDENCE/name).read_text())


def rbm_figures():
    x,y,tx,ty=load_mnist()
    spins=2*(x[:16]>=128).astype(np.float32)-1
    fig=plt.figure(figsize=(15,5))
    outer=fig.add_gridspec(1,3,width_ratios=[1.25,1,1.2],wspace=.3)
    grid=outer[0].subgridspec(4,4)
    for i in range(16):
        show_image(fig.add_subplot(grid[i//4,i%4]),spins[i],str(y[i]),spin=True)
    ax=fig.add_subplot(outer[1])
    show_image(ax,spins[0],"784 visible spins\nink +1 (black), background -1 (white)",spin=True)
    ax=fig.add_subplot(outer[2])
    for i in range(7):
        for j in range(4):
            ax.plot([0,1],[i,j*2],color="#a4b9ca",lw=.6,zorder=1)
    ax.scatter(np.zeros(7),np.arange(7),s=130,color="#669fcd",zorder=2)
    ax.scatter(np.ones(4),np.arange(4)*2,s=160,color="#d8847c",zorder=2)
    ax.text(0,-.8,"784 visible bits",ha="center")
    ax.text(1,-.8,"128 hidden bits",ha="center")
    ax.text(.5,7,"Only cross-layer couplings W",ha="center")
    ax.set(xlim=(-.4,1.4),ylim=(-1.3,7.5))
    ax.axis("off")
    fig.suptitle("First 16 stored training images, one spin configuration, and the RBM",fontsize=15)
    savefig(fig,"rbm-spins-network.png")

    training=load_json("rbm-training.json")["models"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for model,color in (("cd1","#3378ae"),("cd10","#d35f3b")):
        hist=training[model]["history"]
        axes[0].plot([r["epoch"] for r in hist[1:]],[r["cd_surrogate"] for r in hist[1:]],"o-",ms=3,label=model.upper(),c=color)
        axes[1].plot([r["epoch"] for r in hist],[r["reconstruction_error"] for r in hist],"o-",ms=3,label=model.upper(),c=color)
    axes[0].set(xlabel="epoch",ylabel="CD surrogate (not NLL)",title="Mean pre-update batch surrogate")
    axes[1].set(xlabel="epoch",ylabel="sampled Hamming fraction",title="First 64 held-out images; fixed Gibbs randomness")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    savefig(fig,"rbm-learning.png")

    raw={k:np.load(ART/"rbm"/f"samples-cd{k}.npz") for k in (1,10)}
    test_bits=(tx[:64].reshape(-1,784)>=128).astype(np.float32)
    fig,axes=plt.subplots(3,8,figsize=(12,5))
    for i in range(8):
        for row,im in enumerate([test_bits[i],raw[1]["reconstruction"][i],raw[10]["reconstruction"][i]]):
            show_image(axes[row,i],im.reshape(28,28),f"label {ty[i]}" if row==0 else None,continuous=True)
    for row,label in enumerate(["Held-out input","CD-1: 1 step","CD-10: 1 step"]):
        axes[row,0].set_ylabel(label,fontsize=10)
    fig.suptitle("Reconstruction starts from a complete test digit")
    savefig(fig,"rbm-reconstructions.png")

    fig,axes=plt.subplots(8,5,figsize=(9,13))
    for model_index,k in enumerate((1,10)):
        for chain in range(4):
            row=model_index*4+chain
            for col,step in enumerate((0,1,10,100,1000)):
                show_image(axes[row,col],raw[k]["generation"][col,chain].reshape(28,28),
                           f"{step} steps" if row==0 else None,continuous=True)
            axes[row,0].set_ylabel(f"CD-{k}\nchain {chain}",fontsize=9)
    fig.suptitle("Fresh generation: first 4 matched random starts per model; no digit supplied")
    savefig(fig,"rbm-generation.png")

    fig,axes=plt.subplots(8,6,figsize=(10,13))
    for model_index,k in enumerate((1,10)):
        frames=raw[k]["completion_true"]
        for i in range(4):
            row=model_index*4+i
            show_image(axes[row,0],test_bits[i].reshape(28,28),"target" if row==0 else None,continuous=True)
            for col,step in enumerate((0,1,10,100,1000),1):
                show_image(axes[row,col],frames[col-1,i,0].reshape(28,28),
                           ("random lower half" if step==0 else f"{step} steps") if row==0 else None,continuous=True)
            axes[row,0].set_ylabel(f"CD-{k}\nlabel {ty[i]}",fontsize=9)
    fig.suptitle("Conditional completion: upper 14 rows stay clamped; first chain, first 4 test images")
    savefig(fig,"rbm-completion.png")

    fig,axes=plt.subplots(8,5,figsize=(9,13))
    for model_index,k in enumerate((1,10)):
        frames=raw[k]["completion_true"][-1]
        for i in range(4):
            row=model_index*4+i
            show_image(axes[row,0],test_bits[i].reshape(28,28),"target" if row==0 else None,continuous=True)
            for chain in range(4):
                show_image(axes[row,chain+1],frames[i,chain].reshape(28,28),
                           f"chain {chain}" if row==0 else None,continuous=True)
            axes[row,0].set_ylabel(f"CD-{k}\nlabel {ty[i]}",fontsize=9)
    fig.suptitle("Completion diversity after 1000 steps; first 4 of 8 chains per image")
    savefig(fig,"rbm-completion-diversity.png")

    sampling=load_json("rbm-sampling.json")
    fig,ax=plt.subplots(figsize=(8,4.5))
    for k,color in ((1,"#3378ae"),(10,"#d35f3b")):
        for condition,style in (("true","-"),("wrong","--")):
            metrics=sampling["models"][f"cd{k}"][condition]
            ax.errorbar([1,10,100,1000],metrics["mean"][1:],yerr=metrics["standard_error"][1:],
                        fmt="o"+style,c=color,capsize=3,label=f"CD-{k}, {condition} upper half")
    ax.set(xscale="log",xlabel="Gibbs steps",ylabel="lower-half ink overlap (Dice)",
           ylim=(0,.65),title="Conditional information improves completion; SE across 64 digits")
    ax.legend()
    ax.grid(alpha=.2)
    savefig(fig,"rbm-completion-overlap.png")


def hopfield_figures():
    data={n:load_json(f"hopfield-N{n}.json") for n in (100,200,400,800)}
    fig,ax=plt.subplots(figsize=(8,5))
    alpha=np.linspace(.05,.25,300)
    ax.plot(alpha,ndtr(-1/np.sqrt(alpha)),"k-",lw=2,label="Gaussian: Phi(-1/sqrt(alpha))")
    colors=plt.cm.viridis(np.linspace(.05,.8,4))
    for (n,run),color in zip(data.items(),colors):
        rows=run["rows"]
        ax.errorbar([r["alpha"] for r in rows],[r["unstable_mean"] for r in rows],
                    yerr=[r["unstable_se"] for r in rows],fmt="o",ms=3,capsize=2,c=color,label=f"N={n}")
    p=np.round(100*alpha)
    ax.plot(alpha,ndtr(-np.sqrt(99/(p-1))),"--",c=colors[0],label="finite-N prediction, N=100")
    ax.set(xlabel="load alpha = P/N",ylabel="fraction of unstable stored bits",
           title="Crosstalk destabilizes stored random patterns")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    savefig(fig,"hopfield-unstable.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    crossings={}
    for (n,run),color in zip(data.items(),colors):
        rows=run["rows"]
        al=np.array([r["alpha"] for r in rows])
        means=np.array([r["overlap_mean"] for r in rows])
        for ax,mean,se in [(axes[0],"overlap_mean","overlap_se"),(axes[1],"success_mean","success_se")]:
            ax.errorbar(al,[r[mean] for r in rows],yerr=[r[se] for r in rows],fmt="o-",ms=3,capsize=2,c=color,label=f"N={n}")
        i=next(i for i in range(1,len(al)) if means[i]<.9 and means[i-1]>=.9)
        crossings[n]=float(al[i-1]+(al[i]-al[i-1])*(means[i-1]-.9)/(means[i-1]-means[i]))
    for ax in axes:
        ax.axvline(.138,c="black",ls="--",label="large-N capacity 0.138")
        ax.set(xlabel="load alpha = P/N",ylim=(0,1.05))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].axhline(.9,c="gray",ls=":")
    axes[0].set(ylabel="mean final overlap",title="Finite-size recall transition")
    axes[1].set(ylabel="fraction with final overlap >= 0.9",title="Nearly complete recall")
    savefig(fig,"hopfield-capacity.png")
    references={100:.186,200:.170,400:.156,800:.150}
    assert all(abs(crossings[n]-references[n])<.02 for n in crossings),crossings
    assert all(a>b for a,b in zip(crossings.values(),list(crossings.values())[1:])),crossings
    return crossings


def modern_figures():
    run=load_json("modern-metrics.json")
    with np.load(ART/"modern/P100.npz") as f:
        values={k:f[k] for k in f.files}
    betas=values["betas"]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,beta in zip(axes,(.003,.03,.3)):
        i=int(np.argmin(abs(betas-beta)))
        weights=values["weights"][i,0]
        ax.bar(np.arange(100),weights,color="#5689b4",width=1)
        ax.set(xlabel="stored pattern index",ylabel="softmax weight",
               title=f"beta={beta:g}; target weight={weights[0]:.3f}")
    fig.suptitle("One fixed query, P=100: temperature concentrates attention")
    fig.subplots_adjust(top=.76,wspace=.35)
    savefig(fig,"attention-weights.png")

    fig,axes=plt.subplots(4,8,figsize=(13,6.5))
    for row in range(4):
        imgs=[values["targets"][row],values["queries"][row]]
        imgs.extend(values["modern"][int(np.argmin(abs(betas-b))),row] for b in (.003,.01,.03,.1,.3))
        imgs.append(values["classical"][row])
        titles=["target","query","beta=.003","beta=.01","beta=.03","beta=.1","beta=.3","classical"]
        for col,(im,title) in enumerate(zip(imgs,titles)):
            show_image(axes[row,col],im.reshape(28,28),title if row==0 else None,spin=True)
    fig.suptitle("P=100: modern weighted averages become a stored digit as beta grows")
    savefig(fig,"modern-temperature.png")

    fig,ax=plt.subplots(figsize=(9,4.5))
    for beta in run["betas"]:
        rows=[r for r in run["rows"] if r["beta"]==beta]
        ax.plot([r["p"] for r in rows],[r["modern_success"] for r in rows],"o-",
                lw=2.5 if beta==.3 else 1,ms=4,label=f"modern beta={beta:g}")
    rows=[r for r in run["rows"] if r["beta"]==.3]
    ax.plot([r["p"] for r in rows],[r["classical_success"] for r in rows],"ks--",label="classical")
    ax.set(xscale="log",xlabel="stored training digits P",ylabel="retrieval success",
           ylim=(-.04,1.06),title="100 matched queries per load; overlap >= 0.95 and nearest stored target")
    ax.legend(fontsize=8,ncol=2)
    ax.grid(alpha=.2)
    savefig(fig,"modern-success.png")

    fig,axes=plt.subplots(4,8,figsize=(12,6))
    index=int(np.argmin(abs(betas-.3)))
    for col in range(8):
        for row,im in enumerate((values["targets"][col],values["queries"][col],
                                values["classical"][col],values["modern"][index,col])):
            show_image(axes[row,col],im.reshape(28,28),spin=True)
    for row,label in enumerate(("target","query","classical","modern beta=.3")):
        axes[row,0].set_ylabel(label,fontsize=10)
    fig.suptitle("First 8 matched stored targets and half-observed queries at P=100")
    savefig(fig,"modern-retrieval.png")


def main():
    rbm_figures()
    crossings=hopfield_figures()
    modern_figures()
    write_json(EVIDENCE/"capacity-crossings.json",
               {"criterion":"first interpolated crossing of mean final overlap below 0.9",
                "crossings":crossings,"within_reference_tolerance":True,"decreases_with_N":True})
    print("Rendered all experiment figures; crossings:",crossings)


if __name__=="__main__":main()
