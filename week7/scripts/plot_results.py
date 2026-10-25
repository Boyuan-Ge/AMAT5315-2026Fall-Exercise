"""Render every committed figure and required movie from the measured raw runs."""
import argparse
import importlib.util
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from common import *

COLORS=np.array(['#168aad','#f28436','#a86db3','#2a9d70'])
CENTRES=np.array([[3,0],[0,3],[-3,0],[0,-3]])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def plane(ax,title='',limits=5):
    ax.set(xlim=(-limits,limits),ylim=(-limits,limits),aspect='equal',title=title,xlabel='x',ylabel='y')
    for c in CENTRES:ax.add_patch(Circle(c,.6,fill=False,color='0.65',ls='--',lw=.7))
    ax.grid(alpha=.15)

def cloud(ax,points,title='',colors=None):
    plane(ax,title)
    if colors is None:colors=COLORS[np.argmin(np.sum((points[:,None]-CENTRES)**2,axis=-1),axis=1)]
    ax.scatter(*points.T,s=3,c=colors,alpha=.55,rasterized=True)

def loss_panel(ax,kind,until=25000):
    loss=np.load(ART/kind/'loss.npy')
    means=loss.reshape(-1,100).mean(axis=1)
    x=np.arange(100,25001,100)
    select=x<=until
    ax.plot(x[select],means[select],color='#168aad',lw=1)
    ax.set(xlim=(0,25000),ylim=(0,.65 if kind=='diffusion' else 5),xlabel='training update',ylabel='MSE / coordinate',title=f'{kind.capitalize()} loss (100-update means)')
    ax.grid(alpha=.2)

def mixture_figures():
    folder=ART/'mixture'
    f=np.load(folder/'forward.npz');states=f['states'];colors=COLORS[f['components']]
    fig,axes=plt.subplots(1,6,figsize=(17,3.2),layout='constrained')
    for ax,t in zip(axes,[0,50,150,300,600,1000]):cloud(ax,states[t],f't = {t}',colors)
    fig.suptitle('Forward diffusion: the SAME 2,000 points, fixed coordinate limits')
    savefig(fig,'forward-stages.png')
    stage=np.load(folder/'training-stages.npz');updates=stage['updates'];paths=stage['paths']
    fig=plt.figure(figsize=(14,7),layout='constrained');grid=fig.add_gridspec(2,4,height_ratios=[2,1])
    for column,k in enumerate([0,3,6,9]):
        ax=fig.add_subplot(grid[0,column]);draw_paths(ax,paths[k],f'update {updates[k]:,}')
    ax=fig.add_subplot(grid[1,:]);loss_panel(ax,'diffusion')
    for k in [0,3,6,9]:ax.axvline(updates[k],color='0.4',ls=':',lw=.8)
    fig.suptitle('Learning changes the weights; all stages reuse identical starts AND reverse noise')
    savefig(fig,'training-stages.png')
    fig,axes=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
    for ax,kind in zip(axes,['diffusion','flow']):loss_panel(ax,kind)
    savefig(fig,'training-loss.png')
    s=np.load(folder/'samplers.npz');report=json.loads((EVIDENCE/'mixture-metrics.json').read_text())
    fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for ax,label,points in zip(axes,['Exact target','DDPM, reverse noise A','DDPM, reverse noise B','DDIM, repeated paths identical'],[s['exact'],s['ddpm'][-1],s['ddpm2'][-1],s['ddim'][-1]]):cloud(ax,points,label)
    fig.suptitle('2,000 samples per panel; all learned samplers use 1,000 evaluations')
    savefig(fig,'sampler-samples.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,method in zip(axes,['ddim','ddpm']):
        plane(ax,'DDIM: two identical runs' if method=='ddim' else 'DDPM: shared starts, independent reverse noise')
        for i in range(6):
            color=COLORS[i%4];path=s[method][:,i]
            ax.plot(*path.T,color=color,alpha=.75,lw=.8)
            path2=path if method=='ddim' else s['ddpm2'][:,i]
            ax.plot(*path2.T,color=color,ls='--',lw=.8,alpha=.8)
            ax.scatter(*path[0],s=25,facecolor='white',edgecolor=color,zorder=4)
            ax.scatter(*path[-1],s=25,color=color,zorder=4)
            ax.scatter(*path2[-1],s=32,marker='x',color=color,zorder=4)
    fraction=report['sampling_2000']['ddpm_same_cluster_fraction']
    fig.suptitle(f'DDIM repeat max difference = 0; DDPM same-cluster fraction = {fraction:.4f}')
    savefig(fig,'determinism-paths.png')
    fields=np.load(folder/'velocity-fields.npz');grid=fields['grid'];rng=np.random.default_rng(7129)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for k,(ax,t) in enumerate(zip(axes,[.1,.5,.9])):
        sample=(1-t)*(CENTRES[rng.integers(0,4,2000)]+.3*rng.normal(size=(2000,2)))+t*rng.normal(size=(2000,2))
        ax.scatter(*sample.T,s=2,color='0.75',alpha=.45)
        # Clip BOTH arrow lengths identically for visibility, not the measured errors.
        for values,color,label in [(fields['exact'][k],'#e78625','exact'),(fields['learned'][k],'#1675b8','learned')]:
            length=np.linalg.norm(values,axis=1)
            clipped=values*np.minimum(1,2/np.maximum(length,1e-9))[:,None]
            support=fields['support'][k]
            ax.quiver(*grid[support].T,*clipped[support].T,color=color,label=label,angles='xy',scale_units='xy',scale=7,width=.003,alpha=.65)
            ax.quiver(*grid[~support].T,*clipped[~support].T,color=color,angles='xy',scale_units='xy',scale=7,width=.002,alpha=.15)
        errors=report['velocity_errors'][str(t)]
        ax.set(xlim=(-4.3,4.3),ylim=(-4.3,4.3),aspect='equal',xlabel='x',ylabel='y',title=f's={t}: 99%-region grid RMS {errors["grid_99percent_region_relative_rms"]:.3f}\nmarginal RMS {errors["marginal_sample_relative_rms"]:.3f}; full-grid RMS {errors["grid_relative_rms"]:.3f}')
        ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Velocities point toward increasing s; generation moves AGAINST them\nStrong arrows: union of 99% component disks; faint arrows: low-density extrapolation')
    savefig(fig,'velocity-fields.png')
    comparison=report['comparisons'];fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    keys=['energy_distance','max_population_error','max_width_error']
    for method,color in zip(['ddpm','ddim','flow'],['#d87829','#168aad','#8557b0']):
        for exact in [False,True]:
            rows=[r for r in comparison if r['method']==method and r['exact_predictor']==exact]
            for ax,key in zip(axes,keys):
                y=np.array([r[key] if r[key] is not None else np.nan for r in rows])
                if key=='energy_distance':y=np.maximum(y,1e-4)
                ax.plot([r['steps'] for r in rows],y,marker='o' if not exact else None,color=color,ls='--' if exact else '-',alpha=.5 if exact else 1,label=method.upper()+(' exact' if exact else ' learned'))
    axes[0].axhspan(1e-4,report['null_95_quantile'],color='0.8',alpha=.7,label='exact/exact null 95%')
    axes[1].axhspan(0,report['cluster_fraction_two_se'],color='0.8',alpha=.7,label='2 binomial SE')
    axes[2].axhline(.03,color='0.5',ls=':',label='width error 0.03')
    for ax,title in zip(axes,['Energy-distance U-statistic (display floor 1e-4)','Largest |cluster fraction - 0.25|','Largest |coordinate SD - 0.30|']):
        ax.set(xscale='log',xlabel='predictor evaluations / sample',title=title)
        ax.axvline(20,color='0.3',ls=':',lw=1);ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[2].set_yscale('log')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);axes[2].legend(fontsize=7)
    fig.suptitle('4,000 samples per method/grid; solid = trained network, dashed = analytic predictor')
    savefig(fig,'transport-cost.png')
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for row,steps in enumerate([1,2]):
        for ax,method in zip(axes[row],['ddpm','ddim','flow']):
            points=np.load(folder/f'{method}-{steps}-exact.npy');cloud(ax,points,f'{method.upper()}, {steps} step(s), exact predictor')
    fig.suptitle('Coarse-grid negative controls: failure remains even with NO learning error')
    savefig(fig,'exact-coarse-controls.png')

def draw_paths(ax,path,title):
    plane(ax,title)
    for i in range(path.shape[1]):
        color=COLORS[i%4];ax.plot(*path[:,i].T,color=color,lw=.6,alpha=.65)
        ax.scatter(*path[0,i],facecolor='white',edgecolor=color,s=18,zorder=4)
        ax.scatter(*path[-1,i],color=color,s=18,zorder=4)
    ax.text(.02,.02,f'max |coordinate| = {np.max(abs(path)):.1f}',transform=ax.transAxes,fontsize=8)

def movies():
    folder=ART/'mixture';forward=np.load(folder/'forward.npz');s=np.load(folder/'samplers.npz')
    for name,states,colors,descending in [('forward-diffusion.gif',forward['states'],COLORS[forward['components']],False),('reverse-ddpm.gif',s['ddpm'],COLORS[np.argmin(np.sum((s['ddpm'][-1,:,None]-CENTRES)**2,axis=-1),axis=1)],True)]:
        fig,ax=plt.subplots(figsize=(5,5),layout='constrained');plane(ax)
        scatter=ax.scatter(*states[0].T,s=4,c=colors,alpha=.6);title=ax.set_title('')
        def update(i):
            scatter.set_offsets(states[i]);title.set_text(f'{"Reverse DDPM" if descending else "Forward diffusion"}: t={1000-i if descending else i}\nSame 2,000 particles; fixed [-5,5] axes')
            return scatter,title
        animation=FuncAnimation(fig,update,frames=np.arange(0,1001,20),interval=100)
        animation.save(EVIDENCE/name,writer=PillowWriter(fps=10),dpi=90);plt.close(fig)
    stage=np.load(folder/'training-stages.npz');fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    def update_stage(k):
        for ax in axes:ax.clear()
        draw_paths(axes[0],stage['paths'][k],f'Predictor after update {stage["updates"][k]:,}')
        loss_panel(axes[1],'diffusion',int(stage['updates'][k]))
        fig.suptitle('10 fixed starts + fixed reverse noise; only learned weights change')
    animation=FuncAnimation(fig,update_stage,frames=len(stage['updates']),interval=900)
    animation.save(EVIDENCE/'training-progress.gif',writer=PillowWriter(fps=1),dpi=100);plt.close(fig)

def photograph_figures():
    folder=ART/'photographs';states=np.load(folder/'generation.npy');levels=[1000,800,600,400,200,100,50,0]
    fig,axes=plt.subplots(4,len(levels),figsize=(14,7),layout='constrained')
    for row in range(4):
        for column,level in enumerate(levels):show_rgb(axes[row,column],states[1000-level,row],f't={level}' if row==0 else None)
    fig.suptitle('Four NEW photographs: 1,000-step JAX DDPM with a fixed pretrained U-Net\nNo state clipping; clipping applied ONLY to these display pixels')
    savefig(fig,'photograph-stages.png')
    # Reuse Week 6's actual image loader AND native-pixel gallery helper.
    spec=importlib.util.spec_from_file_location('week6_gallery',ROOT.parent/'week6/scripts/common.py')
    gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery)
    if (INPUT/'cifar10/cifar-10-binary.tar.gz').exists():gallery.INPUT=INPUT
    _,_,real,labels=gallery.load_cifar()
    fig,axes=plt.subplots(2,4,figsize=(10,5.8),layout='constrained')
    for i in range(4):
        gallery.show_image(axes[0,i],real[i],f'Real test #{i}: {gallery.CLASSES[labels[i]]}')
        show_rgb(axes[1,i],states[-1,i],f'Generated #{i+1}, seed 7200')
    fig.suptitle('Real CIFAR-10 examples (Week 6 gallery code) versus new unconditional samples')
    savefig(fig,'real-vs-generated.png')
    data=np.load(folder/'inpainting.npz');mse=np.array(json.loads((EVIDENCE/'photograph-metrics.json').read_text())['inpainting']['hidden_mse_by_seed_then_image'])
    fig,axes=plt.subplots(4,7,figsize=(16,9),layout='constrained')
    vmax=float(data['variance'].mean(axis=1).max())
    for row in range(4):
        show_rgb(axes[row,0],data['truth'][row],f'Test image {row+1}: withheld truth')
        show_rgb(axes[row,1],data['observed'][row],'Observation (hole = zero)')
        axes[row,1].add_patch(Rectangle((5.5,5.5),20,20,fill=False,edgecolor='#e87625',lw=1))
        for i,seed in enumerate(data['seeds']):show_rgb(axes[row,i+2],data['states'][i,-1,row],f'Seed {seed}\nhidden MSE={mse[i,row]:.4f}')
        image=axes[row,6].imshow(data['variance'][row].mean(axis=0),vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
        axes[row,6].set_title('Across-seed variance\nRGB-channel mean',fontsize=9);axes[row,6].set_xticks([]);axes[row,6].set_yticks([])
    fig.colorbar(image,ax=axes[:,6],shrink=.6,label='variance in [-1,1] pixel units')
    fig.suptitle('RePaint: observations + mask ONLY; 4 independent seeds per test image\nObserved pixels change by exactly 0; all measured disagreement lies in the hidden square')
    savefig(fig,'inpainting-results.png')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-movies',action='store_true');args=parser.parse_args()
    mixture_figures();photograph_figures()
    if not args.no_movies:movies()
    print('All figures and requested movies rendered.',flush=True)

if __name__=='__main__':main()
