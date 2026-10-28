"""Query EVERY stored key; block nonlinear Gram products to bound CPU memory."""
import time
import argparse
from PIL import Image,ImageDraw
from common import *

STORES=['outer product','least squares','ReLU','ReLU²','ReLU³']
SIZES=2**np.arange(3,17)
DIMS=[16,32,64]

def capacity(sizes,ratios):
    crossing=np.flatnonzero(ratios>=1)
    if not len(crossing):return None
    i=int(crossing[0])
    if i==0:return float(sizes[0])
    x=np.log(sizes[i-1:i+1]);y=np.log(np.maximum(ratios[i-1:i+1],1e-30))
    return float(np.exp(x[0]-y[0]*(x[1]-x[0])/(y[1]-y[0])))

def evaluate_stores(keys,values):
    m,d=keys.shape
    answers=keys@(keys.T@values)
    outer=np.linalg.norm(answers-values,axis=1)
    w=np.linalg.lstsq(keys,values,rcond=None)[0]
    diagonal=np.ones(m) if m<=d else np.sum(np.linalg.qr(keys,mode='reduced')[0]**2,axis=1)
    # Least-squares signal is H_jj*v_j, not v_j. Dividing an ordinary residual
    # by ||v_j|| would asymptote to 1 and would NOT measure interference/signal.
    least=np.linalg.norm(keys@w-diagonal[:,None]*values,axis=1)/diagonal
    ratios=np.empty((5,m));ratios[0]=outer;ratios[1]=least
    for start in range(0,m,512):
        stop=min(start+512,m)
        similarity=keys[start:stop]@keys.T
        np.maximum(similarity,0,out=similarity)
        for row in range(2,5):
            if row==2:weights=similarity
            elif row==3:weights=similarity*similarity
            else:np.multiply(weights,similarity,out=weights)
            answer=weights@values
            ratios[row,start:stop]=np.linalg.norm(answer-values[start:stop],axis=1)
    return ratios

def glyph():
    """Portable, explicitly drawn 32x32 算 fixture; no platform font dependency."""
    image=Image.new('1',(32,32),0);draw=ImageDraw.Draw(image)
    for points in [[(8,2),(5,7)],[(7,5),(14,5)],[(11,5),(12,9)],[(22,2),(18,7)],[(20,5),(28,5)],[(24,5),(25,9)],
                   [(8,11),(24,11),(24,21),(8,21),(8,11)],[(8,14),(24,14)],[(8,18),(24,18)],
                   [(4,24),(28,24)],[(12,22),(12,27),(9,30)],[(22,22),(22,30)]]:
        draw.line(points,fill=1,width=2)
    return 2*np.asarray(image,dtype=np.float64)-1

def picture_recall():
    picture=glyph();sizes=[64,256,1024,4096];names=['outer product','ReLU','ReLU²','softmax β=8']
    records=[];raw=[]
    for m in sizes:
        draws=[]
        for rep in range(20):
            rng=np.random.default_rng(1780+rep)
            keys=rng.normal(size=(m,64));keys/=np.linalg.norm(keys,axis=1,keepdims=True)
            values=rng.choice([-1.,1.],size=(m,1024));values[0]=picture.ravel()
            similarity=keys@keys[0]
            weights=[similarity,np.maximum(similarity,0),np.maximum(similarity,0)**2,softmax(8*similarity)]
            recalls=np.stack([w@values/w[0] for w in weights])
            ratios=np.linalg.norm(recalls-picture.ravel(),axis=1)/np.linalg.norm(picture)
            draws.append(ratios)
            if rep==0:raw.append(recalls.reshape(4,32,32))
        draws=np.stack(draws)
        records.append({'memories':m,'draws':20,'seeds':list(range(1780,1800)),
                        'single_draw_ratios':draws[0].tolist(),'median_ratios':np.median(draws,axis=0).tolist(),
                        'all_ratios':draws.tolist()})
    report={'dimension':64,'beta':8,'character':'算','fixture':'portable explicit strokes in memory.py, 32x32, ink +1/background -1',
            'stores':names,'runs':records,'environment':environment()}
    np.savez_compressed(ART/'picture-recall.npz',truth=picture,recalls=np.stack(raw))
    write_json(EVIDENCE/'picture-metrics.json',report)
    fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained');xx,yy=np.meshgrid(np.arange(32),np.arange(32))
    for row,(m,recalls) in enumerate(zip(sizes,raw)):
        for col,recall in enumerate(recalls):
            ax=axes[row,col];ratio=records[row]['single_draw_ratios'][col]
            # Dot AREA is linear in the positive recalled pixel. Keep the same
            # scale across panels; cap only extreme marker areas for display.
            area=5*np.clip(recall.ravel(),0,8)
            ax.scatter(xx.ravel(),yy.ravel(),s=area,c='#dc5447' if ratio<1 else '#777777',linewidths=0)
            ax.set(xlim=(-1,32),ylim=(32,-1),aspect='equal',xticks=[],yticks=[],xlabel=f'R = {ratio:.4f}')
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(names[col])
            if col==0:ax.set_ylabel(f'M = {m:,}')
    fig.suptitle('Recall of 算 among random ±1 pictures (first predetermined draw)\nEach answer divided by its OWN signal weight; red indicates interference / signal < 1',fontfamily='DejaVu Sans')
    # The glyph itself is drawn from pixels; use ASCII in the title so no CJK
    # font is needed on a fresh Linux installation.
    fig._suptitle.set_text('Recall of the stored character among random +/-1 pictures\nAnswer divided by its OWN signal weight; red indicates interference / signal < 1')
    savefig(fig,'picture-recall.png')
    m1024=records[2]['median_ratios']
    assert abs(m1024[0]/4-1)<.15 and abs(m1024[1]/np.sqrt(8)-1)<.15
    assert records[3]['median_ratios'][2]>1
    assert max(r['median_ratios'][3] for r in records)<.1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    start=time.perf_counter();folder=ART/'memory';folder.mkdir(exist_ok=True)
    previous=json.loads((EVIDENCE/'memory-metrics.json').read_text()) if (EVIDENCE/'memory-metrics.json').exists() else {}
    reused=0;computed=0
    values=np.zeros((3,3,len(SIZES),5))
    for di,d in enumerate(DIMS):
        for rep in range(3):
            rng=np.random.default_rng(1800+10*d+rep)
            keys=rng.normal(size=(65536,d));keys/=np.linalg.norm(keys,axis=1,keepdims=True)
            targets=rng.normal(size=(65536,d));targets/=np.linalg.norm(targets,axis=1,keepdims=True)
            for mi,m in enumerate(SIZES):
                path=folder/f'd{d}-draw{rep}-M{m}.npz'
                if path.exists() and not args.fresh:
                    ratios=np.load(path)['ratios'];reused+=1
                else:
                    ratios=evaluate_stores(keys[:m],targets[:m]);np.savez_compressed(path,ratios=ratios);computed+=1
                values[di,rep,mi]=np.sqrt(np.mean(ratios**2,axis=1))
                if m>=8192:print(f'd={d} draw={rep} M={m}: R={values[di,rep,mi]}',flush=True)
    median=np.median(values,axis=1)
    capacities=np.array([[capacity(SIZES,median[di,:,si]) for si in range(5)] for di in range(3)])
    slopes=[float(np.polyfit(np.log(DIMS),np.log(capacities[:,si]),1)[0]) for si in range(5)]
    r_slopes=[]
    for si in range(5):
        keep=median[2,:,si]>=.1
        r_slopes.append(float(np.polyfit(np.log(SIZES[keep]),np.log(median[2,keep,si]),1)[0]))
    report={'dimensions':DIMS,'sizes':SIZES.tolist(),'stores':STORES,'independent_draws':3,
            'seeds':'1800+10*d+draw; nested prefixes within a draw',
            'ratio_aggregation':'RMS interference/signal across EVERY stored key, then median across 3 independent draws',
            'ratios_by_dimension_draw_size_store':values.tolist(),'median_ratios':median.tolist(),
            'capacities':capacities.tolist(),'capacity_loglog_slopes':slopes,'d64_ratio_loglog_slopes':r_slopes,
            'slope_region':'R>=0.1 for every store; least-squares curvature close to M=d is retained',
            'seconds':time.perf_counter()-start,'environment':environment()}
    report['cache']={'reused_cases':reused,'newly_computed_cases':computed,'force_recompute_command':'python scripts/memory.py --fresh'}
    report['full_uncached_run_seconds']=report['seconds'] if reused==0 else previous.get('full_uncached_run_seconds')
    write_json(folder/'settings.json',{'dimensions':DIMS,'sizes':SIZES.tolist(),'seeds':report['seeds'],
                                     'aggregation':report['ratio_aggregation'],'stores':STORES})
    write_json(EVIDENCE/'memory-metrics.json',report)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');colors=['#3285aa','#e48b27','#669b42','#a55ca3','#d45853']
    theory=[np.sqrt(SIZES/64),np.sqrt(np.maximum(SIZES/64-1,0)),np.sqrt(SIZES/128),np.sqrt(3*SIZES/(2*64**2)),np.sqrt(15*SIZES/(2*64**3))]
    for si,(name,color) in enumerate(zip(STORES,colors)):
        axes[0].loglog(SIZES,np.maximum(median[2,:,si],1e-15),'-o',color=color,label=name)
        axes[0].loglog(SIZES,np.maximum(theory[si],1e-15),'--',color=color,alpha=.5)
        axes[1].loglog(DIMS,capacities[:,si],'-o',color=color,label=f'{name}: slope {slopes[si]:.2f}')
    axes[0].axhline(1,color='black',ls=':');axes[0].set(xlabel='stored memories M',ylabel='interference / signal R',ylim=(1e-3,1e2),title='d=64; solid measured, dashed Gaussian predictions')
    axes[1].set(xlabel='dimension d',ylabel='capacity at R=1 (log-log interpolation)',xticks=DIMS,title='Nonlinearity changes the capacity exponent')
    from matplotlib.ticker import FixedLocator,NullLocator,FixedFormatter
    axes[1].xaxis.set_major_locator(FixedLocator(DIMS));axes[1].xaxis.set_major_formatter(FixedFormatter(['16','32','64']))
    axes[1].xaxis.set_minor_locator(NullLocator())
    for ax in axes:ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
    savefig(fig,'memory-capacity.png')
    print('Capacities:',capacities,'slopes:',slopes,flush=True)
    picture_recall()
    assert 50<capacities[2,0]<85 and 100<capacities[2,1]<170
    assert max(median[2,SIZES<=64,1])<1e-10
    assert all(abs(s-1)<.1 for s in slopes[:3]) and abs(slopes[3]-2)<.15 and slopes[4]>2.5

if __name__=='__main__':main()
