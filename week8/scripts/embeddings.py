"""Leave-one-pair-out geometry and the seed-1730 shuffled-row control."""
from common import *
from safetensors.numpy import load_file
from tokenizers import Tokenizer

RELATIONS={
 'Country → capital':[('iran','tehran'),('turkey','ankara'),('russia','moscow'),('china','beijing'),('france','paris'),('sweden','stockholm'),('japan','tokyo'),('peru','lima'),('kenya','nairobi')],
 'Male → female':[('king','queen'),('man','woman'),('he','she'),('brother','sister'),('father','mother'),('son','daughter'),('uncle','aunt'),('boy','girl'),('husband','wife')],
 'Adjective → comparative':[('big','bigger'),('small','smaller'),('fast','faster'),('dark','darker'),('cold','colder'),('old','older'),('strong','stronger'),('long','longer'),('tall','taller')]}

def cosines(table,ids):
    offsets=table[ids[:,1]]-table[ids[:,0]]
    other=(offsets.sum(axis=0,keepdims=True)-offsets)/8
    return np.sum(offsets*other,axis=1)/(np.linalg.norm(offsets,axis=1)*np.linalg.norm(other,axis=1))

def main():
    raw=load_file(DATA/'model.safetensors');print({k:v.shape for k,v in raw.items()},flush=True)
    table=next(array for array in raw.values() if array.shape==(29528,256)).astype(np.float64)
    table/=np.maximum(np.linalg.norm(table,axis=1,keepdims=True),1e-30)
    tok=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    def word_id(word):
        values=tok.encode(word,add_special_tokens=False).ids
        assert len(values)==1,(word,values)
        return values[0]
    shuffled=table[np.random.default_rng(1730).permutation(len(table))]
    report={'table_shape':list(table.shape),'revision':'bf8b056','shuffle_seed':1730,'relations':{},'environment':environment()}
    fig,axes=plt.subplots(1,3,figsize=(16,7),layout='constrained')
    for ax,(name,pairs) in zip(axes,RELATIONS.items()):
        ids=np.array([[word_id(a),word_id(b)] for a,b in pairs]);trained=cosines(table,ids);control=cosines(shuffled,ids)
        many=[float(cosines(table[np.random.default_rng(1730+i).permutation(len(table))],ids).mean()) for i in range(20)]
        report['relations'][name]={'pairs':pairs,'token_ids':ids.tolist(),'cosines':trained.tolist(),'control_cosines':control.tolist(),'mean_cosine':float(trained.mean()),'control_mean_cosine':float(control.mean()),'twenty_shuffle_means':many}
        for i,(a,b) in enumerate(pairs):
            x=(i-4)*.34
            for value,sign,color in [(trained[i],1,'#168aad'),(control[i],-1,'#999999')]:
                ax.quiver(x,0,value,sign*np.sqrt(max(0,1-value*value)),angles='xy',scale_units='xy',scale=1,color=color,width=.006)
            ax.text(x,-1.17,f'{a}\n→ {b}',rotation=60,ha='right',va='top',fontsize=8)
        ax.axhline(0,color='black',lw=.8);ax.quiver(-1.6,0,3.7,0,angles='xy',scale_units='xy',scale=1,color='black',width=.004)
        ax.set(xlim=(-1.8,2.5),ylim=(-1.9,1.3),aspect='equal',title=f'{name}\nmean {trained.mean():.3f}; shuffled {control.mean():.3f}',yticks=[],xticks=[])
        assert trained.mean()>=.3 and abs(control.mean())<=.15
    fig.suptitle('Each unit arrow: angle to the OTHER eight offsets (not to itself)\nBlue above: pretrained table; grey below: identical rows, shuffled word assignments')
    savefig(fig,'embedding-relations.png');write_json(EVIDENCE/'embedding-metrics.json',report)
    print({name:(v['mean_cosine'],v['control_mean_cosine']) for name,v in report['relations'].items()},flush=True)

if __name__=='__main__':main()
