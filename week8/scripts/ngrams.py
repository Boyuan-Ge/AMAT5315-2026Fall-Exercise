"""Exact sparse n-gram counts, training-only smoothing selection and Markov tests."""
import argparse
import time
from common import *
from prepare_data import tokenizer

V=512

def codes(tokens,n,eos=2):
    padded=np.pad(np.asarray(tokens,np.uint64),(n-1,0),constant_values=eos)
    out=np.zeros(len(tokens),np.uint64)
    for lag in range(n):out=(out<<np.uint64(9))|padded[lag:lag+len(tokens)]
    return out

def lookup(keys,values,queries):
    index=np.searchsorted(keys,queries)
    safe=np.minimum(index,len(keys)-1)
    return np.where((index<len(keys))&(keys[safe]==queries),values[safe],0)

class Ngram:
    def __init__(self,n,keys,counts,contexts,totals,k):
        self.n=n;self.keys=keys;self.counts=counts;self.contexts=contexts;self.totals=totals;self.k=k
    @classmethod
    def fit(cls,tokens,n,k=0.):
        key,count=np.unique(codes(tokens,n),return_counts=True)
        contexts,starts=np.unique(key>>np.uint64(9),return_index=True)
        totals=np.add.reduceat(count,starts)
        return cls(n,key,count,contexts,totals,k)
    def score_codes(self,queries,k=None):
        count=lookup(self.keys,self.counts,queries)
        total=lookup(self.contexts,self.totals,queries>>np.uint64(9))
        k=self.k if k is None else k
        return np.log(total+k*V)-np.log(count+k),count==0
    def probabilities(self,history):
        context=0
        if self.n>1:
            for token in ([2]*max(0,self.n-1-len(history))+list(history))[-self.n+1:]:context=(context<<9)|int(token)
        query=(np.uint64(context)<<np.uint64(9))|np.arange(V,dtype=np.uint64)
        c=lookup(self.keys,self.counts,query);total=lookup(self.contexts,self.totals,np.array([context],np.uint64))[0]
        return (c+self.k)/(total+self.k*V)
    def save(self):np.savez_compressed(ART/f'ngram-{self.n}.npz',n=self.n,keys=self.keys,counts=self.counts,contexts=self.contexts,totals=self.totals,k=self.k)
    @classmethod
    def load(cls,n):
        f=np.load(ART/f'ngram-{n}.npz');return cls(n,f['keys'],f['counts'],f['contexts'],f['totals'],float(f['k']))

def generate(model,n=60,seed=1848,exclude_eos=True,history=None):
    rng=np.random.default_rng(seed);history=[2] if history is None else list(history);out=[]
    for _ in range(n):
        p=np.ones(V)/V if model is None else model.probabilities(history)
        if exclude_eos:p[2]=0;p/=p.sum()
        token=int(rng.choice(V,p=p));out.append(token);history.append(token)
    return out

def openings():
    sp=tokenizer();train=np.load(ART/'train-tokens.npy')
    first_end=int(np.flatnonzero(train==2)[0])
    history=train[first_end-1:first_end+1].tolist()
    output={}
    for label,model in [('uniform',None),('unigram',Ngram.load(1)),('bigram',Ngram.load(2)),('trigram',Ngram.load(3))]:
        ids=generate(model,history=history)
        output[label]={'seed':1848,'tokens':ids,'text':sp.decode(ids),'eos_excluded':True,'initial_history':history}
    return output

def main():
    start=time.perf_counter();train=np.load(ART/'train-tokens.npy');valid=np.load(ART/'validation-tokens.npy')
    data=json.loads((EVIDENCE/'data-provenance.json').read_text());chars=data['validation']['characters']
    cut=int(.9*len(train));candidates=[.0001,.001,.01,.1,1.]
    report={'vocabulary':V,'smoothing_candidates':candidates,'smoothing_split_token':cut,
            'smoothing_rule':'fit first 90% of training tokens; choose k on last 10%; refit all training tokens',
            'validation_tokens':len(valid),'validation_characters':chars,'models':[],'environment':environment()}
    uniform=np.log(V)
    report['uniform']={'loss_nats':float(uniform),'perplexity':float(V),'bits_per_character':float(uniform*len(valid)/(np.log(2)*chars))}
    for n in range(1,7):
        fit=Ngram.fit(train[:cut],n)
        tuning=codes(train,n)[cut:]
        losses=[float(fit.score_codes(tuning,k)[0].mean()) for k in candidates]
        k=candidates[int(np.argmin(losses))]
        model=Ngram.fit(train,n,k);model.save()
        scores,unseen=model.score_codes(codes(valid,n));np.save(ART/f'ngram-{n}-validation-loss.npy',scores)
        row={'n':n,'k':k,'tuning_losses':losses,'loss_nats':float(scores.mean()),
             'perplexity':float(np.exp(scores.mean())),'bits_per_character':float(scores.sum()/(np.log(2)*chars)),
             'unseen_fraction':float(unseen.mean()),'observed_ngrams':len(model.keys),'observed_contexts':len(model.contexts),
             'possible_contexts':int(V**(n-1))}
        report['models'].append(row);print(row,flush=True)
    model=Ngram.load(2);p=np.stack([model.probabilities([i]) for i in range(V)])
    initial=p[2]
    # Explicitly enumerate all 512^length products, including all 134,217,728
    # triples. Chunk only to avoid a 1 GiB temporary; no row-sum shortcut.
    sums=[float(initial.sum()),float(np.sum(initial[:,None]*p)),
          float(sum(np.sum(initial[a]*p[a,:,None]*p) for a in range(V)))]
    report['normalization']={'sequence_lengths':[1,2,3],'explicit_sums':sums,
                             'absolute_errors':np.abs(np.array(sums)-1).tolist(),'start_context':'EOS'}
    length=250000;rng=np.random.default_rng(1849);uniforms=rng.random(length);cdf=np.cumsum(p,axis=1);cdf[:,-1]=1
    previous=2;contexts=np.empty(length,np.int32);draws=np.empty(length,np.int32)
    for i,u in enumerate(uniforms):
        contexts[i]=previous;previous=int(np.searchsorted(cdf[previous],u));draws[i]=previous
    counts=np.bincount(contexts*V+draws,minlength=V*V).reshape(V,V)
    frequent=np.argsort(np.bincount(train,minlength=V))[-20:][::-1]
    report['bigram_sampling']={'tokens':length,'seed':1849,'contexts':frequent.tolist(),
                               **pooled_chisquare(counts[frequent],p[frequent])}
    np.savez_compressed(ART/'bigram-sampling.npz',contexts=contexts,tokens=draws,counts=counts,probabilities=p)
    report['openings']=openings()
    report['seconds']=time.perf_counter()-start;write_json(EVIDENCE/'ngram-metrics.json',report);write_json(ART/'ngram-settings.json',report)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');rows=report['models']
    axes[0].plot([r['n'] for r in rows],[r['bits_per_character'] for r in rows],'-o',color='#168aad')
    axes[0].set(xlabel='n (context = n-1 tokens)',ylabel='held-out bits / character',xticks=range(1,7),title='Longer context eventually hurts')
    axes[1].plot([r['n'] for r in rows],[100*r['unseen_fraction'] for r in rows],'-o',color='#e08030')
    axes[1].set(xlabel='n',ylabel='held-out n-grams absent in training (%)',xticks=range(1,7),title='The curse of dimensionality')
    for ax in axes:ax.grid(alpha=.2)
    savefig(fig,'ngram-ladder.png')
    assert max(report['normalization']['absolute_errors'])<1e-12
    assert report['bigram_sampling']['p_value']>.001

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-openings-only',action='store_true');args=parser.parse_args()
    if args.refresh_openings_only:
        report=json.loads((EVIDENCE/'ngram-metrics.json').read_text());report['openings']=openings();write_json(EVIDENCE/'ngram-metrics.json',report)
    else:main()
