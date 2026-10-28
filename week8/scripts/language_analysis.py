"""Independent sampler distributions, actual draws, context use and self-consistency."""
import functools
from language_models import *
from ngrams import Ngram,generate
from prepare_data import tokenizer

@jax.jit
def opening_logits(p,context,count):return transformer(p,context)[0,count-1]

def common_rng_opening(p,history,seed=1848):
    # The five comparison openings share the SAME PCG64 uniforms, as well as
    # the same numerical seed and EOS-ended training context.
    context=np.full((1,64),2,np.int32);context[0,:len(history)]=history
    count=len(history);rng=np.random.default_rng(seed);ids=[]
    for _ in range(60):
        logits=np.asarray(opening_logits(p,jnp.asarray(context),jnp.int32(count)))
        probability=exact_distribution(logits,exclude_eos=True)
        token=int(rng.choice(512,p=probability));ids.append(token)
        if count<64:context[0,count]=token;count+=1
        else:context=np.concatenate((context[:,1:],np.array([[token]],np.int32)),axis=1)
    return ids

def visible(text):
    return ''.join('□' if ch=='\ufffd' or (ord(ch)<32 and ch!='\n') else ch for ch in text)

def token_label(sp,v):return visible(sp.id_to_piece(int(v))).replace('▁','·')

def exact_distribution(logits,temperature=1.,mode='temperature',cutoff=0.,exclude_eos=False):
    probability=softmax(np.asarray(logits,np.float64)/temperature)
    if mode=='top-k':
        keep=np.zeros(V,bool);keep[np.argsort(-probability,kind='stable')[:int(cutoff)]]=True
    elif mode=='top-p':
        order=np.argsort(-probability,kind='stable');sorted_p=probability[order]
        keep=np.zeros(V,bool);keep[order[np.cumsum(sorted_p)-sorted_p<cutoff]]=True
    elif mode=='min-p':keep=probability>=cutoff*probability.max()
    else:keep=np.ones(V,bool)
    if exclude_eos:keep[2]=False
    probability=np.where(keep,probability,0)
    return probability/probability.sum()

def sampling_distribution(logits,temperature=1.,mode='temperature',cutoff=0.,exclude_eos=False):
    probability=jax.nn.softmax(logits/temperature,axis=-1)
    if mode in ('top-k','top-p'):
        order=jnp.argsort(-probability,axis=-1,stable=True)
        ranked=jnp.take_along_axis(probability,order,axis=-1)
        if mode=='top-k':sorted_keep=jnp.broadcast_to(jnp.arange(V)<int(cutoff),ranked.shape)
        else:sorted_keep=jnp.cumsum(ranked,axis=-1)-ranked<cutoff
        inverse=jnp.argsort(order,axis=-1)
        keep=jnp.take_along_axis(sorted_keep,inverse,axis=-1)
    elif mode=='min-p':keep=probability>=cutoff*jnp.max(probability,axis=-1,keepdims=True)
    else:keep=jnp.ones_like(probability,dtype=bool)
    if exclude_eos:keep=keep.at[...,2].set(False)
    probability=jnp.where(keep,probability,0)
    return probability/jnp.sum(probability,axis=-1,keepdims=True)

def inverse_cdf(probability,uniforms):
    cumulative=jnp.cumsum(probability,axis=-1).at[...,-1].set(1)
    if probability.ndim==1:return jnp.searchsorted(cumulative,uniforms,side='right')
    return jnp.sum(uniforms[:,None]>=cumulative,axis=-1)

@functools.partial(jax.jit,static_argnames=('batch','steps','temperature','mode','cutoff','exclude_eos'))
def generate_sequences(p,prefix,key,batch=1,steps=60,temperature=1.,mode='temperature',cutoff=0.,exclude_eos=False):
    count=jnp.int32(len(prefix));context=jnp.full((batch,CONTEXT),2,jnp.int32).at[:,:len(prefix)].set(prefix[None,:])
    def body(carry,_):
        context,count,rng=carry;rng,k=random.split(rng)
        logits=transformer(p,context)[:,count-1]
        base_logp=jax.nn.log_softmax(logits,axis=-1);base_p=jnp.exp(base_logp)
        entropy=-jnp.sum(base_p*base_logp,axis=-1)
        probability=sampling_distribution(logits,temperature,mode,cutoff,exclude_eos)
        token=inverse_cdf(probability,random.uniform(k,(batch,)))
        surprise=-jnp.take_along_axis(base_logp,token[:,None],axis=-1)[:,0]
        context=lax.cond(count<CONTEXT,lambda c:c.at[:,count].set(token),
                         lambda c:jnp.concatenate((c[:,1:],token[:,None]),axis=1),context)
        return (context,jnp.minimum(count+1,CONTEXT),rng),(token,surprise,entropy)
    _,(tokens,surprise,entropy)=lax.scan(body,(context,count,key),None,length=steps)
    return tokens.T,surprise.T,entropy.T

def main():
    sp=tokenizer();p=load_model('transformer');ngram=json.loads((EVIDENCE/'ngram-metrics.json').read_text())
    trained=json.loads((EVIDENCE/'transformer-training.json').read_text())
    scores=np.load(ART/'transformer-validation-loss.npy').astype(float)
    windows=scores[:len(scores)//64*64].reshape(-1,64)
    mean=windows.mean(axis=0);se=windows.std(axis=0,ddof=1)/np.sqrt(len(windows))
    report={'scoring':'all 1078996 held-out tokens; previous token plus growing context within consecutive target windows of 64; EOS only for stream start',
            'context_curve':{'windows':len(windows),'loss_by_position':mean.tolist(),'standard_error_by_position':se.tolist(),
                             'mean_positions_32_to_64':float(mean[31:].mean()),'last_eight_mean':float(mean[-8:].mean())},
            'openings':dict(ngram['openings']),'environment':environment()}
    history=ngram['openings']['trigram']['initial_history'];opening=common_rng_opening(p,history)
    report['openings']['transformer']={'seed':1848,'tokens':opening,'text':visible(sp.decode(opening)),'eos_excluded':True,'initial_history':history}
    report['opening_protocol']='60 draws each; seed 1848 and identical NumPy PCG64 uniforms; shared real training-story boundary [last pre-EOS token,EOS]; EOS excluded and probabilities renormalized for this opening comparison ONLY'
    for row in report['openings'].values():row['text']=visible(row['text'])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    labels=['uniform','unigram','bigram','trigram','4-gram','MLP','transformer']
    mlp_record=json.loads((EVIDENCE/'mlp-training.json').read_text())
    bpc=[ngram['uniform']['bits_per_character']]+[r['bits_per_character'] for r in ngram['models'][:4]]+[mlp_record['bits_per_character'],trained['bits_per_character']]
    bars=axes[0].bar(labels,bpc,color=['#bbbbbb']*5+['#e18b34','#168aad'])
    axes[0].bar_label(bars,fmt='%.3f',fontsize=9);axes[0].tick_params(axis='x',rotation=35)
    axes[0].set(ylabel='held-out bits / character',title='Same stories, tokenizer, targets and character denominator',ylim=(0,5))
    for kind,color in [('mlp','#e18b34'),('transformer','#168aad')]:
        record=json.loads((EVIDENCE/f'{kind}-training.json').read_text());values=record['loss_by_50_steps']
        axes[1].plot(50*np.arange(1,len(values)+1),values,label=kind,color=color)
    axes[1].set(xlabel='training update',ylabel='mean training loss per token',title='Prescribed budgets, no extra updates');axes[1].legend();axes[1].grid(alpha=.2)
    savefig(fig,'language-model-comparison.png')
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    x=np.arange(1,65);ax.plot(x,mean,color='#168aad',label='transformer');ax.fill_between(x,mean-2*se,mean+2*se,color='#168aad',alpha=.2,label='±2 window-level SE')
    for n,color in [(2,'#e18b34'),(3,'#8a64af')]:ax.axhline(ngram['models'][n-1]['loss_nats'],color=color,ls='--',label=f'{n}-gram mean loss')
    ax.set(xlabel='tokens read within this 64-token window',ylabel='held-out nats / token',title=f'Context helps: mean of last 8 positions = {mean[-8:].mean():.3f}')
    ax.legend();ax.grid(alpha=.2);savefig(fig,'context-use.png')
    prefix=jnp.array([2]+sp.encode('Once upon a time, a little girl found a',out_type=int))
    logits=np.asarray(transformer(p,prefix[None])[0,-1],dtype=float)
    settings=[('T=1',1.,'temperature',0.),('T=0.5',.5,'temperature',0.),('T=0.8',.8,'temperature',0.),
              ('top-k=3, T=1',1.,'top-k',3.),('top-p=0.8, T=1',1.,'top-p',.8),('min-p=0.2, T=1',1.,'min-p',.2),
              ('top-k=20, T=0.8',.8,'top-k',20.),('top-p=0.9, T=0.8',.8,'top-p',.9),('min-p=0.1, T=0.7',.7,'min-p',.1)]
    tests=[];top=np.argsort(-softmax(logits))[:24]
    fig,axes=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    for i,(label,temp,mode,cutoff) in enumerate(settings):
        expected=exact_distribution(logits,temp,mode,cutoff)
        actual=np.asarray(sampling_distribution(jnp.asarray(logits,jnp.float32),temp,mode,cutoff))
        u=random.uniform(random.PRNGKey(1850+i),(100000,));draws=np.asarray(inverse_cdf(jnp.asarray(actual),u))
        counts=np.bincount(draws,minlength=V)
        test={'label':label,'temperature':temp,'mode':mode,'cutoff':cutoff,'draws':100000,'seed':1850+i,
              'max_probability_error_numpy_vs_jax':float(np.max(abs(actual-expected))),**pooled_chisquare(counts,expected)}
        tests.append(test);print('sampler',test,flush=True)
        ax=axes.ravel()[i];ax.bar(np.arange(len(top)),counts[top]/100000,color='#168aad',alpha=.55,label='100,000 draws')
        ax.plot(np.arange(len(top)),expected[top],'o',ms=3,color='#e18b34',label='exact target')
        ax.set(xticks=np.arange(len(top)),xticklabels=[token_label(sp,v) for v in top],title=f'{label}\nχ² p={test["p_value"]:.4f}',ylabel='probability')
        ax.tick_params(axis='x',rotation=75,labelsize=7)
        if i==0:ax.legend(fontsize=7)
    fig.suptitle('Fixed broad context: "Once upon a time, a little girl found a"\nTarget computed independently in NumPy; rare expected categories pooled for Pearson tests')
    savefig(fig,'sampler-tests.png')
    cold=np.asarray(sampling_distribution(jnp.asarray(logits,jnp.float32),.0001))
    cold_draws=np.asarray(inverse_cdf(jnp.asarray(cold),random.uniform(random.PRNGKey(1862),(100000,))))
    order=np.argsort(-softmax(logits/.8));cumulative=np.cumsum(softmax(logits/.8)[order]);crossing=order[np.searchsorted(cumulative,.9)]
    report['samplers']={'context':'Once upon a time, a little girl found a','context_tokens':np.asarray(prefix).tolist(),
                        'base_entropy_nats':float(-np.sum(softmax(logits)*np.log(softmax(logits)))),'tests':tests,
                        'cold_temperature':.0001,'cold_draws':100000,'all_cold_draws_are_argmax':bool(np.all(cold_draws==logits.argmax())),
                        'top_p_crossing_token':int(crossing),'crossing_token_label':token_label(sp,crossing),
                        'crossing_token_is_kept':bool(exact_distribution(logits,.8,'top-p',.9)[crossing]>0)}
    stories=[]
    for i,(label,temp,mode,cutoff) in enumerate([('temperature',.8,'temperature',0.),('top-k 20',.8,'top-k',20.),('top-p 0.9',.8,'top-p',.9),('min-p 0.1',.7,'min-p',.1)]):
        ids,_,_=generate_sequences(p,jnp.array([2]),random.PRNGKey(1870+i),steps=160,temperature=temp,mode=mode,cutoff=cutoff)
        values=np.asarray(ids[0]).tolist();stop=values.index(2) if 2 in values else len(values)
        stories.append({'sampler':label,'temperature':temp,'seed':1870+i,'max_generated_tokens':160,'ended_with_eos':stop<len(values),
                        'tokens':values,'text':visible(sp.decode(values[:stop]))})
    report['generated_stories']=stories
    calibration=[]
    for i,temp in enumerate([1.,.7]):
        ids,surprise,entropy=generate_sequences(p,jnp.array([2]),random.PRNGKey(1880+i),batch=128,steps=128,temperature=temp)
        ids,surprise,entropy=map(np.asarray,(ids,surprise,entropy))
        difference=(surprise.astype(float)-entropy.astype(float)).mean(axis=1)
        se_difference=float(difference.std(ddof=1)/np.sqrt(128));z=float(difference.mean()/se_difference)
        row={'temperature':temp,'sequences':128,'tokens_per_sequence':128,'seed':1880+i,
             'mean_surprise':float(surprise.astype(float).mean()),'mean_predicted_entropy':float(entropy.astype(float).mean()),
             'mean_sequence_gap':float(difference.mean()),'standard_error_over_independent_sequences':se_difference,'z_score':z,
             'scoring_temperature':1.,'eos_excluded':False,'context_after_64':'sliding last 64 tokens with position vectors rebased'}
        calibration.append(row);np.savez_compressed(ART/f'self-consistency-T{temp}.npz',tokens=ids,surprise=surprise,entropy=entropy)
        print('self-consistency',row,flush=True)
    report['self_consistency']=calibration
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    positions=np.arange(2);gaps=[r['mean_sequence_gap'] for r in calibration];errors=[2*r['standard_error_over_independent_sequences'] for r in calibration]
    ax.bar(positions,gaps,color=['#168aad','#e18b34']);ax.errorbar(positions,gaps,yerr=errors,fmt='none',color='black',capsize=6)
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=positions,xticklabels=[f'T={r["temperature"]}, z={r["z_score"]:.2f}' for r in calibration],ylabel='mean surprise - mean T=1 predicted entropy',title='128 independent 128-token sequences per temperature; error bars ±2 SE')
    savefig(fig,'self-consistency.png')
    # Optional extension: all eight heads, not a selected pair of attractive maps.
    valid_text=(DATA/'TinyStoriesV2-GPT4-valid.txt').read_text().split('<|endoftext|>')
    stories_list=[s.strip() for s in valid_text if s.strip()];opening_ids=[2]+sp.encode(stories_list[24867],out_type=int)[:31]
    _,attention=transformer(p,jnp.array(opening_ids)[None],return_attention=True);attention=np.asarray(attention)[:,0]
    np.save(ART/'attention-heads.npy',attention)
    labels=[token_label(sp,v) for v in opening_ids];fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    for layer in range(2):
        for head in range(4):
            ax=axes[layer,head];ax.imshow(attention[layer,head],vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
            ax.set(title=f'Layer {layer+1}, head {head+1}',xticks=np.arange(len(labels)),yticks=np.arange(len(labels)),xticklabels=labels,yticklabels=labels)
            ax.tick_params(axis='x',rotation=90,labelsize=6);ax.tick_params(axis='y',labelsize=6)
    fig.suptitle('All eight causal attention heads on the first held-out story\nRows predict the next token; columns are positions read. Maps show where, not why.')
    savefig(fig,'attention-heads.png')
    report['attention']={'tokens':opening_ids,'max_future_attention':float(np.max(np.triu(attention,k=1))),
                         'max_row_sum_error':float(np.max(abs(attention.sum(axis=-1)-1)))}
    write_json(EVIDENCE/'language-analysis.json',report)
    assert trained['bits_per_character']<ngram['models'][3]['bits_per_character']
    assert abs(mlp_record['validation_loss_nats']-ngram['models'][1]['loss_nats'])<.01
    assert mean[31:].mean()<ngram['models'][2]['loss_nats']
    assert all(t['p_value']>.001 for t in tests) and report['samplers']['all_cold_draws_are_argmax']
    assert -4<calibration[0]['z_score']<4 and abs(calibration[1]['z_score'])>40

if __name__=='__main__':main()
