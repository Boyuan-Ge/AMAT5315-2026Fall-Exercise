"""Train with the prescribed actual batch sizes; evaluate the untouched validation."""
import argparse
import time
from language_models import *

SETTINGS={
 'mlp':{'seed':1838,'embedding_width':64,'hidden_width':512,'activation':'tanh','updates':4000,'batch_examples':16384,'peak_learning_rate':.006,'warmup_steps':200},
 'transformer':{'seed':1840,'layers':2,'width':96,'heads':4,'context':64,'mlp_multiplier':4,'activation':'GELU','updates':6000,'batch_sequences':32,'peak_learning_rate':.003,'warmup_steps':300}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=SETTINGS);args=parser.parse_args();kind=args.kind
    settings={**SETTINGS[kind],'optimizer':'AdamW','beta1':.9,'beta2':.999,'epsilon':1e-8,'weight_decay':.01,
              'decay_mask':'matrices only, not biases or layer-norm scales','gradient_clip_global_norm':1.,
              'schedule':'linear warmup, cosine decay to 0.1 times peak','precision':'float32'}
    key,init_key=random.split(random.PRNGKey(settings['seed']));params=initialize(kind,init_key)
    if kind=='mlp':save_model('mlp-initial',params)
    first=jax.tree.map(jnp.zeros_like,params);second=jax.tree.map(jnp.zeros_like,params)
    tokens=jnp.asarray(np.load(ART/'train-tokens.npy'));losses=[];step=jnp.int32(0);start=time.perf_counter()
    for _ in range(settings['updates']//50):
        params,first,second,key,step,values=train_chunk(params,first,second,key,step,tokens,kind)
        losses.extend(np.asarray(values).tolist())
        if int(step)%500==0:print(f'{kind}: {int(step)}/{settings["updates"]}; mean last 200 loss={np.mean(losses[-200:]):.5f}',flush=True)
    seconds=time.perf_counter()-start;save_model(kind,params);np.save(ART/f'{kind}-training-loss.npy',np.array(losses))
    validation=np.load(ART/'validation-tokens.npy');scores=evaluate(kind,params,validation)
    np.save(ART/f'{kind}-validation-loss.npy',scores)
    chars=json.loads((EVIDENCE/'data-provenance.json').read_text())['validation']['characters']
    report={'kind':kind,'settings':settings,'parameter_count':sum(p.size for p in jax.tree.leaves(params)),
            'training_seconds':seconds,'validation_tokens':len(validation),'validation_characters':chars,
            'validation_loss_nats':float(scores.mean()),'perplexity':float(np.exp(scores.mean())),
            'bits_per_character':float(scores.sum()/(np.log(2)*chars)),
            'loss_by_50_steps':np.array(losses).reshape(-1,50).mean(axis=1).tolist(),'environment':environment()}
    if kind=='mlp':report['batch_implementation']='draw all 16384 examples; exactly aggregate repeated previous-token inputs before computing equivalent gradients'
    write_json(EVIDENCE/f'{kind}-training.json',report);write_json(ART/f'{kind}-settings.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
