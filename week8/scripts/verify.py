"""Audit every numerical requirement; optionally recompute from ignored raw arrays."""
import argparse
import re
import subprocess
from PIL import Image
from common import *
from memory import capacity
from scaling import optimum,predict

FIGURES=['embedding-relations.png','ngram-ladder.png','memory-capacity.png','picture-recall.png',
         'language-model-comparison.png','context-use.png','sampler-tests.png','self-consistency.png',
         'attention-heads.png','mlp-unit-audit.png','scaling-fit.png','scaling-residuals.png','compute-optimal.png']

def read(name):return json.loads((EVIDENCE/(name+'.json')).read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--with-artifacts',action='store_true');args=parser.parse_args()
    checks={}
    def check(name,value):
        assert bool(value),name
        checks[name]=True
    data=read('data-provenance');embedding=read('embedding-metrics');ngram=read('ngram-metrics')
    memory=read('memory-metrics');picture=read('picture-metrics');mlp=read('mlp-training');transformer=read('transformer-training')
    analysis=read('language-analysis');scaling=read('scaling-metrics');contracts=read('contract-checks')
    check('story-level split: 24867 training, 2763 validation',data['training_stories']==24867 and data['validation_stories']==2763)
    check('512 BPE tokens, about 2.05 characters/token',data['vocabulary']==512 and abs(data['validation']['characters_per_token']-2.05)<.01)
    check('exactly three whitespace-normalization decode exceptions',data['validation_decode_mismatches']==3)
    for name,relation in embedding['relations'].items():
        check(name+': all nine single-token pairs',np.array(relation['token_ids']).shape==(9,2))
        check(name+': mean leave-one-out cosine >=0.3',relation['mean_cosine']>=.3)
        check(name+': seed-1730 shuffled control within +/-0.15',abs(relation['control_mean_cosine'])<=.15)
        check(name+': all twenty shuffle means within +/-0.09',max(abs(np.array(relation['twenty_shuffle_means'])))<=.09)
    check('bigram explicit length-1/2/3 enumeration sums to one',max(ngram['normalization']['absolute_errors'])<1e-12)
    check('uniform probability and perplexity are exact',abs(ngram['uniform']['loss_nats']-np.log(512))<1e-12 and ngram['uniform']['perplexity']==512)
    for n,nats,ppl,bpc in [(1,4.839,126.4,3.403),(2,3.038,20.9,2.136),(3,2.094,8.12,1.473)]:
        row=ngram['models'][n-1]
        check(f'{n}-gram: all three held-out metrics match required tolerance',abs(row['loss_nats']-nats)<.03 and abs(row['perplexity']/ppl-1)<.03 and abs(row['bits_per_character']-bpc)<.03)
        check(f'{n}-gram: metric conversions consistent',abs(np.exp(row['loss_nats'])-row['perplexity'])<1e-10 and abs(row['loss_nats']*ngram['validation_tokens']/np.log(2)/ngram['validation_characters']-row['bits_per_character'])<1e-10)
    check('smoothing chosen only within training tokens',ngram['smoothing_split_token']==int(.9*data['train']['tokens']) and ngram['smoothing_candidates']==[.0001,.001,.01,.1,1.])
    for row in ngram['models']:check(f'n={row["n"]}: selected k minimizes training-tuning loss',row['k']==ngram['smoothing_candidates'][int(np.argmin(row['tuning_losses']))])
    check('250000-token ancestral bigram sample passes pooled chi-square',ngram['bigram_sampling']['tokens']==250000 and ngram['bigram_sampling']['p_value']>.001 and ngram['bigram_sampling']['minimum_expected_bin']>=5)
    bpc=np.array([r['bits_per_character'] for r in ngram['models']])
    check('n=4 is the best count model',np.argmin(bpc)==3)
    check('n=5/6 losses and worsening at n=6',abs(bpc[4]-1.46)<.03 and abs(bpc[5]-1.79)<.03 and bpc[5]>bpc[2])
    for n,expected in [(3,.006),(4,.042),(5,.122),(6,.235)]:check(f'n={n}: unseen fraction within two percentage points',abs(ngram['models'][n-1]['unseen_fraction']-expected)<.02)
    ratios=np.array(memory['median_ratios']);sizes=np.array(memory['sizes']);capacities=np.array(memory['capacities'])
    check('all requested dimensions and 14 memory sizes',memory['dimensions']==[16,32,64] and np.array_equal(sizes,2**np.arange(3,17)))
    check('three independent capacity draws and every key queried',np.array(memory['ratios_by_dimension_draw_size_store']).shape==(3,3,14,5) and 'EVERY' in memory['ratio_aggregation'])
    check('all d=64 R slopes agree with 1/2',max(abs(np.array(memory['d64_ratio_loglog_slopes'])-.5))<.1)
    check('d=64 outer-product capacity between 50 and 85',50<capacities[2,0]<85)
    check('least-squares exact through M=d; capacity 100..170',np.max(ratios[2,sizes<=64,1])<1e-10 and 100<capacities[2,1]<170)
    slopes=np.array(memory['capacity_loglog_slopes'])
    check('outer, least-squares, ReLU capacity slopes within 0.1 of one',np.max(abs(slopes[:3]-1))<.1)
    check('ReLU squared slope near two; cubic above 2.5',abs(slopes[3]-2)<.15 and slopes[4]>2.5)
    check('MLP prescribed budget, batch and architecture',mlp['settings']['updates']==4000 and mlp['settings']['batch_examples']==16384 and mlp['settings']['embedding_width']==64 and mlp['settings']['hidden_width']==512 and mlp['settings']['peak_learning_rate']==.006)
    check('MLP held-out loss within 0.01 nats of bigram',abs(mlp['validation_loss_nats']-ngram['models'][1]['loss_nats'])<.01)
    recalled=picture['runs'];at1024=np.array(recalled[2]['median_ratios'])
    check('picture experiment uses all requested M, dimension and beta',picture['dimension']==64 and picture['beta']==8 and [r['memories'] for r in recalled]==[64,256,1024,4096])
    check('1024-memory outer/ReLU medians within 15% of theory',abs(at1024[0]/4-1)<.15 and abs(at1024[1]/np.sqrt(8)-1)<.15)
    check('ReLU squared loses the 4096-memory picture',recalled[3]['single_draw_ratios'][2]>1 and recalled[3]['median_ratios'][2]>1)
    check('softmax R below 0.1 in every M and all twenty draws',max(np.array(row['all_ratios'])[:,3].max() for row in recalled)<.1)
    settings=transformer['settings']
    check('transformer exact prescribed architecture and training budget',all(settings[k]==v for k,v in {'layers':2,'width':96,'heads':4,'context':64,'updates':6000,'batch_sequences':32,'peak_learning_rate':.003,'warmup_steps':300}.items()))
    check('transformer beats EVERY count model in bits/character',transformer['bits_per_character']<bpc.min())
    check('all models use identical held-out targets and character denominator',all(record['validation_tokens']==ngram['validation_tokens'] and record['validation_characters']==ngram['validation_characters'] for record in [mlp,transformer]))
    openings=analysis['openings'];histories=[tuple(row['initial_history']) for row in openings.values()]
    check('five 60-token openings: same seed and real EOS-ended history',len(openings)==5 and all(len(row['tokens'])==60 and row['seed']==1848 and row['eos_excluded'] for row in openings.values()) and len(set(histories))==1 and histories[0][-1]==2)
    curve=np.array(analysis['context_curve']['loss_by_position'])
    check('every context position from 32 onward beats trigram',curve.shape==(64,) and np.max(curve[31:])<ngram['models'][2]['loss_nats'])
    tests=analysis['samplers']['tests']
    check('all nine 100000-draw sampler chi-square tests pass',len(tests)==9 and all(t['draws']==100000 and t['p_value']>.001 and t['minimum_expected_bin']>=5 for t in tests))
    check('JAX samplers independently match NumPy distributions',max(t['max_probability_error_numpy_vs_jax'] for t in tests)<1e-6)
    check('top-p keeps its crossing token',analysis['samplers']['crossing_token_is_kept'])
    check('temperature 0.0001 always returns argmax in 100000 draws',analysis['samplers']['all_cold_draws_are_argmax'] and analysis['samplers']['cold_draws']==100000)
    check('all four specified story-generation rules exercised',len(analysis['generated_stories'])==4 and [r['temperature'] for r in analysis['generated_stories']]==[.8,.8,.8,.7])
    calibration=analysis['self_consistency']
    check('two self-consistency experiments have 128 independent 128-token sequences',all(r['sequences']==128 and r['tokens_per_sequence']==128 and r['scoring_temperature']==1 and not r['eos_excluded'] for r in calibration))
    check('T=1 self-consistency z within +/-4',abs(calibration[0]['z_score'])<4)
    check('T=0.7 mismatch exceeds 40 SE',abs(calibration[1]['z_score'])>40)
    check('all eight attention maps are causal and normalized',analysis['attention']['max_future_attention']==0 and analysis['attention']['max_row_sum_error']<1e-6)
    check('all 1024 signed MLP unit sides inspected',len(read('mlp-unit-audit')['all_signed_unit_sides'])==1024)
    check('scaling input 245 runs, highest five removed',scaling['input_rows']==245 and scaling['fitted_rows']==240 and len(scaling['discarded_indices_highest_loss'])==5)
    e,a,b,alpha,beta=scaling['parameters']
    check('refit exponents and floor in required ranges',.32<=alpha<=.38 and .32<=beta<=.42 and 1.75<=e<=1.90)
    check('162 grid starts and 400 bootstrap refits converge',scaling['grid_starts']==162 and scaling['grid_success_count']==162 and scaling['bootstrap_resamples']==400 and scaling['bootstrap_success_count']==400)
    check('every fitted parameter has a positive finite bootstrap SE',len(scaling['bootstrap_standard_errors'])==5 and np.all(np.isfinite(scaling['bootstrap_standard_errors'])) and min(scaling['bootstrap_standard_errors'])>0)
    check('refit mean log residual near zero; printed residual >0.01',abs(scaling['mean_log_residual_refit'])<.005 and scaling['mean_log_residual_printed']>.01)
    check('compute-optimal ratio 10..30; 95% interval contains 20',10<scaling['tokens_per_parameter_at_budget']<30 and scaling['ratio_bootstrap_95_interval'][0]<20<scaling['ratio_bootstrap_95_interval'][1])
    budgets=np.array(scaling['budgets']);n=np.array(scaling['optimal_models']);d=np.array(scaling['optimal_tokens'])
    check('compute grid spans 1e18 through 1e25; preserves compute',budgets[0]==1e18 and budgets[-1]==1e25 and np.max(abs(6*n*d/budgets-1))<1e-12)
    check('independent mathematical and implementation contracts passed',contracts['passed']==40 and all(contracts['checks'].values()))
    readme=(ROOT/'README.md').read_text();embeds=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',readme)
    for name,row in analysis['openings'].items():check(name+': full decoded 60-token opening appears verbatim in README',row['text'] in readme)
    for row in analysis['generated_stories']:check(row['sampler']+': full displayed story appears verbatim in README',row['text'] in readme)
    for name in FIGURES:
        check(name+': embedded directly in README','evidence/'+name in embeds)
        with Image.open(EVIDENCE/name) as image:image.verify()
    check('every README image path exists',all((ROOT/path).is_file() for path in embeds))
    tracked=subprocess.check_output(['git','ls-files','week8'],cwd=ROOT.parent,text=True).splitlines()
    check('Git contains no corpus, downloaded weights, environment or raw runs',not any(path.startswith(('week8/data/','week8/artifacts/','week8/.venv/')) or path.endswith('.safetensors') for path in tracked))
    if args.with_artifacts:
        from language_models import load_model,evaluate
        from ngrams import Ngram,codes
        train=np.load(ART/'train-tokens.npy');valid=np.load(ART/'validation-tokens.npy')
        check('training/validation token fingerprints match provenance',sha256(ART/'train-tokens.npy')==data['train']['sha256_tokens'] and sha256(ART/'validation-tokens.npy')==data['validation']['sha256_tokens'])
        for file in data['files']:check(file['file']+': pinned input hash',sha256(ROOT/file['file'])==file['sha256'])
        for n in range(1,7):
            actual,unseen=Ngram.load(n).score_codes(codes(valid,n));row=ngram['models'][n-1]
            check(f'n={n}: held-out loss and unseen rate recomputed from count table',abs(actual.mean()-row['loss_nats'])<1e-12 and abs(unseen.mean()-row['unseen_fraction'])<1e-12)
        for kind,record in [('mlp',mlp),('transformer',transformer)]:
            losses=np.load(ART/f'{kind}-training-loss.npy')
            check(kind+': complete prescribed raw loss history',losses.shape==(record['settings']['updates'],) and np.isfinite(losses).all())
            actual=evaluate(kind,load_model(kind),valid).astype(float)
            check(kind+': all validation predictions recomputed from checkpoint',abs(actual.mean()-record['validation_loss_nats'])<1e-6)
            check(kind+': bits/character independently recomputed',abs(actual.sum()/np.log(2)/data['validation']['characters']-record['bits_per_character'])<1e-6)
        for di,dimension in enumerate(memory['dimensions']):
            for mi,size in enumerate(memory['sizes']):
                raw=[]
                for rep in range(3):
                    values=np.load(ART/f'memory/d{dimension}-draw{rep}-M{size}.npz')['ratios']
                    check(f'd={dimension}, M={size}, draw={rep}: EVERY stored-key ratio retained',values.shape==(5,size) and np.isfinite(values).all())
                    raw.append(np.sqrt(np.mean(values**2,axis=1)))
                check(f'd={dimension}, M={size}: median-of-three RMS recomputed',np.allclose(np.median(raw,axis=0),ratios[di,mi],rtol=1e-12,atol=1e-14))
        for row in calibration:
            f=np.load(ART/f'self-consistency-T{row["temperature"]}.npz')
            gap=(f['surprise'].astype(float)-f['entropy'].astype(float)).mean(axis=1)
            z=gap.mean()/(gap.std(ddof=1)/np.sqrt(128))
            check(f'T={row["temperature"]}: z recomputed using independent sequence SE',abs(z-row['z_score'])<1e-10)
        raw=np.load(ART/'scaling-fit.npz');bootstrap=raw['bootstrap']
        check('all 400 bootstrap parameter vectors retained',bootstrap.shape==(400,5) and np.isfinite(bootstrap).all())
        check('all five bootstrap SEs recomputed',np.allclose(bootstrap.std(axis=0,ddof=1),scaling['bootstrap_standard_errors'],rtol=1e-12))
        ratio=np.array([optimum(p,5.76e23)[1]/optimum(p,5.76e23)[0] for p in bootstrap])
        check('bootstrap 95% compute-ratio interval recomputed',np.allclose(np.quantile(ratio,[.025,.975]),scaling['ratio_bootstrap_95_interval'],rtol=1e-12))
    report={'passed':len(checks),'checks':checks,'raw_artifacts_audited':args.with_artifacts,'figures':FIGURES,
            'source_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in sorted((ROOT/'scripts').glob('*')) if path.is_file()},'environment':environment()}
    if args.with_artifacts:write_json(EVIDENCE/'acceptance.json',report)
    print(f'{len(checks)} audit checks passed; all {len(FIGURES)} figures are inline and decode successfully.',flush=True)

if __name__=='__main__':main()
