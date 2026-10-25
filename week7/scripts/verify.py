"""Audit measured evidence and, optionally, recompute metrics from local raw runs."""
import argparse
import re
import tomllib
from PIL import Image
from transport import *
from mixture_experiments import validate_report

FIGURES=['forward-stages.png','training-stages.png','training-loss.png','sampler-samples.png',
         'determinism-paths.png','velocity-fields.png','transport-cost.png','exact-coarse-controls.png',
         'photograph-stages.png','real-vs-generated.png','inpainting-results.png']
MOVIES={'forward-diffusion.gif':51,'reverse-ddpm.gif':51,'training-progress.gif':10}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--with-artifacts',action='store_true');args=parser.parse_args()
    checks={}
    def check(name,value):
        assert bool(value),name
        checks[name]=True
    mixture=json.loads((EVIDENCE/'mixture-metrics.json').read_text());validate_report(mixture)
    check('mixture numerical acceptance checks',True)
    check('DDPM independent noise same cluster about one quarter',abs(mixture['sampling_2000']['ddpm_same_cluster_fraction']-.25)<.04)
    for kind,seed in [('diffusion',2026),('flow',2029)]:
        trained=json.loads((EVIDENCE/f'{kind}-training.json').read_text())
        config=tomllib.loads((ROOT/f'{kind}.toml').read_text())
        check(f'{kind}: recorded settings match TOML',trained['settings']==config and trained['settings']['training']['seed']==seed)
        check(f'{kind}: actual configuration fingerprint',trained['config_sha256']==sha256(ROOT/f'{kind}.toml'))
        check(f'{kind}: 25000-update loss history',len(trained['loss_by_100_updates'])==250)
        check(f'{kind}: loss decreased',trained['last_1000_mean_loss']<trained['first_500_mean_loss'])
    expected_stages=[0,50,100,200,500,1000,2000,5000,10000,25000]
    check('all ten requested training checkpoints recorded',mixture['stage_updates']==expected_stages)
    for method in ['ddpm','ddim','flow']:
        expected=[1,2,5,10,20,50,100]+([] if method=='flow' else [1000])
        for exact in [False,True]:
            rows=[row for row in mixture['comparisons'] if row['method']==method and row['exact_predictor']==exact]
            check(f'{method}: complete {"exact" if exact else "learned"} step grid',[row['steps'] for row in rows]==expected)
    check('4000 samples and twenty independent null pairs',mixture['comparison_samples']==4000 and len(mixture['null_energy_distances'])==20)
    check('null 95 percent quantile recomputed',np.isclose(mixture['null_95_quantile'],np.quantile(mixture['null_energy_distances'],.95)))
    oracle=json.loads((EVIDENCE/'unet-reference.json').read_text())
    photo=json.loads((EVIDENCE/'photograph-metrics.json').read_text())
    check('pinned image checkpoint and unchanged weights',photo['weights_unchanged'] and photo['revision']=='267b167dc01f0e4e61923ea244e8b988f84deb80' and photo['weights_sha256']==oracle['weight_sha256'])
    check('JAX predictor matches diffusers v0.40.0 at three time indices',oracle['all_reference_checks_passed'] and oracle['diffusers_version']=='0.40.0' and len(oracle['comparisons'])==3 and all(r['max_absolute_error']<5e-4 and r['relative_rms_error']<1e-4 for r in oracle['comparisons']))
    gen=photo['generation'];inp=photo['inpainting']
    check('four 1000-step unclipped photographs',gen['images']==4 and gen['steps']==1000 and not gen['state_clipping'])
    check('all sixteen inpainting measurements',np.array(inp['hidden_mse_by_seed_then_image']).shape==(4,4) and inp['seeds']==[7201,7202,7203,7204])
    check('RePaint observed pixels preserved',inp['max_observed_change']<1e-5)
    check('variation only in hidden region',inp['max_variance_observed']==0 and inp['mean_variance_inside_hole']>0)
    check('RePaint 50 jumps and 99 calls',inp['reverse_jumps']==50 and inp['network_evaluations_per_completion']==99)
    contracts=json.loads((EVIDENCE/'contract-checks.json').read_text())
    check('independent numerical and instrumented sampler checks',contracts['passed']==17 and all(contracts['checks'].values()))
    readme=(ROOT/'README.md').read_text()
    embeds=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',readme)
    check('all eleven figures embedded, not just linked',all('evidence/'+name in embeds for name in FIGURES))
    check('all three movies embedded',all('evidence/'+name in embeds for name in MOVIES))
    for name in FIGURES:
        with Image.open(EVIDENCE/name) as image:image.verify()
    check('all PNG files decode successfully',True)
    for name,count in MOVIES.items():
        with Image.open(EVIDENCE/name) as image:check(f'{name}: all movie frames',image.n_frames==count)
    check('every local embedded image resolves',all((ROOT/url).is_file() for url in embeds if not url.startswith('http')))
    if args.with_artifacts:
        data=np.load(ART/'mixture/samplers.npz')
        for label in ['ddpm','ddpm2','ddim']:
            key='ddpm_second' if label=='ddpm2' else label
            actual=cluster_stats(data[label][-1]);reported=mixture['sampling_2000'][key]
            check(f'{label}: raw cluster measurements match report',all(np.allclose(actual[k],reported[k],atol=1e-10) for k in ['populations','centers','widths']))
        for row in mixture['comparisons']:
            name=f"{row['method']}-{row['steps']}-{'exact' if row['exact_predictor'] else 'learned'}.npy"
            points=np.load(ART/'mixture'/name)
            check(name+': 4000 finite raw samples',points.shape==(4000,2) and np.isfinite(points).all())
            measured=cluster_stats(points)
            for metric in ['max_population_error','max_width_error','max_center_error']:
                check(name+': '+metric,measured[metric]==row[metric])
        for kind in ['diffusion','flow']:
            losses=np.load(ART/kind/'loss.npy')
            check(f'{kind}: all raw update losses retained',losses.shape==(25000,) and np.isfinite(losses).all())
        raw=np.load(ART/'photographs/inpainting.npz');states=raw['states'];truth=raw['truth'];mask=raw['mask']
        check('raw RePaint states: four seeds, 51 levels, four RGB images',states.shape==(4,51,4,3,32,32) and np.isfinite(states).all())
        actual_mse=np.mean((states[:,-1,:,:,6:26,6:26]-truth[None,:,:,6:26,6:26])**2,axis=(2,3,4))
        check('all hidden MSEs independently recomputed',np.allclose(actual_mse,inp['hidden_mse_by_seed_then_image'],atol=1e-7))
        check('all observed raw pixels exactly identical to observations',np.max(abs((states[:,-1]-raw['observed'])*mask))==0)
        check('pixel variance independently recomputed',np.array_equal(np.var(states[:,-1],axis=0),raw['variance']))
        check('downloaded weights still match fixed hash',sha256(INPUT/'cifar-model/diffusion_pytorch_model.safetensors')==photo['weights_sha256'])
    warnings=[
        'DDIM right-mode widths are 0.27765 and 0.27879: within Part 1 tolerance 0.30 +/- 0.03, but below the illustrative Part 2 reference interval 0.29..0.32.',
        'Velocity RMS <0.1 holds on the grid over the >=99% probability region and on independent marginal test samples. The full unweighted [-4,4]^2 square (including rare extrapolation) has RMS 0.308, 0.250, 0.091; do not claim a uniform whole-square bound.'
    ]
    manifest={str(path.relative_to(ROOT)):sha256(path) for path in sorted((ROOT/'scripts').glob('*')) if path.is_file()}
    write_json(EVIDENCE/'acceptance.json',{'passed':len(checks),'checks':checks,'raw_artifacts_audited':args.with_artifacts,'figures':FIGURES,'movies':MOVIES,'caveats':warnings,'source_sha256':manifest,'environment':environment()})
    print(f'{len(checks)} audit checks passed; {len(FIGURES)} inline figures and {len(MOVIES)} inline movies verified.',flush=True)
    for warning in warnings:print('CAVEAT:',warning,flush=True)

if __name__=='__main__':main()
