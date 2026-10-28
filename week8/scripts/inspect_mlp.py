"""Optional extension: all 1,024 signed tanh unit sides, not a claim of clean concepts."""
from language_models import *
from language_analysis import token_label
from prepare_data import tokenizer

def activation(p):return np.asarray(jnp.tanh(linear(p['hidden'],p['embedding'])),dtype=float)

def main():
    sp=tokenizer();p=load_model('mlp');initial=load_model('mlp-initial')
    a=activation(p);untrained=activation(initial)
    def shares(values):
        side=np.stack([np.maximum(values,0)**2,np.maximum(-values,0)**2])
        return np.sort(side,axis=1)[:,-5:,:].sum(axis=1)/np.maximum(side.sum(axis=1),1e-30)
    trained_shares=shares(a);initial_shares=shares(untrained);rows=[]
    readout=np.asarray(p['readout']['w'])
    for unit in range(512):
        for sign,si in [(1,0),(-1,1)]:
            previous=np.argsort(-sign*a[:,unit])[:5];following=np.argsort(-sign*readout[unit])[:5]
            rows.append({'unit':unit,'side':'+' if sign==1 else '-','top5_activation_energy_share':float(trained_shares[si,unit]),
                         'previous_token_ids':previous.tolist(),'next_token_ids':following.tolist(),
                         'previous_labels':[token_label(sp,v) for v in previous],'next_labels':[token_label(sp,v) for v in following]})
    selected=np.argsort(-trained_shares[0])[:4];by_key={(r['unit'],r['side']):r for r in rows};table=[]
    for unit in selected:
        for side in ['+','-']:
            r=by_key[(int(unit),side)]
            table.append([f'{unit} {side}',f'{r["top5_activation_energy_share"]:.3f}','  '.join(r['previous_labels']),'→','  '.join(r['next_labels'])])
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');ax.axis('off')
    render=ax.table(cellText=table,colLabels=['Unit side','Top-5 share','Previous tokens that excite this side','','Next tokens promoted'],loc='center',cellLoc='left',colWidths=[.08,.09,.37,.03,.43])
    render.auto_set_font_size(False);render.set_fontsize(10);render.scale(1,1.8)
    fig.suptitle(f'Optional signed-unit audit: four most concentrated positive sides\nMedian top-5 share: trained {np.median(trained_shares):.3f}; initial {np.median(initial_shares):.3f}; shared features need not be readable memories')
    savefig(fig,'mlp-unit-audit.png')
    write_json(EVIDENCE/'mlp-unit-audit.json',{'definition':'sum of squared activations of top 5 input tokens / total side squared activation; unweighted over vocabulary',
               'median_share_trained':float(np.median(trained_shares)),'median_share_initial':float(np.median(initial_shares)),
               'all_signed_unit_sides':rows,'selected_units':selected.tolist()})

if __name__=='__main__':main()
