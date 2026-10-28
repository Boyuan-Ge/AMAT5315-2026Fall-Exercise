"""Multistart robust Chinchilla refit, 400 bootstrap refits, analytic budget split."""
import itertools
import time
import csv
from scipy.optimize import minimize
from common import *

DELTA=.001
PRINTED=np.array([1.69,406.4,410.7,.34,.28])

def unpack(theta):
    e,a,b,alpha,beta=theta
    return np.array([np.exp(e),np.exp(a+alpha*np.log(1e9)),np.exp(b+beta*np.log(1e10)),alpha,beta])

def objective(theta,n,d,loss):
    e,a,b,alpha,beta=theta
    ln=np.log(n/1e9);ld=np.log(d/1e10)
    terms=np.stack([np.full(len(n),e),a-alpha*ln,b-beta*ld],axis=1)
    prediction=logsumexp(terms,axis=1);weights=np.exp(terms-prediction[:,None])
    residual=prediction-np.log(loss);absolute=np.abs(residual)
    huber=np.where(absolute<=DELTA,.5*residual**2,DELTA*(absolute-.5*DELTA))
    influence=np.clip(residual,-DELTA,DELTA)/DELTA**2
    jacobian=np.column_stack([weights,-weights[:,1]*ln,-weights[:,2]*ld])
    return float(np.mean(huber)/DELTA**2),np.mean(influence[:,None]*jacobian,axis=0)

def fit(start,n,d,loss):
    return minimize(objective,start,args=(n,d,loss),jac=True,method='L-BFGS-B',
                    bounds=[(-5,np.log(5)),(-10,5),(-10,5),(.05,1.5),(.05,1.5)],
                    options={'maxiter':3000,'ftol':1e-13,'gtol':1e-8,'maxls':50})

def predict(parameters,n,d):
    e,a,b,alpha,beta=parameters
    return e+a/np.asarray(n)**alpha+b/np.asarray(d)**beta

def optimum(parameters,c):
    e,a,b,alpha,beta=parameters
    logg=(np.log(alpha*a)-np.log(beta*b))/(alpha+beta)
    logn=logg+beta/(alpha+beta)*np.log(np.asarray(c)/6)
    logd=np.log(np.asarray(c)/6)-logn
    return np.exp(logn),np.exp(logd)

def main():
    start=time.perf_counter()
    with (DATA/'chinchilla-runs.csv').open() as f:table=list(csv.DictReader(f))
    n=np.array([float(row['Model Size']) for row in table])
    c=np.array([float(row['Training FLOP']) for row in table])
    loss=np.array([float(row['loss']) for row in table]);d=c/(6*n)
    keep=np.argsort(loss)[:-5];discard=np.argsort(loss)[-5:]
    n,c,d,loss=n[keep],c[keep],d[keep],loss[keep]
    starts=list(itertools.product([1.5,1.8],[.1,.5,1.],[.1,.5,1.],[.2,.35,.5],[.2,.35,.5]))
    best=None;grid=[]
    for index,(e,a,b,alpha,beta) in enumerate(starts):
        initial=np.array([np.log(e),np.log(a),np.log(b),alpha,beta])
        result=fit(initial,n,d,loss)
        grid.append({'start':initial.tolist(),'objective':float(result.fun),'success':bool(result.success),'iterations':int(result.nit)})
        if best is None or result.fun<best.fun:best=result
        if (index+1)%30==0:print(f'Scaling grid: {index+1}/{len(starts)}, best scaled Huber={best.fun:.8f}',flush=True)
    parameters=unpack(best.x);print('Best E,A,B,alpha,beta:',parameters,flush=True)
    rng=np.random.default_rng(1860);samples=[];statuses=[]
    for i in range(400):
        indices=rng.integers(0,len(n),len(n));result=fit(best.x,n[indices],d[indices],loss[indices])
        samples.append(unpack(result.x));statuses.append(bool(result.success))
        if (i+1)%50==0:print(f'Bootstrap refits: {i+1}/400',flush=True)
    samples=np.stack(samples);standard_errors=samples.std(axis=0,ddof=1)
    budgets=np.logspace(18,25,141);optimal_n,optimal_d=optimum(parameters,budgets)
    printed_n,printed_d=optimum(PRINTED,budgets)
    at_n,at_d=optimum(parameters,5.76e23)
    ratios=np.array([optimum(p,5.76e23)[1]/optimum(p,5.76e23)[0] for p in samples])
    residual=np.log(predict(parameters,n,d))-np.log(loss)
    old_residual=np.log(predict(PRINTED,n,d))-np.log(loss)
    report={'input_rows':len(table),'fitted_rows':len(n),'discarded_indices_highest_loss':discard.tolist(),
            'law':'E + A/N**alpha + B/D**beta; D=C/(6*N)',
            'fit':'L-BFGS-B, Huber(log predicted L - log measured L), delta=0.001; objective scaled by delta^-2 only for conditioning',
            'parameterization':'log E, log(A/1e9**alpha), log(B/1e10**beta), alpha, beta',
            'parameter_order':['E','A','B','alpha','beta'],'parameters':parameters.tolist(),'bootstrap_standard_errors':standard_errors.tolist(),
            'grid_starts':len(starts),'grid_success_count':sum(r['success'] for r in grid),'best_scaled_huber':float(best.fun),
            'bootstrap_resamples':400,'bootstrap_seed':1860,'bootstrap_success_count':sum(statuses),
            'mean_log_residual_refit':float(residual.mean()),'mean_log_residual_printed':float(old_residual.mean()),
            'printed_parameters':PRINTED.tolist(),'compute_exponents':{'model':float(parameters[4]/(parameters[3]+parameters[4])),
                                                                    'data':float(parameters[3]/(parameters[3]+parameters[4]))},
            'chinchilla_budget':5.76e23,'optimal_model_at_budget':float(at_n),'optimal_tokens_at_budget':float(at_d),
            'tokens_per_parameter_at_budget':float(at_d/at_n),'ratio_bootstrap_95_interval':np.quantile(ratios,[.025,.975]).tolist(),
            'printed_tokens_per_parameter_at_budget':float(optimum(PRINTED,5.76e23)[1]/optimum(PRINTED,5.76e23)[0]),
            'budgets':budgets.tolist(),'optimal_models':optimal_n.tolist(),'optimal_tokens':optimal_d.tolist(),
            'seconds':time.perf_counter()-start,'environment':environment()}
    np.savez_compressed(ART/'scaling-fit.npz',n=n,c=c,d=d,loss=loss,theta=best.x,parameters=parameters,bootstrap=samples,
                        bootstrap_success=statuses,residual=residual,printed_residual=old_residual)
    write_json(ART/'scaling-grid.json',grid);write_json(EVIDENCE/'scaling-metrics.json',report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');axes[0].scatter(n,loss,s=10,color='0.7',label='240 retained runs')
    xx=np.logspace(np.log10(n.min()),np.log10(n.max()),200)
    for budget,color in zip([1e19,1e20,1e21],['#168aad','#e18630','#a563ad']):
        near=abs(c/budget-1)<=.15;axes[0].scatter(n[near],loss[near],s=18,color=color)
        axes[0].plot(xx,predict(parameters,xx,budget/(6*xx)),color=color,label=f'C={budget:.0e} FLOP')
    axes[0].set(xscale='log',xlabel='model parameters N',ylabel='final loss (nats / token)',title='Fixed-compute fitted curves have an optimum');axes[0].legend(fontsize=8)
    image=axes[1].scatter(predict(parameters,n,d),loss,c=np.log10(c),s=18,cmap='viridis')
    bounds=[min(loss.min(),predict(parameters,n,d).min()),max(loss.max(),predict(parameters,n,d).max())]
    axes[1].plot(bounds,bounds,'k--',lw=1);axes[1].set(xlabel='fitted loss',ylabel='measured loss',title='Measured versus fitted: 240 runs')
    fig.colorbar(image,ax=axes[1],label='log10 training FLOP');savefig(fig,'scaling-fit.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for values,color,label in [(residual,'#168aad','refit'),(old_residual,'#e18630','printed parameters')]:
        axes[0].scatter(n,values,s=12,alpha=.75,color=color,label=f'{label}: mean {values.mean():+.5f}')
        axes[1].hist(values,bins=24,histtype='step',lw=1.5,color=color,label=label)
    axes[0].set(xscale='log',xlabel='model parameters N',ylabel='log predicted L - log measured L',title='Printed parameters systematically overpredict')
    axes[0].axhline(0,color='black',ls=':');axes[1].set(xlabel='log-loss residual',ylabel='number of runs',title='Residual distribution')
    for ax in axes:ax.legend(fontsize=8)
    savefig(fig,'scaling-residuals.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,fitted,printed,star,title in zip(axes,[optimal_n,optimal_d],[printed_n,printed_d],[70e9,1.4e12],['Compute-optimal parameters N*','Compute-optimal training tokens D*']):
        ax.loglog(budgets,fitted,color='#168aad',label='refit');ax.loglog(budgets,printed,'--',color='#e18630',label='printed law')
        ax.scatter([5.76e23],[star],s=140,marker='*',color='#c14444',label='Chinchilla (rounded published values)',zorder=4)
        ax.set(xlabel='training compute C (FLOP)',ylabel=title,title=title);ax.legend(fontsize=8);ax.grid(alpha=.2,which='both')
    savefig(fig,'compute-optimal.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ['environment','budgets','optimal_models','optimal_tokens']},indent=2),flush=True)
    e,a,b,alpha,beta=parameters
    assert .32<=alpha<=.38 and .32<=beta<=.42 and 1.75<=e<=1.90
    assert abs(residual.mean())<.005 and old_residual.mean()>.01
    assert 10<at_d/at_n<30

if __name__=='__main__':main()
