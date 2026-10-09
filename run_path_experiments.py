"""Second-stage pilot: complete BIC single/pair paths with verified witnesses.

Run with BLAS threads set to one. Timings include Arb verification, but exclude
the separate replay check and writing the artifacts. No old pilot is overwritten.
"""
from pathlib import Path
import copy
import json
import platform
import time
import numpy as np
import scipy
import flint
from scipy.linalg import cholesky, solve_triangular
from certified_path import (BlockGaussian,ArbGaussian,selection_path,exhaustive_path,
                            replay_path,verify_decision,family_search)


def generate_path(seed,scenario,n=240,p=10):
    rng=np.random.default_rng(seed)
    zs=[np.eye(4)[rng.integers(4,size=n)],np.eye(5)[rng.integers(5,size=n)]]
    x=rng.normal(size=(n,p))
    beta=np.zeros(p)
    sd=[.7,.8]
    if scenario=='signals':
        beta[[0,2,4]]=[.8,.5,.3]
    elif scenario=='suppressor':
        x[:,1]=.98*x[:,0]+np.sqrt(1-.98**2)*rng.normal(size=n)
        beta[[0,1,2]]=[4.,-4.,.55]
    elif scenario=='boundary':
        sd=[0.,.2];beta[[0,2]]=[.6,.3]
    elif scenario=='near_tie':
        x[:,1]=.9995*x[:,0]+np.sqrt(1-.9995**2)*rng.normal(size=n)
        beta[[0,2]]=[.8,.4]
    elif scenario=='exact_tie':
        x[:,1]=x[:,0];beta[0]=.9
    elif scenario!='null':
        raise ValueError(scenario)
    x=(x-x.mean(axis=0))/x.std(axis=0)
    y=x@beta+zs[0]@rng.normal(scale=sd[0],size=4)+zs[1]@rng.normal(scale=sd[1],size=5)+rng.normal(size=n)
    return y,np.ones((n,1)),x,zs


def dense_profile(args,lam,support):
    y,base,x,zs=args
    R=np.eye(len(y))
    for ratio,z in zip(lam,zs):
        R+=ratio*(z@z.T)
    l=cholesky(R,lower=True)
    design=np.column_stack([base,x[:,support]]) if support else base
    wy=solve_triangular(l,y,lower=True)
    wx=solve_triangular(l,design,lower=True)
    residual=wy-wx@np.linalg.lstsq(wx,wy,rcond=None)[0]
    return 2*np.log(np.diag(l)).sum()+len(y)*np.log((residual@residual)/len(y))


def numeric_checks():
    args=generate_path(611,'suppressor',n=72,p=6)
    model=BlockGaussian(*args);vg=ArbGaussian(model)
    dense_error=gradient_error=0.;count=0
    rng=np.random.default_rng(772)
    points=[np.zeros(2),np.array([0.,4.]),np.array([4.,0.]),np.full(2,4.),*rng.uniform(.1,3.9,(6,2))]
    supports=[(),(0,),(0,1),(0,1,2),(1,3,4)]
    for lam in points:
        f,h,grad,ggrad=model.evaluate(lam,supports)
        for j,s in enumerate(supports):
            av=float(vg.profile(lam,s)[0])
            dense_error=max(dense_error,abs(f[j]-dense_profile(args,lam,s)),abs(av-dense_profile(args,lam,s)))
            count+=1
        if np.all(lam>0) and np.all(lam<4):
            for k in range(2):
                d=np.eye(2)[k]*1e-5
                fd=(model.evaluate(lam+d,supports)[0]-model.evaluate(lam-d,supports)[0])/2e-5
                gradient_error=max(gradient_error,float(np.max(np.abs(fd-grad[:,k]-ggrad[k]))))
    # Precision failure must be explicit. At high precision the same nearly
    # collinear design is still admissible as exact binary data.
    y,b,x,z=args
    x=x.copy();x[:,1]=x[:,0]+1e-9*rng.normal(size=len(y))
    ill=BlockGaussian(y,b,x,z)
    precision={}
    for bits in (64,192):
        try:
            a=ArbGaussian(ill,bits).profile([.2,.4],(0,1))[0]
            precision[str(bits)]={'evaluated':True,'objective':str(a),'radius':str(a.rad())}
        except (ValueError,ZeroDivisionError,RuntimeError) as exc:
            precision[str(bits)]={'evaluated':False,'reason':str(exc)}
    assert dense_error<1e-7 and gradient_error<1e-4
    return {'objective_comparisons':count,'max_dense_objective_error':dense_error,
            'max_gradient_error':gradient_error,'ill_conditioned_point_evaluation':precision}


def rejection_checks(args,result):
    model=BlockGaussian(*args)
    # Choose a stage with at least one split to exercise coverage checks.
    witness=next(s['witness'] for s in result['stages'] if len(s['witness']['nodes'])>1)
    outcomes={}
    a=copy.deepcopy(witness);a['data_sha256']='0'*64
    outcomes['wrong_data_hash']=verify_decision(model,a)
    a=copy.deepcopy(witness);a['nodes'][0]['children']=a['nodes'][0]['children'][:1]
    outcomes['missing_half_of_box']=verify_decision(model,a)
    a=copy.deepcopy(witness);a['winner_lambda']=[10.,10.]
    outcomes['infeasible_incumbent']=verify_decision(model,a)
    a=copy.deepcopy(witness);a['winner_index']=(a['winner_index']+1)%len(a['family'])
    outcomes['wrong_winner']=verify_decision(model,a)
    a=copy.deepcopy(result)
    # Remove a rival from a reported complete family (metadata aligned).
    w=a['stages'][0]['witness'];idx=len(w['family'])-1
    if idx==w['winner_index']:idx-=1
    w['family'].pop(idx);w['added_sizes'].pop(idx)
    if w['winner_index']>idx:w['winner_index']-=1
    outcomes['incomplete_candidate_family']=replay_path(*args,a)
    assert all(not r['valid'] for r in outcomes.values())
    return outcomes


def main():
    dest=Path('results/path');dest.mkdir(exist_ok=True,parents=True)
    checks=numeric_checks()
    rows=[];all_replay=True;first=None
    for k,scenario in enumerate(['signals','suppressor','boundary','null','near_tie']):
        for rep in range(3):
            seed=914260+k*100+rep
            args=generate_path(seed,scenario)
            label=f'{scenario}_{seed}'
            np.savez_compressed(dest/f'{label}_data.npz',y=args[0],base=args[1],x=args[2],z0=args[3][0],z1=args[3][1])
            if seed%2:
                ref=exhaustive_path(*args);res=selection_path(*args,max_splits=1500,time_budget=30.)
            else:
                res=selection_path(*args,max_splits=1500,time_budget=30.);ref=exhaustive_path(*args)
            replay=replay_path(*args,res) if res['status']=='complete' else {'valid':False,'reason':'unresolved path'}
            all_replay &= replay['valid']
            raw={'scenario':scenario,'seed':seed,'verified_path':res,'exhaustive_multistart':ref,'replay':replay}
            (dest/f'{label}.json').write_text(json.dumps(raw,indent=2))
            row={'scenario':scenario,'seed':seed,'status':res['status'],
                 'selected':res['selected'],'additions':res['additions'],
                 'same_full_path':[(s['phase'],s['winner']) for s in res['stages']]==[(s['phase'],s['winner']) for s in ref['stages']],
                 'verified_replay':replay['valid'],'stages':len(res['stages']),
                 'pair_additions':sum(len(s)==2 for s in res['additions']),
                 'splits':sum(s['splits'] for s in res['stages']),
                 'model_fits':res['model_fits'],'reference_model_fits':ref['model_fits'],
                 'seconds':res['seconds'],'reference_seconds':ref['seconds'],
                 'reference_over_verified_time':ref['seconds']/res['seconds'],
                 'verification_seconds':sum(s['verification']['seconds'] for s in res['stages'] if s['verification'])}
            rows.append(row);print(json.dumps(row),flush=True)
            if first is None and res['status']=='complete':first=(args,res)
    checks['certificate_rejections']=rejection_checks(*first)
    tie_args=generate_path(918991,'exact_tie',n=160,p=6)
    tie=selection_path(*tie_args,max_splits=120,time_budget=15.)
    assert tie['status']=='unresolved_budget' and not tie['additions']
    (dest/'exact_tie.json').write_text(json.dumps(tie,indent=2))
    # Deliberate zero split budget; this is not a completed selection result.
    budget_args=generate_path(914560,'null')
    budget=selection_path(*budget_args,max_splits=0,time_budget=15.)
    assert budget['status']=='unresolved_budget'
    (dest/'zero_budget.json').write_text(json.dumps(budget,indent=2))
    checks['exact_tie_status']=tie['status'];checks['zero_budget_status']=budget['status']
    summary={'cases':len(rows),'complete':sum(r['status']=='complete' for r in rows),
             'same_full_paths':sum(r['same_full_path'] for r in rows),'all_replayed':all_replay,
             'total_verified_stages':sum(r['stages'] for r in rows if r['verified_replay']),
             'total_pair_additions':sum(r['pair_additions'] for r in rows),
             'model_fits':sum(r['model_fits'] for r in rows),
             'reference_model_fits':sum(r['reference_model_fits'] for r in rows),
             'median_reference_over_verified_time':float(np.median([r['reference_over_verified_time'] for r in rows])),
             'minimum_reference_over_verified_time':min(r['reference_over_verified_time'] for r in rows),
             'maximum_reference_over_verified_time':max(r['reference_over_verified_time'] for r in rows),
             'precision_bits':128,'n':240,'p':10,'random_effect_columns':9,
             'box':[[0.,4.],[0.,4.]],'replicates_per_scenario':3,
             'qualification':'Exploratory BIC path pilot. Not ARSS-JPS; comparator uses local multistart ML. All certificate costs included.',
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'python_flint':flint.__version__}}
    (dest/'rows.json').write_text(json.dumps(rows,indent=2))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'verification_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
