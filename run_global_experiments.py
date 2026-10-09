"""Stage three: rigorous tail exclusion, 18 paths and targeted soundness checks.

The old 15 raw datasets are reused without regenerating or changing them. Three
prespecified large-variance seeds add fixed predictors aligned with group designs.
The finite-box local comparator uses [0,1024]^2 and four/five deterministic starts.
All global certificate construction costs are included in its recorded runtime;
separate replay, data writing and diagnostic reruns are excluded.
"""
from pathlib import Path
import copy
import itertools
import json
import platform
import numpy as np
import scipy
import flint
from flint import fmpq,fmpq_mat,arb
from certified_path import BlockGaussian,exhaustive_path,selection_path,replay_path
from run_path_experiments import generate_path
from certified_global_path import (TailArbGaussian,exact_column_basis,
    global_selection_path,replay_global_path,global_family_search,verify_global_decision)


def generate_high_variance(seed,n=240,p=10):
    rng=np.random.default_rng(seed)
    zs=[np.eye(4)[rng.integers(4,size=n)],np.eye(5)[rng.integers(5,size=n)]]
    x=rng.normal(size=(n,p))
    x[:,0]=zs[0]@np.array([-1.,-.3,.4,1.2])
    x[:,1]=zs[1]@np.array([-1.,-.5,0.,.5,1.])
    x=(x-x.mean(axis=0))/x.std(axis=0)
    beta=np.zeros(p);beta[[2,4]]=[.65,.4]
    y=x@beta+zs[0]@rng.normal(scale=4.,size=4)+zs[1]@rng.normal(scale=6.,size=5)+rng.normal(size=n)
    return y,np.ones((n,1)),x,zs


def load_data(path):
    d=np.load(path,allow_pickle=False)
    return d['y'],d['base'],d['x'],[d['z0'],d['z1']]


def exact_projection_floor(args,support):
    y,base,x,zs=args
    w=np.column_stack([base,*zs,x[:,support]])
    rational=lambda a:fmpq_mat([[fmpq(*float(v).as_integer_ratio()) for v in row] for row in a])
    # Independent basis route: row-reduce the transposed full design and use
    # its nonzero rows, rather than the production code's original pivot columns.
    reduced,rank=rational(w).transpose().rref()
    a=fmpq_mat([[reduced[i,j] for i in range(rank)] for j in range(len(y))])
    b=rational(y[:,None])
    by=a.transpose()*b
    return (b.transpose()*b-by.transpose()*(a.transpose()*a).solve(by))[0,0]


def floor_checks():
    # Integer-valued data make exact rational residuals an independent reference.
    rng=np.random.default_rng(916991);n=18
    zs=[np.eye(3)[np.arange(n)%3],np.eye(2)[np.arange(n)%2]]
    x=rng.integers(-4,5,(n,3)).astype(float);x[:,0]=zs[0]@np.array([-1.,0.,1.])
    y=rng.integers(-5,6,n).astype(float);base=np.ones((n,1))
    args=y,base,x,zs
    vg=TailArbGaussian(BlockGaussian(*args));rows=[]
    for support in [(),(0,),(1,),(0,1),(1,2),(0,1,2)]:
        floor=vg.residual_floor(support);exact=exact_projection_floor(args,support)
        enclosed=floor.contains(exact)
        assert enclosed and exact>0
        rows.append({'support':list(support),'exact_floor':str(exact),'arb_floor':str(floor),'contains_exact':enclosed})
    # Here a missing Z column would give an invalidly positive residual floor.
    y0=zs[0]@np.array([1.,2.,4.])+zs[1]@np.array([-1.,3.])
    bad=global_selection_path(y0,base,x,zs,max_splits=10)
    assert bad['status']=='unresolved_tail' and not bad['additions']
    # A structurally zero component is outside this sufficient tail construction.
    zero=global_selection_path(y,base,x,[zs[0],np.zeros_like(zs[1])],max_splits=10)
    assert zero['status']=='unresolved_tail' and not zero['additions']
    return {'exact_rational_floor_checks':rows,
            'projection_reference_method':'Exact rational row-space basis of the transposed full augmented design; direct normal-equation solve.',
            'zero_floor_status':bad['status'],
            'zero_floor_reason':bad['stages'][0]['reason'],'zero_component_status':zero['status'],
            'zero_component_reason':zero['stages'][0]['reason']}


def corruption_checks(args,result,old_box_result):
    model=BlockGaussian(*args);out={}
    w=copy.deepcopy(result['stages'][0]['witness']);w.pop('tail_method')
    out['missing_tail_certificate']=verify_global_decision(model,w)
    w=copy.deepcopy(result['stages'][0]['witness']);w['data_sha256']='0'*64
    out['wrong_data_hash']=verify_global_decision(model,w)
    w=copy.deepcopy(result['stages'][0]['witness']);w['hi'][0]*=.5
    out['altered_cutoff_without_repartition']=verify_global_decision(model,w)
    r=copy.deepcopy(result);w=r['stages'][0]['witness'];idx=len(w['family'])-1
    if idx==w['winner_index']:idx-=1
    w['family'].pop(idx);w['added_sizes'].pop(idx)
    if w['winner_index']>idx:w['winner_index']-=1
    out['missing_rival']=replay_global_path(*args,r)
    # This has a genuine verified complete partition of [0,4]^2. Relabelling it
    # as global must fail solely because the recomputed tails cannot be excluded.
    w=copy.deepcopy(old_box_result['stages'][0]['witness'])
    w.update(domain='nonnegative_orthant',tail_method='projection_floor_v1')
    out['finite_box_falsely_labelled_global']=verify_global_decision(model,w)
    assert all(not v['valid'] for v in out.values())
    assert 'tail bounds' in out['finite_box_falsely_labelled_global']['reason']
    return out


def decisions(result):
    return [(s['phase'],s['winner']) for s in result['stages']]


def main():
    dest=Path('results/global_path');dest.mkdir(parents=True,exist_ok=True)
    checks=floor_checks();rows=[];oldrows=json.loads(Path('results/path/rows.json').read_text())
    cases=[(r['scenario'],r['seed']) for r in oldrows]+[('high_variance',s) for s in (915260,915261,915262)]
    first_high=None
    for scenario,seed in cases:
        label=f'{scenario}_{seed}'
        args=generate_high_variance(seed) if scenario=='high_variance' else load_data(Path('results/path')/f'{label}_data.npz')
        np.savez_compressed(dest/f'{label}_data.npz',y=args[0],base=args[1],x=args[2],z0=args[3][0],z1=args[3][1])
        if seed%2:
            ref=exhaustive_path(*args,hi=[1024.,1024.]);res=global_selection_path(*args,time_budget=45.)
        else:
            res=global_selection_path(*args,time_budget=45.);ref=exhaustive_path(*args,hi=[1024.,1024.])
        replay=replay_global_path(*args,res)
        if scenario=='high_variance':
            old=selection_path(*args,max_splits=2500,time_budget=45.)
            wider=exhaustive_path(*args,hi=[2048.,2048.])
            if first_high is None:first_high=(args,res,old)
        else:
            old=json.loads((Path('results/path')/f'{label}.json').read_text())['verified_path'];wider=None
        raw={'scenario':scenario,'seed':seed,'verified_path':res,'exhaustive_multistart':ref,
             'replay':replay,'box4_verified_path':old if scenario=='high_variance' else None,
             'box4_selected':old['selected'],'box2048_multistart':wider}
        (dest/f'{label}.json').write_text(json.dumps(raw,indent=2))
        row={'scenario':scenario,'seed':seed,'status':res['status'],'selected':res['selected'],
             'additions':res['additions'],'same_full_path_box1024':decisions(res)==decisions(ref),
             'same_full_path_box2048':decisions(res)==decisions(wider) if wider else None,
             'same_full_path_box4':decisions(res)==decisions(old),'box4_status':old['status'],
             'box4_selected':old['selected'],'verified_replay':replay['valid'],'stages':len(res['stages']),
             'pair_additions':sum(len(s)==2 for s in res['additions']),
             'splits':sum(s['splits'] for s in res['stages']),
             'model_fits':res['model_fits'],'reference_model_fits':ref['model_fits'],
             'seconds':res['seconds'],'reference_seconds':ref['seconds'],
             'reference_over_verified_time':ref['seconds']/res['seconds'],
             'largest_cutoff':max(max(s['witness']['hi']) for s in res['stages'] if s['witness']),
             'largest_incumbent_ratio':max(max(s['witness']['winner_lambda']) for s in res['stages'] if s['witness']),
             'minimum_tail_margin':min(s['verification'].get('tail_margin_for_display',float('nan')) for s in res['stages'] if s['verification'])}
        rows.append(row);print(json.dumps(row),flush=True)
    checks['corruption_rejections']=corruption_checks(*first_high)
    tie_args=generate_path(918991,'exact_tie',n=160,p=6)
    tie=global_selection_path(*tie_args,max_splits=120,time_budget=15.)
    assert tie['status']=='unresolved_budget' and not tie['additions']
    checks['global_exact_tie_status']=tie['status']
    (dest/'exact_tie.json').write_text(json.dumps(tie,indent=2))
    np.savez_compressed(dest/'exact_tie_data.npz',y=tie_args[0],base=tie_args[1],x=tie_args[2],z0=tie_args[3][0],z1=tie_args[3][1])
    for label,folder,replay_function in [('high_variance_915260','global_path',replay_global_path),
                                         ('suppressor_914360','path',replay_path)]:
        record=Path('results')/folder/f'{label}.json'
        result=json.loads(record.read_text())['verified_path']
        checked=replay_function(*load_data(record.with_name(label+'_data.npz')),result)
        assert checked['valid']
        checks[f'fresh_disk_replay_{folder}']={'valid':checked['valid'],'decisions':len(checked['checks'])}
    args,res,old=first_high
    model=BlockGaussian(*args);vg=TailArbGaussian(model)
    current=(2,4);blocks=[(j,) for j in range(model.p) if j not in current]
    linear=global_family_search(model,current,blocks,vg,max_splits=2500,time_budget=45.,geometry='linear')
    logarithmic=next(s for s in res['stages'] if s['phase']=='single' and s['current']==list(current))
    checks['subdivision_diagnostic']={'scenario':'high_variance','seed':915260,'current':list(current),
        'linear_status':linear['status'],'linear_splits':linear['splits'],
        'log1p_status':logarithmic['status'],'log1p_splits':logarithmic['splits'],
        'qualification':'One prespecified stage; different subdivision and anchor locations, same analytic box bound.'}
    (dest/'linear_subdivision_diagnostic.json').write_text(json.dumps(linear,indent=2))
    summary={'cases':len(rows),'complete':sum(r['status']=='complete' for r in rows),
             'same_full_paths_box1024':sum(r['same_full_path_box1024'] for r in rows),
             'all_replayed':all(r['verified_replay'] for r in rows),
             'total_verified_stages':sum(r['stages'] for r in rows if r['verified_replay']),
             'total_pair_additions':sum(r['pair_additions'] for r in rows),
             'model_fits':sum(r['model_fits'] for r in rows),'reference_model_fits':sum(r['reference_model_fits'] for r in rows),
             'median_reference_over_verified_time':float(np.median([r['reference_over_verified_time'] for r in rows])),
             'minimum_reference_over_verified_time':min(r['reference_over_verified_time'] for r in rows),
             'maximum_reference_over_verified_time':max(r['reference_over_verified_time'] for r in rows),
             'changed_full_paths_vs_box4':sum(not r['same_full_path_box4'] for r in rows),
             'largest_cutoff':max(r['largest_cutoff'] for r in rows),
             'precision_bits':128,'n':240,'p':10,'random_effect_columns':9,
             'domain':'nonnegative_orthant','replicates_per_scenario':3,
             'comparator_box':[[0.,1024.],[0.,1024.]],
             'qualification':'Exploratory BIC path pilot. Global verified decisions versus finite-box local multistart; not a speed claim for identical optimization domains or starts. All certification costs included.',
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'python_flint':flint.__version__}}
    (dest/'rows.json').write_text(json.dumps(rows,indent=2))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'verification_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
