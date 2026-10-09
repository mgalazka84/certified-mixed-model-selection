"""Exact-rational and cross-implementation checks for the structured verifier.

Run after the scaling benchmark so validation does not contend with timed workers.
"""
from pathlib import Path
import copy
import json
import numpy as np
from flint import arb,arb_mat,fmpq,fmpq_mat,ctx
from certified_path import BlockGaussian
from certified_global_path import TailArbGaussian,global_selection_path,replay_global_path,verify_global_decision
from certified_crossed_backend import CrossedArbGaussian
from run_path_experiments import generate_path


def rational(a):
    return fmpq_mat([[fmpq(*float(v).as_integer_ratio()) for v in row] for row in np.asarray(a)])


def exact_reference(args,lam,support):
    y,base,x,zs=args;n=len(y)
    yy=rational(y[:,None]);xx=rational(np.column_stack([base,x[:,support]]))
    h=rational(np.column_stack([base,x,y]));z=rational(np.column_stack(zs))
    rr=fmpq_mat([[int(i==j) for j in range(n)] for i in range(n)])
    zz=[rational(a) for a in zs]
    for a,v in zip(zz,lam):rr+=fmpq(*float(v).as_integer_ratio())*(a*a.transpose())
    inv=rr.inv();fixed=xx.transpose()*inv*xx
    pp=inv-inv*xx*fixed.solve(xx.transpose()*inv)
    py=pp*yy;s=(yy.transpose()*py)[0,0]
    grad=[-n*((a.transpose()*py).transpose()*(a.transpose()*py))[0,0]/s for a in zz]
    ctx.prec=256
    g=arb(rr.det()).log();f=g+arb(n)*(arb(s)/n).log()
    return {'g':g,'f':f,'grad':grad,'gh':h.transpose()*inv*h,'eh':z.transpose()*inv*h}


def numeric_checks():
    rng=np.random.default_rng(918400);rows=[]
    layouts=[('larger_second',24,(3,5)),('larger_first',24,(7,4)),('unused_columns',24,(5,4)),('disconnected',24,(4,4))]
    for label,n,groups in layouts:
        if label=='unused_columns':
            labels=[np.arange(n)%3,np.arange(n)%4]
        elif label=='disconnected':
            half=np.arange(n)//12
            labels=[half*2+np.arange(n)%2,half*2+(np.arange(n)//2)%2]
        else:
            labels=[rng.permutation(np.arange(n)%g) for g in groups]
        zs=[np.eye(g)[v] for g,v in zip(groups,labels)]
        x=rng.integers(-3,4,(n,3)).astype(float);y=rng.integers(-5,6,n).astype(float)
        args=y,np.ones((n,1)),x,zs;model=BlockGaussian(*args)
        generic=TailArbGaussian(model);structured=CrossedArbGaussian(model,covariance_cache_size=1,profile_cache_size=1)
        count=0
        for lam in [[0.,0.],[0.,4.],[4.,0.],[.25,1.5],[64.,128.]]:
            for support in [(),(0,),(0,1),(0,1,2)]:
                ref=exact_reference(args,lam,support)
                for engine in (generic,structured):
                    g,gh,eh=engine.covariance(lam);f,h,grad=engine.profile(lam,support)
                    assert g.contains(ref['g']) and f.contains(ref['f'])
                    assert gh.contains(ref['gh']) and eh.contains(ref['eh'])
                    assert all(a.contains(b) for a,b in zip(grad,ref['grad']))
                count+=1
        # Revisit an evicted point; cached scalar determinants remain legitimate.
        point=[.25,1.5];ref=exact_reference(args,point,(0,1))
        assert structured.profile(point,(0,1))[0].contains(ref['f'])
        rows.append({'layout':label,'n':n,'groups':list(groups),'support_point_checks':count,
                     'exact_covariance_and_gradient_enclosures':True,'independent_256_bit_log_reference_contained':True,
                     'cache_eviction_revisit':True,'statistics':structured.statistics()})
    return rows


def main():
    dest=Path('results/scaling_path');checks={'exact_checks':numeric_checks()}
    args=generate_path(918410,'signals',n=60,p=6)
    y,base,x,zs=args
    rejected={}
    for label,zbad in [('weighted_membership',[zs[0]*.5,zs[1]]),
                       ('multiple_membership',[zs[0]+np.roll(zs[0],1,axis=1),zs[1]]),
                       ('one_component',[zs[0]])]:
        try:CrossedArbGaussian(BlockGaussian(y,base,x,zbad))
        except ValueError as exc:rejected[label]=str(exc)
        else:raise AssertionError('Unsupported covariance structure accepted')
    checks['unsupported_design_rejections']=rejected
    for label in ['high_variance_915260','high_variance_915261','high_variance_915262']:
        path=Path('results/global_path')/f'{label}.json';raw=json.loads(path.read_text())
        d=np.load(path.with_name(label+'_data.npz'),allow_pickle=False)
        aa=d['y'],d['base'],d['x'],[d['z0'],d['z1']]
        checked=replay_global_path(*aa,raw['verified_path'],verifier_factory=CrossedArbGaussian)
        assert checked['valid'];checks[label]={'valid':True,'stages':len(checked['checks'])}
    # A new generated witness is independently replayed with the old generic
    # engine from fresh files, beyond checking matching witness hashes.
    label='groups100_917200'
    raw=json.loads((dest/f'{label}_crossed.json').read_text())
    d=np.load(dest/f'{label}_data.npz',allow_pickle=False)
    aa=d['y'],d['base'],d['x'],[d['z0'],d['z1']]
    for label,factory in [('generic',None),('crossed',CrossedArbGaussian)]:
        checked=replay_global_path(*aa,raw['verified_path'],verifier_factory=factory)
        assert checked['valid'];checks[f'fresh_file_replay_{label}']={'valid':True,'stages':len(checked['checks'])}
    model=BlockGaussian(*aa);out={}
    original=next(s['witness'] for s in raw['verified_path']['stages'] if len(s['witness']['nodes'])>1)
    w=copy.deepcopy(original);w['data_sha256']='0'*64
    out['wrong_data_hash']=verify_global_decision(model,w,CrossedArbGaussian(model))
    w=copy.deepcopy(original);w['nodes'][0]['children']=w['nodes'][0]['children'][:1]
    out['missing_half_of_box']=verify_global_decision(model,w,CrossedArbGaussian(model))
    w=copy.deepcopy(original);w['winner_index']=(w['winner_index']+1)%len(w['family'])
    out['wrong_winner']=verify_global_decision(model,w,CrossedArbGaussian(model))
    assert all(not r['valid'] for r in out.values());checks['corruption_rejections']=out
    tieargs=generate_path(918991,'exact_tie',n=160,p=6)
    tie=global_selection_path(*tieargs,max_splits=120,time_budget=15.,verifier_factory=CrossedArbGaussian)
    assert tie['status']=='unresolved_budget' and not tie['additions']
    checks['exact_tie_status']=tie['status']
    y0=zs[0]@np.arange(zs[0].shape[1])+zs[1]@np.arange(zs[1].shape[1])
    zero=global_selection_path(y0,base,x,zs,max_splits=10,verifier_factory=CrossedArbGaussian)
    assert zero['status']=='unresolved_tail' and not zero['additions']
    checks['zero_floor_status']=zero['status']
    (dest/'verification_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({'exact_support_point_checks':sum(r['support_point_checks'] for r in checks['exact_checks']),
                      'fresh_file_replay_generic':checks['fresh_file_replay_generic'],
                      'unsupported_designs_rejected':len(rejected),'exact_tie':tie['status'],'zero_floor':zero['status']},indent=2))


if __name__=='__main__':main()
