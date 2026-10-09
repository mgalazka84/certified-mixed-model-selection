"""Targeted checks for comparator policy, prediction, and split isolation."""
from pathlib import Path
import ast
import hashlib
import inspect
import itertools
import json
import numpy as np
from scipy.linalg import cholesky,solve_triangular

from comparison_methods import (BlockGaussian,LocalFits,arss_path,residual_scores,
                                coefficients,crossed_cv_splits,performance)
from run_comparison_experiments import generate
import reference_arss_policy


def direct_arss(model,fit,alpha=.01,K=3,gate=.6,L=5):
    """Independent compact statement of the paper's rule for a tiny dataset."""
    from scipy.stats import chi2
    selected=[];pairs_added=0
    while len(selected)<model.p:
        remaining=[j for j in range(model.p) if j not in selected]
        base=fit.fit(selected)
        u,g,scores=residual_scores(model,base,selected,remaining)
        order=list(np.argsort(-scores));tested={};prefix=min(K,len(order))
        while True:
            for i in order[:prefix]:
                if i not in tested:tested[i]=fit.fit(selected+[remaining[i]],base['lambda'])['f']
            eligible=[i for i in tested if chi2.sf(max(0.,base['f']-tested[i]),1)<alpha]
            if eligible or prefix==len(order):break
            prefix=min(len(order),prefix*2)
        if eligible:
            best=min(eligible,key=lambda i:tested[i]);selected.append(remaining[best]);continue
        pool=[]
        for i,j in itertools.combinations(range(len(remaining)),2):
            rho=g[i,j]/np.sqrt(g[i,i]*g[j,j])
            if abs(rho)>=gate:
                v=u[[i,j]];score=float(v@np.linalg.solve(g[np.ix_([i,j],[i,j])],v))
                pool.append((max(0.,score),abs(rho),i,j,rho))
        pool.sort(reverse=True);passing=[]
        for _,_,i,j,_ in pool[:L]:
            f=fit.fit(selected+[remaining[i],remaining[j]],base['lambda'])['f']
            if (chi2.sf(max(0.,base['f']-f),2)<alpha/len(pool)
                and chi2.sf(max(0.,tested[i]-f),1)<alpha
                and chi2.sf(max(0.,tested[j]-f),1)<alpha):passing.append((base['f']-f,i,j))
        if not passing:break
        _,i,j=max(passing,key=lambda v:v[0]);selected.extend([remaining[i],remaining[j]]);pairs_added+=1
    return sorted(selected),pairs_added


def main():
    root=Path(__file__).parent;out=root/'results/comparison/verification_checks.json'
    source=inspect.getsource(reference_arss_policy.arss_family).rstrip('\n')
    provenance=json.loads((root/'results/comparison/arss_source.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==provenance['function_sha256']
    checks={'archived_arss_function_hash_matches':True,'policy_checks':[]}
    for i,scenario in enumerate(['signals','suppressor','correlated_null']):
        a=generate(981000+i,scenario,n=180,p=6,ntest=120)
        args=a['y'],a['base'],a['x'],[a['z0'],a['z1']]
        m=BlockGaussian(*args);f=LocalFits(m);r=arss_path(m,f)
        other=BlockGaussian(*args);selected,pairs=direct_arss(other,LocalFits(other))
        assert r['selected']==selected and r['pair_additions']==pairs
        checks['policy_checks'].append({'scenario':scenario,'selected':selected,'pairs':pairs})
    # Independent full-n whitening checks score geometry and post-ML beta.
    a=generate(982001,'suppressor',n=90,p=6,ntest=120)
    m=BlockGaussian(a['y'],a['base'],a['x'],[a['z0'],a['z1']])
    fit={'lambda':[.4,1.7]};selected=[2];active=[0,1,3,4,5]
    u,g,s=residual_scores(m,fit,selected,active)
    R=np.eye(m.n)+.4*a['z0']@a['z0'].T+1.7*a['z1']@a['z1'].T
    chol=cholesky(R,lower=True)
    b=solve_triangular(chol,np.column_stack([a['base'],a['x'][:,selected]]),lower=True)
    wx=solve_triangular(chol,a['x'][:,active],lower=True);wy=solve_triangular(chol,a['y'],lower=True)
    q=np.linalg.qr(b,mode='reduced')[0];resx=wx-q@(q.T@wx);resy=wy-q@(q.T@wy)
    assert np.allclose(g,resx.T@resx,atol=1e-9) and np.allclose(u,resx.T@resy,atol=1e-9)
    b0,beta=coefficients(m,selected,fit);dense=np.linalg.lstsq(b,wy,rcond=None)[0]
    assert np.allclose(np.r_[b0,beta[selected]],dense,atol=1e-10)
    checks['dense_score_and_coefficient_checks']=True
    a=generate(982002,'signals')
    splits=crossed_cv_splits([a['origin'],a['destination']],555)
    for train,val in splits:
        assert not set(train)&set(val)
        for key in ('origin','destination'):assert not set(a[key][train])&set(a[key][val])
    checks['inner_fold_sizes']=[{'train':len(t),'validation':len(v)} for t,v in splits]
    # Metrics and the null convention are checked with exact small examples.
    fake={'status':'complete_local','selected':[0,2,4],'estimated_intercept':0.,'estimated_beta':a['beta'].tolist()}
    met=performance(fake,a,'signals');assert met['tpr']==1 and met['fdp']==0 and met['mean_mse']==0
    fake['status']='unresolved_budget';assert performance(fake,a,'signals') is None
    checks['metrics_and_unresolved_contract']=True
    # Test arrays never enter a selector's argument dictionary in the runner.
    tree=ast.parse((root/'run_comparison_experiments.py').read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_method']
    assert len(calls)==1 and isinstance(calls[0].args[1],ast.Name) and calls[0].args[1].id=='train'
    checks['selector_receives_training_dictionary']=True
    out.write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
