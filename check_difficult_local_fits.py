"""Post-hoc diagnosis of flagged local fits using independent dense algebra.

Does not replace or alter any primary fit or outcome. Checks all 23 flagged
crossed fits, plus the multistart fits on the two datasets with changed choices.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/difficult_decisions'
protocol=json.loads((DEST/'protocol.json').read_text())
checks=[]; decision_differences=[]

def dense(a, support, lam):
    y=a['y']; n=len(y)
    zs=[a[k] for k in sorted(a) if k.startswith('z')]
    covs=[z@z.T for z in zs]
    r=np.eye(n)+sum(t*c for t,c in zip(lam,covs))
    ch=cho_factor(r,lower=True)
    x=np.column_stack([a['base'],a['x'][:,support]])
    ri=cho_solve(ch,np.eye(n)); rix=ri@x
    gram=x.T@rix
    beta=np.linalg.solve(gram,x.T@ri@y)
    residual=y-x@beta; u=ri@residual; s=residual@u
    value=2*np.log(np.diag(ch[0])).sum()+n*np.log(s/n)
    gradient=np.array([np.trace(ri@c)-n*(u@c@u)/s for c in covs])
    return float(value),gradient

for case in protocol['cases']:
    if case['kind']!='crossed':continue
    p=DEST/(case['id']+'.json'); r=json.loads(p.read_text())
    with np.load(DEST/(case['id']+'.npz')) as f:a={k:f[k] for k in f.files}
    v=r['methods']['verified']; assert v['status']=='verified' and r['offline_replay']['valid']
    differs=False
    for s in r['methods']['single_starts']:
        if s['winner_index']!=v['winner_index']:
            differs=True
            decision_differences.append({'id':case['id'],'start':s['start'],
                'local_winner':s['winner_index'],'certified_winner':v['winner_index'],
                'all_optimizer_success':all(f['optimizer_success'] for f in s['fits']),
                'all_stationary':all(f['stationary'] for f in s['fits'])})
        for j,fit in enumerate(s['fits']):
            if fit['stationary'] and fit['optimizer_success'] and not fit['final_upper_active']:continue
            support=[] if j==0 else [j-1]
            val,grad=dense(a,support,fit['lambda'])
            pg=np.where((np.array(fit['lambda'])<=1e-7)&(grad>0),0.,grad)
            opt=minimize(lambda x:dense(a,support,x),np.array(fit['lambda']),jac=True,
                         method='L-BFGS-B',bounds=[(0.,128.)]*2,
                         options={'ftol':1e-12,'gtol':1e-7,'maxiter':350,'maxls':40})
            checks.append({'id':case['id'],'start':s['start'],'support_index':j,
                'kind':'flagged_single_start','objective_absolute_difference':abs(val-fit['f']),
                'projected_gradient_norm_absolute_difference':abs(float(np.max(np.abs(pg)))-fit['projected_gradient_max']),
                'dense_projected_gradient_max':float(np.max(np.abs(pg))),
                'original_success':fit['optimizer_success'],'original_message':fit['message'],
                'dense_restart_decrease':val-float(opt.fun),'restart_success':bool(opt.success),
                'restart_lambda':opt.x.tolist(),
                'primary_result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if differs:
        for j,fit in enumerate(r['methods']['multistart']['fits']):
            val,grad=dense(a,[] if j==0 else [j-1],fit['lambda'])
            checks.append({'id':case['id'],'support_index':j,'kind':'multistart_control',
                           'objective_absolute_difference':abs(val-fit['f']),
                           'primary_result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

flagged=[c for c in checks if c['kind']=='flagged_single_start']
out={'status':'post_hoc_diagnostic_not_primary_benchmark',
     'frozen_primary_protocol_sha256':hashlib.sha256((DEST/'protocol.json').read_bytes()).hexdigest(),
     'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'checks':checks,'decision_differences':decision_differences,
     'maximum_objective_absolute_difference':max(c['objective_absolute_difference'] for c in checks),
     'maximum_projected_gradient_norm_absolute_difference':max(c['projected_gradient_norm_absolute_difference'] for c in flagged),
     'flagged_fits':len(flagged),'dense_restart_decrease_above_1e_6':sum(c['dense_restart_decrease']>1e-6 for c in flagged),
     'nonstationary_changed_decisions':sum(not c['all_stationary'] for c in decision_differences)}
assert out['maximum_objective_absolute_difference']<1e-8
assert out['maximum_projected_gradient_norm_absolute_difference']<1e-7
assert all(not c['all_stationary'] for c in decision_differences)
(DEST/'dense_local_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['checks','decision_differences']},indent=2))
