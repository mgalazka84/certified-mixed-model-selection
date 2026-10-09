"""Penalty-grid sensitivity, preserving the frozen primary comparison.

The wider grid includes every primary penalty, extends to 10, and adds 1e6
as an intercept-only candidate. It is a secondary development analysis;
it is not substituted into the original five-method timing comparison.
Run only after the main benchmark completes to avoid CPU contention.
"""
from pathlib import Path
import gc
import hashlib
import json
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from comparison_methods import BlockGaussian,LocalFits,coefficients,crossed_cv_splits,performance


def main():
    root=Path(__file__).parent;src=root/'results/comparison'
    dest=root/'results/enet_sensitivity';dest.mkdir(exist_ok=True)
    paths=sorted(src.glob('*_[0-9]*.json'))
    assert len(paths)==300,'Wait for all main benchmark records'
    alphas=np.r_[np.logspace(-3,1,17),1e6]
    assert np.array_equal(alphas[:13],np.logspace(-3,0,13))
    protocol={'version':'stage5-enet-grid-sensitivity-v2','alphas':alphas.tolist(),
              'l1_ratios':[.5,.9,1.],'source':'same 300 primary input files and same inner folds',
              'reason':'The fixed upper penalty 1 may omit the intercept-only model at larger response scales.',
              'interpretation':'Secondary grid-sensitivity analysis; original results are retained.',
              'timing':'One process, sequential datasets; not part of the five-method timing comparison.',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pp=dest/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Sensitivity protocol changed'
    pp.write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for path in paths:
        original=json.loads(path.read_text());output=dest/path.name
        if output.exists():
            row=json.loads(output.read_text());assert row['source_data_sha256']==original['data_sha256']
            rows.append(row);continue
        with np.load(path.with_name(path.stem+'_data.npz'),allow_pickle=False) as data:
            a={k:data[k] for k in data.files}
        # Independent selector sees training predictors/response only.
        gc.collect();start=time.perf_counter()
        splits=crossed_cv_splits([a['origin'],a['destination']],original['seed']+700000)
        # KKT condition at beta=0 proves the last penalty includes a constant
        # fit for every mixing value, within each split and on the full data.
        for train in [t for t,_ in splits]+[np.arange(len(a['y']))]:
            xx=StandardScaler().fit_transform(a['x'][train]);yy=a['y'][train]
            max_score=float(np.max(np.abs(xx.T@(yy-yy.mean())))/len(train))
            assert 1e6*.5>max_score
        model=Pipeline([('scale',StandardScaler()),('enet',ElasticNet(max_iter=30000,tol=1e-7,selection='cyclic'))])
        cv=GridSearchCV(model,{'enet__alpha':alphas,'enet__l1_ratio':[.5,.9,1.]},
                        cv=splits,scoring='neg_mean_squared_error',n_jobs=1,error_score='raise',refit=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);cv.fit(a['x'],a['y'])
        scaler=cv.best_estimator_.named_steps['scale'];enet=cv.best_estimator_.named_steps['enet']
        coef=enet.coef_/scaler.scale_;intercept=float(enet.intercept_-scaler.mean_@coef)
        support=np.flatnonzero(np.abs(coef)>1e-7).tolist()
        # The original 39 combinations form an exact subgrid: best validation
        # MSE cannot increase. This checks CV split and loss comparability.
        old=original.get('results',{}).get('enet_ml',{})
        if old.get('status')=='complete_local':
            assert -cv.best_score_<=old['cv_mean_mse']+1e-8
        bg=BlockGaussian(a['y'],a['base'],a['x'],[a['z0'],a['z1']]);fitter=LocalFits(bg)
        refit=fitter.fit(support);b0,beta=coefficients(bg,support,refit)
        result={'status':'complete_local','selected':support,'estimated_beta':beta.tolist(),
                'estimated_intercept':float(b0[0]),'native_coefficients':coef.tolist(),'native_intercept':intercept,
                'best_parameters':{k:float(v) for k,v in cv.best_params_.items()},
                'cv_mean_mse':float(-cv.best_score_),'final_refit':refit,
                'warning_count':sum(issubclass(w.category,ConvergenceWarning) for w in caught),
                'development_seconds':time.perf_counter()-start}
        # Reconstructed coefficients must reproduce the package's prediction.
        assert np.allclose(cv.predict(a['x_test']),intercept+a['x_test']@coef,atol=1e-10)
        metrics=performance(result,a,original['scenario'])
        row={'scenario':original['scenario'],'seed':original['seed'],'source_data_sha256':original['data_sha256'],
             'result':result,'metrics':metrics,'same_support_primary':support==old.get('selected'),
             'original_alpha':old.get('best_parameters',{}).get('enet__alpha'),
             'grid_changed_choice':bool(cv.best_params_['enet__alpha']>1.)}
        output.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
        if len(rows)%25==0:print(json.dumps({'complete':len(rows),'larger_penalty_selected':sum(r['grid_changed_choice'] for r in rows)}),flush=True)
    (dest/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'complete':len(rows),'larger_penalty_selected':sum(r['grid_changed_choice'] for r in rows),
                      'changed_supports':sum(not r['same_support_primary'] for r in rows),
                      'warnings':sum(r['result']['warning_count'] for r in rows)}),flush=True)


if __name__=='__main__':main()
