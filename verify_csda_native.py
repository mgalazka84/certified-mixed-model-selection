"""Independent native statsmodels validation on full real-data models.

This validates the objective at common positive variance ratios, and reports
native local refits. It is not a selection-method speed benchmark or a proof
that either optimizer attains global ML. Run after the timed experiments.
"""
from pathlib import Path
import json,time,warnings
import numpy as np
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM,VCSpec,MixedLMParams
from certified_path import BlockGaussian
from comparison_methods import LocalFits,coefficients
from run_csda_validation import DEST,dump

def main():
    records=[]
    for dataset in ['carrots','cake']:
        ident=f'{dataset}_full'
        with np.load(DEST/f'{ident}_data.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        r=json.loads((DEST/f'{ident}.json').read_text())
        supports={(),tuple(range(a['x'].shape[1]))}
        supports.update(tuple(v['selected']) for v in r['results'].values() if v['status'] in ('complete','complete_local'))
        zs=[a[k] for k in sorted(a) if k.startswith('z')];n=len(a['y'])
        block=BlockGaussian(a['y'],a['base'],a['x'],zs);fitter=LocalFits(block)
        for support in sorted(supports):
            xx=np.column_stack([a['base'],a['x'][:,support]])
            vc=VCSpec([f'z{k}' for k in range(len(zs))],
                      [[[str(j) for j in range(z.shape[1])]] for z in zs],[[z] for z in zs])
            native=MixedLM(a['y'],xx,groups=np.ones(n),exog_re=None,exog_vc=vc)
            own=fitter.fit(support);b0,beta=coefficients(block,support,own)
            start=MixedLMParams.from_components(fe_params=np.r_[b0,beta[list(support)]],cov_re=np.zeros((0,0)),vcomp=np.maximum(own['lambda'],1e-6))
            begun=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result=native.fit(start_params=start,reml=False,method=['lbfgs','powell'],maxiter=1000,disp=False)
            point_checks=[]
            for ratios in [np.full(len(zs),.2),np.full(len(zs),1.3),np.maximum(own['lambda'],1e-5)]:
                par=MixedLMParams.from_components(fe_params=np.zeros(xx.shape[1]),cov_re=np.zeros((0,0)),vcomp=ratios)
                ll=float(native.loglike(par,profile_fe=True))
                expected=-.5*(float(block.evaluate(ratios,[support])[0][0])+n*(1+np.log(2*np.pi)))
                point_checks.append({'ratios':ratios.tolist(),'native_loglike':ll,'profile_loglike':expected,'absolute_gap':abs(ll-expected)})
                assert abs(ll-expected)<1e-6*max(1,abs(expected))
            objective=-2*float(result.llf)-n*(1+np.log(2*np.pi))
            local_pred=float(b0[0])+a['x']@beta;native_pred=xx@np.asarray(result.fe_params)
            records.append({'dataset':dataset,'support':list(support),'points':point_checks,
                            'native_converged':bool(result.converged),'native_objective':objective,
                            'shared_objective':own['f'],'native_minus_shared':objective-own['f'],
                            'max_prediction_difference':float(np.max(np.abs(local_pred-native_pred))),
                            'native_vcomp':result.vcomp.tolist(),'native_scale':float(result.scale),
                            'seconds':time.perf_counter()-begun,
                            'warnings':[str(w.message) for w in caught]})
            print(dataset,list(support),'deviance gap',objective-own['f'],'converged',bool(result.converged),flush=True)
    dump(DEST/'native_validation.json',{'package':'statsmodels','version':statsmodels.__version__,
         'warm_start':'Native refits begin at the common-engine fitted point; comparison is of independent likelihood implementations, not starting-point sensitivity.',
         'records':records,'max_profile_loglike_gap':max(p['absolute_gap'] for r in records for p in r['points'])})

if __name__=='__main__':main()
