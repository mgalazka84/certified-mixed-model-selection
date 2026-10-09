"""Check multiple current fixed effects and the unbounded-tail construction."""
from pathlib import Path
import json
import numpy as np
from covariance_screening import CrossedGaussian,box_lower
from run_experiments import generate
from verify_math import dense_f


def main():
    y,b,x,z=generate(911,80,9,'weak')
    args=(y,np.column_stack((b,x[:,:2])),x[:,2:],z)
    m=CrossedGaussian(*args)
    err=0.;violation=0.
    for lo,hi in [(np.zeros(2),np.ones(2)*4),(np.array([.2,.3]),np.array([.8,1.4]))]:
        lb,_,_=box_lower(m,lo,hi)
        for w in [.1,.5,.9]:
            point=lo+w*(hi-lo)
            fast=m.evaluate(point)
            for j in range(m.p+1):
                dense=dense_f(args,point,j)
                err=max(err,abs(dense-fast.f[j]))
                violation=max(violation,float(lb[j]-dense))
    tail_err=0.;tail_violation=0.
    for case in json.loads(Path('results/tail_extension.json').read_text()):
        args=generate(case['seed'],400,30,case['scenario'])
        m=CrossedGaussian(*args)
        hi=np.array(case['tail']['hi'])
        for k in range(2):
            point=hi*.1;point[k]=hi[k]*2
            ev=m.evaluate(point)
            for j in [1,2,3]:
                dense=dense_f(args,point,j)
                tail_err=max(tail_err,abs(dense-ev.f[j]))
                tail_violation=max(tail_violation,case['tail']['outside_objective_lower']-dense)
    result={'multiple_fixed_effects_max_dense_error':err,'multiple_fixed_effects_bound_violation':max(0.,violation),
            'tail_region_max_dense_error':tail_err,'sampled_tail_bound_violation':max(0.,tail_violation),
            'status':'passed' if max(err,tail_err)<1e-7 and max(violation,tail_violation)<1e-8 else 'FAILED'}
    Path('results/extensions_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise AssertionError(result)


if __name__=='__main__':main()
