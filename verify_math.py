"""Independent dense checks of algebra, derivatives, convexity, and box bounds."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from covariance_screening import CrossedGaussian, box_lower
from run_experiments import generate


def dense_f(args,lam,j):
    y,base,x,zs=args
    r=np.eye(len(y))
    for a,z in zip(lam,zs):r+=a*(z@z.T)
    l=cholesky(r,lower=True)
    design=base if j==0 else np.column_stack((base,x[:,j-1]))
    wy=solve_triangular(l,y,lower=True)
    wx=solve_triangular(l,design,lower=True)
    beta=np.linalg.lstsq(wx,wy,rcond=None)[0]
    residual=wy-wx@beta
    s=residual@residual
    return 2*np.log(np.diag(l)).sum()+len(y)*np.log(s/len(y))


def main():
    args=generate(79117,96,8,'group_aligned')
    model=CrossedGaussian(*args)
    rng=np.random.default_rng(86420)
    dense_error=0.;gradient_error=0.;jensen_violation=0.;bound_violation=0.
    dense_checks=0;bound_checks=0
    for lam in [np.zeros(2),np.array([0.,4.]),np.array([4.,0.]),np.array([.3,.7]),np.array([4.,4.])]:
        fast=model.evaluate(lam)
        for j in range(model.p+1):
            dense_error=max(dense_error,abs(fast.f[j]-dense_f(args,lam,j)));dense_checks+=1
    for _ in range(30):
        lam=rng.uniform(.1,3.8,2)
        ev=model.evaluate(lam)
        for k in range(2):
            d=np.eye(2)[k]*1e-5
            grad=(model.evaluate(lam+d).f-model.evaluate(lam-d).f)/(2e-5)
            gradient_error=max(gradient_error,float(np.max(np.abs(grad-ev.h_gradient[:,k]-ev.g_gradient[k]))))
        a,b=rng.uniform(0,4,(2,2));t=rng.uniform()
        diff=model.evaluate(t*a+(1-t)*b).h-t*model.evaluate(a).h-(1-t)*model.evaluate(b).h
        jensen_violation=max(jensen_violation,float(diff.max()))
    for _ in range(50):
        edges=np.sort(rng.uniform(0,4,(2,2)),axis=0)
        lo,hi=edges
        lower,_,_=box_lower(model,lo,hi)
        for t in range(15):
            lam=rng.uniform(lo,hi)
            truth=np.array([dense_f(args,lam,j) for j in range(model.p+1)])
            bound_violation=max(bound_violation,float((lower-truth).max()))
            bound_checks+=len(truth)
    result={"dense_objective_checks":dense_checks,"max_dense_objective_error":dense_error,
            "max_gradient_error":gradient_error,"max_convexity_violation":max(0.,jensen_violation),
            "dense_box_bound_checks":bound_checks,"max_box_bound_violation":max(0.,bound_violation),
            "status":"passed" if dense_error<1e-8 and gradient_error<1e-5 and jensen_violation<1e-8 and bound_violation<1e-8 else "FAILED",
            "qualification":"Numerical diagnostics support implementation; proof is in the note. This is not interval-arithmetic verification."}
    Path('results').mkdir(exist_ok=True)
    Path('results/validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise AssertionError(result)


if __name__=='__main__':main()
