"""Reproducible pilot. Set BLAS thread counts to one BEFORE starting Python."""
from pathlib import Path
import argparse
import csv
import json
import time
import platform
import numpy as np
import scipy
from covariance_screening import CrossedGaussian, screen_step, exhaustive_reference


def generate(seed, n=400, p=30, scenario="separated"):
    rng = np.random.default_rng(seed)
    go, gd = 10, 12
    if scenario == "unbalanced":
        probs = np.arange(1,go+1,dtype=float)**-1.5
        origin = rng.choice(go,n,p=probs/probs.sum())
    else:
        origin = rng.integers(go,size=n)
    dest = rng.integers(gd,size=n)
    zo, zd = np.eye(go)[origin], np.eye(gd)[dest]
    x = rng.normal(size=(n,p))
    beta = np.zeros(p)
    sdo, sdd = 0.7, 0.8
    if scenario in ("separated", "unbalanced"):
        beta[0] = 0.9
    elif scenario == "weak":
        beta[0], beta[1] = 0.24,0.14
    elif scenario == "near_tie":
        x[:,1] = 0.9995*x[:,0] + np.sqrt(1-0.9995**2)*x[:,1]
        beta[0] = 0.7
    elif scenario == "exact_tie":
        x[:,1] = x[:,0]
        beta[0] = 0.9
    elif scenario == "boundary":
        sdo,sdd = 0.0,0.2
        beta[0],beta[1] = 0.30,0.1
    elif scenario == "group_aligned":
        x[:,0] = zo @ rng.normal(size=go) + 0.12*x[:,0]
        x[:,1] = zd @ rng.normal(size=gd) + 0.15*x[:,1]
        beta[0],beta[1],beta[2] = 0.7,0.5,0.15
    elif scenario == "null":
        pass
    else:
        raise ValueError(scenario)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    y = x @ beta + zo @ rng.normal(scale=sdo,size=go) + zd @ rng.normal(scale=sdd,size=gd) + rng.normal(size=n)
    base = np.ones((n,1))
    return y,base,x,[zo,zd]


def run_case(seed,n,p,scenario,hi=4.0,max_splits=2000,time_budget=30.0):
    args = generate(seed,n,p,scenario)
    def run_screen():
        before=time.perf_counter()
        model=CrossedGaussian(*args)
        setup=time.perf_counter()-before
        return screen_step(model,[0,0],[hi,hi],max_splits=max_splits,time_budget=time_budget),setup
    def run_reference():
        before=time.perf_counter()
        model=CrossedGaussian(*args)
        setup=time.perf_counter()-before
        return exhaustive_reference(model,[0,0],[hi,hi]),setup
    if seed % 2:
        reference,reference_setup=run_reference()
        screened,screen_setup=run_screen()
    else:
        screened,screen_setup=run_screen()
        reference,reference_setup=run_reference()
    ref_f=np.array(reference['f'])
    lb=np.array(screened['global_f_lower'])
    upper_viol=float(np.max(lb-ref_f))
    fixed_base=screened['base_feasible_f']
    lr=fixed_base-ref_f
    q=np.array(screened['q_scores'])
    sort=np.sort(ref_f)
    row={"scenario":scenario,"seed":seed,"n":n,"p":p,"box_upper":hi,
         "method_order":"reference_first" if seed%2 else "screen_first",
         "status":screened['status'],"winner":screened['winner'],
         "reference_winner":reference['winner'],"agreement":screened['winner']==reference['winner'],
         "reference_gap":float(sort[1]-sort[0]),
         "full_candidate_fits":screened['distinct_refitted_candidates'],
         "omitted_full_fits":screened['omitted_full_fits'],
         "splits":screened['splits'],"remaining_candidates":screened['remaining_candidates'],
         "screen_seconds":screened['time_seconds']+screen_setup,
         "reference_seconds":reference['time_seconds']+reference_setup,
         "speedup":(reference['time_seconds']+reference_setup)/(screened['time_seconds']+screen_setup),
         "bound_violation":max(0.,upper_viol),
         "lower_Q_violation":max(0.,float(np.max(q-lr))),
         "median_B":float(np.median(screened['b_c'])),
         "separation_margin":screened['separation_margin']}
    return row,{"screen":screened,"reference":reference}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results')
    parser.add_argument('--replicates',type=int,default=3)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--max-splits',type=int,default=1200)
    parser.add_argument('--time-budget',type=float,default=20)
    opts=parser.parse_args()
    out=Path(opts.out);out.mkdir(exist_ok=True,parents=True)
    rows=[];details=[]
    scenarios=['separated','weak','group_aligned','unbalanced','boundary','null','near_tie']
    if opts.quick: scenarios=['separated','weak','group_aligned']
    for k,scenario in enumerate(scenarios):
        for r in range(opts.replicates):
            seed=202609140+k*100+r
            row,detail=run_case(seed,400,30,scenario,max_splits=opts.max_splits,time_budget=opts.time_budget)
            rows.append(row);details.append(detail)
            print(json.dumps(row),flush=True)
            (out/'details.json').write_text(json.dumps(details,indent=2))
            with (out/'pilot.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    meta={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
          "blas_threads":"Run command fixes OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1",
          "note":"Pilot one-step ranking, not complete selection, FDR assessment, or global unrestricted ML."}
    (out/'environment.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':main()
