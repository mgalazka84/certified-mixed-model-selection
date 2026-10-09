"""Matched full-path backend scaling in fresh, single-threaded subprocesses.

Only the verified numerical backend changes. Both runs use the same float64
search, raw data, budgets, precision and full-orthant BIC policy. Exact equality
of their serialized decision witnesses checks the finite boxes, partitions,
anchors and feasible points, in addition to equality of selected paths.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import numpy as np


CONFIGS=[
    ('groups25',800,20,(10,15)),
    ('groups50',800,20,(20,30)),
    ('groups100',800,20,(40,60)),
    ('groups200',800,20,(80,120)),
    ('candidates40',800,40,(20,30)),
    ('candidates60',800,60,(20,30)),
]


def generate(seed,n,p,groups,scenario):
    rng=np.random.default_rng(seed)
    # Balanced label counts with independently permuted memberships; every
    # intended group is represented. No full Cartesian grid is required.
    labels=[rng.permutation(np.arange(n)%g) for g in groups]
    zs=[np.eye(g)[v] for g,v in zip(groups,labels)]
    x=rng.normal(size=(n,p));beta=np.zeros(p)
    if scenario=='signals':
        beta[[0,2]]=[.9,.55]
    elif scenario=='suppressor':
        x[:,1]=.995*x[:,0]+np.sqrt(1-.995**2)*rng.normal(size=n)
        beta[[0,1,2]]=[6.,-6.,.6]
    else:
        raise ValueError(scenario)
    x=(x-x.mean(0))/x.std(0)
    y=x@beta+sum(z@rng.normal(scale=s,size=g) for z,g,s in zip(zs,groups,[.7,.8]))+rng.normal(size=n)
    return y,np.ones((n,1)),x,zs


def witness_hash(result):
    data=json.dumps([s['witness'] for s in result['stages']],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def worker(args):
    if sys.platform!='linux':
        raise RuntimeError('This process-RSS benchmark currently requires Linux')
    import scipy,flint
    from certified_global_path import global_selection_path
    from certified_crossed_backend import CrossedArbGaussian
    data=np.load(args.worker,allow_pickle=False)
    modelargs=data['y'],data['base'],data['x'],[data['z0'],data['z1']]
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result=global_selection_path(*modelargs,bits=128,max_splits=2000,time_budget=45.,
                                 verifier_factory=CrossedArbGaussian if args.backend=='crossed' else None)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    record={'backend':args.backend,'verified_path':result,
            'process_peak_rss_mib':peak,'process_peak_before_path_mib':before,
            'peak_increase_mib':max(0.,peak-before),'witness_sha256':witness_hash(result),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,
                           'scipy':scipy.__version__,'python_flint':flint.__version__,
                           'platform':sys.platform,'blas_threads':1}}
    args.out.write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps({'backend':args.backend,'status':result['status'],'seconds':result['seconds'],
                      'process_peak_rss_mib':peak,'stages':len(result['stages'])}),flush=True)


def main(args):
    dest=Path('results/scaling_path');dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for ci,(name,n,p,groups) in enumerate(CONFIGS):
        for rep in range(3):
            seed=917000+ci*100+rep;scenario='suppressor' if rep==2 else 'signals'
            label=f'{name}_{seed}';datafile=dest/f'{label}_data.npz'
            aa=generate(seed,n,p,groups,scenario)
            np.savez_compressed(datafile,y=aa[0],base=aa[1],x=aa[2],z0=aa[3][0],z1=aa[3][1])
            records={};failures={}
            order=['generic','crossed'] if seed%2 else ['crossed','generic']
            for backend in order:
                out=dest/f'{label}_{backend}.json'
                print(json.dumps({'starting':label,'backend':backend,'groups':groups,'p':p}),flush=True)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(datafile),
                     '--backend',backend,'--out',str(out)]
                try:
                    done=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=240,check=True)
                    record=json.loads(out.read_text());records[backend]=record
                    print(done.stdout.strip(),flush=True)
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                    failure={'status':'external_timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'worker_error',
                             'reason':str(exc),'stderr':str(getattr(exc,'stderr',''))[-3000:]}
                    failures[backend]=failure;out.write_text(json.dumps(failure,indent=2))
                    print(json.dumps(failure),flush=True)
            row={'configuration':name,'n':n,'p':p,'groups':list(groups),'random_columns':sum(groups),
                 'seed':seed,'scenario':scenario,'run_order':order,'failures':failures}
            if len(records)==2:
                a,b=records['generic'],records['crossed'];ra,rb=a['verified_path'],b['verified_path']
                row.update(generic_status=ra['status'],crossed_status=rb['status'],
                    same_full_path=[(s['phase'],s['winner']) for s in ra['stages']]==[(s['phase'],s['winner']) for s in rb['stages']],
                    identical_witness=a['witness_sha256']==b['witness_sha256'],
                    selected=rb['selected'],additions=rb['additions'],stages=len(rb['stages']),
                    pair_additions=sum(len(s)==2 for s in rb['additions']),
                    splits=sum(s['splits'] for s in rb['stages']),
                    generic_seconds=ra['seconds'],crossed_seconds=rb['seconds'],
                    generic_over_crossed_time=ra['seconds']/rb['seconds'],
                    generic_peak_rss_mib=a['process_peak_rss_mib'],crossed_peak_rss_mib=b['process_peak_rss_mib'],
                    generic_peak_increase_mib=a['peak_increase_mib'],crossed_peak_increase_mib=b['peak_increase_mib'],
                    rss_reduction_percent=100*(1-b['process_peak_rss_mib']/a['process_peak_rss_mib']),
                    generic_model_fits=ra['model_fits'],crossed_model_fits=rb['model_fits'],
                    generic_verification_seconds=sum(s['verification']['seconds'] for s in ra['stages'] if s['verification']),
                    crossed_verification_seconds=sum(s['verification']['seconds'] for s in rb['stages'] if s['verification']),
                    verifier_statistics=rb['verifier_statistics'])
            rows.append(row);(dest/'rows.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    good=[r for r in rows if not r['failures'] and r['generic_status']=='complete' and r['crossed_status']=='complete']
    summary={'cases':len(rows),'both_complete':len(good),
             'same_full_paths':sum(r['same_full_path'] for r in good),
             'identical_witnesses':sum(r['identical_witness'] for r in good),
             'verified_stages_per_backend':sum(r['stages'] for r in good),
             'pair_additions_per_backend':sum(r['pair_additions'] for r in good),
             'median_generic_over_crossed_time':float(np.median([r['generic_over_crossed_time'] for r in good])) if good else None,
             'minimum_generic_over_crossed_time':min((r['generic_over_crossed_time'] for r in good),default=None),
             'maximum_generic_over_crossed_time':max((r['generic_over_crossed_time'] for r in good),default=None),
             'median_peak_rss_reduction_percent':float(np.median([r['rss_reduction_percent'] for r in good])) if good else None,
             'maximum_generic_peak_rss_mib':max((r['generic_peak_rss_mib'] for r in good),default=None),
             'maximum_crossed_peak_rss_mib':max((r['crossed_peak_rss_mib'] for r in good),default=None),
             'n':800,'candidate_counts':[20,40,60],'random_columns':[25,50,100,200],
             'qualification':'Development scaling pilot, three datasets per configuration: two signal and one suppressor. Matched full-orthant policy and float search. Fresh subprocess per backend; Linux process high-water RSS includes interpreter, imports and data. No profiling or replay is included in timed runs.'}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path);parser.add_argument('--backend',choices=['generic','crossed'])
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    worker(args) if args.worker else main(args)
