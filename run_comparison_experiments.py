"""Repeated method comparison, separate from the earlier development pilots.

One fresh process per dataset, independent model objects per method, rotating
method order, and one BLAS thread. All selection and final ML refitting are
timed. Test evaluation, independent replay, file I/O and imports are excluded.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np


SCENARIOS=['signals','weak','suppressor','null','correlated_null','group_aligned']
REPLICATES=50


def protocol():
    return {'version':'stage5-20260914-v1','replicates_per_scenario':REPLICATES,
            'scenarios':SCENARIOS,'seeds':'918000 + 1000*scenario_index + replicate',
            'train_n':300,'test_n':1200,'p':12,'train_groups':[6,8],'test_groups':[24,32],
            'test_target':'Population mean and responses in entirely new groups on BOTH margins',
            'default_random_sd':[.7,.8],'residual_sd':1.,
            'signal_beta':{'signals':[.8,.5,.3],'weak':[.35,.25,.15]},
            'signal_indices':[0,2,4],
            'suppressor':{'correlation':.995,'indices':[0,1,2],'beta':[6.,-6.,.45]},
            'correlated_null':{'equicorrelation':.8},
            'group_aligned':{'random_sd':[2.,3.],'indices':[2,4],'beta':[.65,.4],
                             'x0':'origin-level N(0,1) plus 0.1 N(0,1)',
                             'x1':'destination-level N(0,1) plus 0.1 N(0,1)'},
            'scaling':'Train means/SD applied to test; beta defined on train-standardized scale',
            'methods':['verified_pair','local_pair','local_single','arss_jps','enet_ml'],
            'verified':{'bits':128,'max_splits':2500,'soft_seconds_per_family':30.},
            'local':{'initial_upper':4.,'max_expansions':6,'factor':4.,
                      'starts':['zero','half upper','0.1 upper','0.9 upper','optional warm point']},
            'arss':{'alpha':.01,'single_K':3,'gate':.60,'pair_L':5,
                    'source_commit':'0cdcf976178be7ed00af4b0737a7ed97540786a4',
                    'adaptation':'Archived search function unchanged; common profile-ML fit and score adapter'},
            'enet':{'alphas':np.logspace(-3,0,13).tolist(),'l1_ratio':[.5,.9,1.],
                     'inner_folds':3,'tuning_target':'MSE on both-margin-held-out cells',
                     'standardization':'Inside each inner training fold',
                     'support_threshold':1e-7,'max_iter':30000,'tol':1e-7,
                     'post_selection':'Unpenalized ML refit with both random intercepts'},
            'worker_timeout_seconds':360,
            'failure_policy':'Keep all failures; no partial path is a finished selection; report denominators',
            'uncertainty':'Dataset-level Monte Carlo SE; Wilson 95% intervals for events; paired differences',
            'qualification':'Repeated simulation on a fixed new seed grid, not a preregistered or external validation study.'}


def generate(seed, scenario, n=300, p=12, ntest=1200):
    rng=np.random.default_rng(seed)
    def draw(rows,groups):
        labels=[rng.permutation(np.arange(rows)%g) for g in groups]
        x=rng.normal(size=(rows,p))
        if scenario=='suppressor':
            x[:,1]=.995*x[:,0]+np.sqrt(1-.995**2)*rng.normal(size=rows)
        elif scenario=='correlated_null':
            x=np.sqrt(.8)*rng.normal(size=(rows,1))+np.sqrt(.2)*x
        elif scenario=='group_aligned':
            x[:,0]=rng.normal(size=groups[0])[labels[0]]+.1*rng.normal(size=rows)
            x[:,1]=rng.normal(size=groups[1])[labels[1]]+.1*rng.normal(size=rows)
        return x,labels
    x,labels=draw(n,(6,8));xt,lt=draw(ntest,(24,32))
    means=x.mean(0);sdx=x.std(0);x=(x-means)/sdx;xt=(xt-means)/sdx
    beta=np.zeros(p);sd=[.7,.8]
    if scenario=='signals':beta[[0,2,4]]=[.8,.5,.3]
    elif scenario=='weak':beta[[0,2,4]]=[.35,.25,.15]
    elif scenario=='suppressor':beta[[0,1,2]]=[6.,-6.,.45]
    elif scenario=='group_aligned':sd=[2.,3.];beta[[2,4]]=[.65,.4]
    elif scenario not in ('null','correlated_null'):raise ValueError(scenario)
    y=x@beta+sum(rng.normal(scale=s,size=g)[v] for g,v,s in zip((6,8),labels,sd))+rng.normal(size=n)
    target=xt@beta
    yt=target+sum(rng.normal(scale=s,size=g)[v] for g,v,s in zip((24,32),lt,sd))+rng.normal(size=ntest)
    return {'y':y,'base':np.ones((n,1)),'x':x,'z0':np.eye(6)[labels[0]],'z1':np.eye(8)[labels[1]],
            'origin':labels[0],'destination':labels[1],'beta':beta,'train_mean':means,'train_sd':sdx,
            'x_test':xt,'mean_test':target,'y_test':yt,'test_origin':lt[0],'test_destination':lt[1]}


def array_hash(arrays):
    h=hashlib.sha256()
    for key in sorted(arrays):
        a=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(str(a.shape).encode())
        h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def worker(args):
    import scipy,sklearn,flint
    from comparison_methods import METHODS,run_method,performance
    from certified_global_path import replay_global_path
    from certified_crossed_backend import CrossedArbGaussian
    with np.load(args.worker,allow_pickle=False) as loaded:arrays={k:loaded[k] for k in loaded.files}
    train={k:arrays[k] for k in ['y','base','x','z0','z1','origin','destination']}
    shift=args.seed%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
    results={}
    for method in order:
        gc.collect()
        begun=time.perf_counter()
        try:
            result=run_method(method,train,args.seed)
        except Exception as exc:
            result={'status':'method_error','reason':repr(exc),'pipeline_seconds':time.perf_counter()-begun}
        results[method]=result
    # Held-out outcomes first enter the computation here, after every selector.
    metrics={m:performance(r,arrays,args.scenario) for m,r in results.items()}
    cert=results['verified_pair']
    replay=None
    if cert['status']=='complete':
        checked=replay_global_path(arrays['y'],arrays['base'],arrays['x'],[arrays['z0'],arrays['z1']],cert,
                                    verifier_factory=CrossedArbGaussian)
        replay={'valid':checked['valid'],'stages':len(checked.get('checks',[])),'reason':checked['reason']}
        if not checked['valid']:raise AssertionError('Saved result failed independent replay')
    local=results['local_pair']
    comparison=None
    if cert['status']=='complete' and local['status']=='complete_local':
        comparison={'same_support':cert['selected']==local['selected'],
                    'same_full_path':[(s['phase'],s['winner']) for s in cert['stages']]==[(s['phase'],s['winner']) for s in local['stages']]}
    record={'scenario':args.scenario,'seed':args.seed,'data_sha256':array_hash(arrays),
            'run_order':order,'results':results,'metrics':metrics,'verified_replay':replay,
            'local_comparison':comparison,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                           'sklearn':sklearn.__version__,'python_flint':flint.__version__,'blas_threads':1}}
    args.out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':args.scenario,'seed':args.seed,
          'statuses':{m:r['status'] for m,r in results.items()},
          'seconds':{m:round(r['pipeline_seconds'],3) for m,r in results.items()},
          'local_comparison':comparison}),flush=True)


def main(args):
    dest=args.out or Path('results/comparison');dest.mkdir(parents=True,exist_ok=True)
    pr=protocol()
    if args.development:pr={**pr,'development_only':True,'replicates_per_scenario':1,'seeds':'990000 + 1000*scenario_index'}
    pp=dest/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=pr:raise ValueError('Existing protocol differs; use a separate output directory')
    pp.write_text(json.dumps(pr,indent=2,allow_nan=False)+'\n')
    sourcefiles=['comparison_methods.py','run_comparison_experiments.py','reference_arss_policy.py',
                 'certified_path.py','certified_global_path.py','certified_crossed_backend.py']
    hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in sourcefiles}
    hp=dest/'code_sha256.json'
    if hp.exists() and json.loads(hp.read_text())!=hashes:raise ValueError('Code differs from frozen run; use a new output directory')
    hp.write_text(json.dumps(hashes,indent=2)+'\n')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for si,scenario in enumerate(SCENARIOS):
        for rep in range(pr['replicates_per_scenario']):
            seed=(990000 if args.development else 918000)+si*1000+rep
            name=f'{scenario}_{seed}';datafile=dest/f'{name}_data.npz';outfile=dest/f'{name}.json'
            arrays=generate(seed,scenario)
            if outfile.exists():
                old=json.loads(outfile.read_text())
                if old.get('data_sha256')!=array_hash(arrays):raise ValueError('Existing result has different data')
                continue
            np.savez_compressed(datafile,**arrays)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(datafile),'--out',str(outfile),
                 '--seed',str(seed),'--scenario',scenario]
            try:
                done=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=360,check=True)
                print(done.stdout.strip(),flush=True)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                failure={'scenario':scenario,'seed':seed,'data_sha256':array_hash(arrays),
                         'worker_failure':'timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'error',
                         'reason':str(exc),'stderr':str(getattr(exc,'stderr',''))[-3000:]}
                outfile.write_text(json.dumps(failure,indent=2)+'\n');print(json.dumps(failure),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',type=Path);ap.add_argument('--out',type=Path)
    ap.add_argument('--seed',type=int);ap.add_argument('--scenario',choices=SCENARIOS)
    ap.add_argument('--development',action='store_true')
    args=ap.parse_args();worker(args) if args.worker else main(args)
