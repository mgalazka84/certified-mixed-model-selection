"""Stage 7: native glmmsel benchmark; prior experiments are immutable.

Use prepare, then run. Fresh workers run sequentially with one BLAS thread.
R itself and glmmsel must be installed; CSDA_RSCRIPT can locate Rscript.
"""
from pathlib import Path
import argparse, gc, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np
from run_csda_validation import dump, crossed_splits, run_one
from run_comparison_experiments import array_hash

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'results/penalized_lmm'
PREVIOUS = ROOT / 'results/csda_validation'
METHODS = ['verified_pair', 'local_pair', 'local_single', 'arss_jps', 'enet_ml', 'glmmsel']
FROZEN = ['run_penalized_lmm_comparison.py', 'run_glmmsel_native.R',
          'run_csda_validation.py', 'comparison_methods.py', 'certified_path.py',
          'certified_global_path.py', 'certified_crossed_backend.py', 'reference_arss_policy.py']


def simulation(seed, scenario):
    rng = np.random.default_rng(seed)
    n, p, groups, nt, gt = 300, 12, 30, 1200, 60
    def design(nn, gg):
        labels = rng.permutation(np.arange(nn) % gg)
        x = rng.normal(size=(nn, p))
        if scenario == 'suppressor':
            x[:, 1] = .995*x[:, 0] + np.sqrt(1-.995**2)*rng.normal(size=nn)
        return x, labels
    raw, labels = design(n, groups)
    xt, lt = design(nt, gt)
    mean, sd = raw.mean(0), raw.std(0)
    x, xt = (raw-mean)/sd, (xt-mean)/sd
    beta = np.zeros(p)
    if scenario == 'signals': beta[[0, 2, 4]] = [.8, .5, .3]
    else: beta[[0, 1, 2]] = [6., -6., .45]
    y = x@beta + rng.normal(scale=.9, size=groups)[labels] + rng.normal(size=n)
    yt = xt@beta + rng.normal(scale=.9, size=gt)[lt] + rng.normal(size=nt)
    return dict(y=y, base=np.ones((n, 1)), x=x, z0=np.eye(groups)[labels],
                label0=labels, x_test=xt, y_test=yt, beta=beta, mean_test=xt@beta,
                train_mean=mean, train_sd=sd)


def native_inputs(a, splits, where):
    folds = np.zeros(len(a['y']), dtype=int)
    for k, (tr, te) in enumerate(splits, 1):
        assert not set(a['label0'][tr]) & set(a['label0'][te])
        assert np.array_equal(np.sort(tr), np.setdiff1d(np.arange(len(folds)), te))
        assert np.all(folds[te] == 0)
        folds[te] = k
    assert np.all(folds > 0)
    raw = a['x']*a['train_sd'] + a['train_mean']
    raw_test = a['x_test']*a['train_sd'] + a['train_mean']
    # Original fixed dictionary; native RMS scaling occurs inside each fit.
    dump(where/'train.json', dict(x=raw.tolist(), y=a['y'].tolist(),
                                  cluster=a['label0'].tolist(), folds=folds.tolist()))
    dump(where/'predictors.json', dict(x=raw_test.tolist(),
                                       cluster=[f'new_{i}' for i in range(len(raw_test))]))


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads((PREVIOUS/'code_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == h for f,h in old.items())
    cases = []
    previous = json.loads((PREVIOUS/'protocol.json').read_text())
    for case in previous['cases']:
        if case['dataset'] != 'cake': continue
        case = dict(case, source='stage6_reused_cake', scenario='cake')
        ident = case['id']
        with np.load(PREVIOUS/f'{ident}_data.npz') as f: a = {k:f[k] for k in f.files}
        assert array_hash(a) == case['data_sha256']
        old_result = json.loads((PREVIOUS/f'{ident}.json').read_text())
        splits = [(np.array(s['train']), np.array(s['test']))
                  for s in old_result['results']['enet_ml']['inner_splits']]
        where = DEST/ident
        where.mkdir(exist_ok=True)
        shutil.copyfile(PREVIOUS/f'{ident}_data.npz', where/'data.npz')
        native_inputs(a, splits, where)
        cases.append(case)
    for si, scenario in enumerate(['signals', 'suppressor']):
        for rep in range(20):
            seed = 965000 + 1000*si + rep
            ident = f'onegroup_{scenario}_{rep:02d}'
            a = simulation(seed, scenario)
            where = DEST/ident
            where.mkdir(exist_ok=True)
            np.savez_compressed(where/'data.npz', **a)
            splits = crossed_splits([a['label0']], seed+700000, 3)
            native_inputs(a, splits, where)
            cases.append(dict(id=ident, dataset='simulation', scenario=scenario, kind='simulation',
                              seed=seed, n_train=300, n_test=1200, p=12, groups=[30],
                              features=[f'x{i+1}' for i in range(12)], data_sha256=array_hash(a)))
    protocol = dict(version='CSDA-stage7-v1', cases=cases, methods=METHODS,
                    comparison='Native glmmsel permits diagonal random slopes; reference BIC has a fixed random-intercept structure.',
                    glmmsel=dict(version='1.0.3', alpha=.8, local_search=True, nlambda=100,
                                 eps=1e-6, max_cd_iter=10000, max_ls_iter=100,
                                 lambda_grid='Native adaptive full-training path; same numerical lambda grid in each inner fit, without midpoint interpolation.',
                                 tuning='Three whole-group folds exactly shared with elastic net; unweighted mean response MSE; first eligible minimum.',
                                 eligibility='Finite full/inner fits and no returned CD/LS iteration limit at that lambda.',
                                 scaling='Native training-only RMS scaling of original predictor dictionary and response; no test outcomes in R inputs.',
                                 support='abs(beta * training_sd) > 1e-7 OR gamma * training_sd^2 > 1e-10; components also reported separately.'),
                    timings='Fresh case workers, sequential methods in seed-rotated order, BLAS=1. Native R fitting/tuning timer excludes startup/imports, input reading, prediction, file writing. Subprocess wall time also retained.',
                    post_ml='Same fixed support from glmmsel; refit reference random-intercept ML without random slopes. No extra tuning; not the native estimator.',
                    simulations='40 new independent one-group inputs, 20 per scenario; separate from all earlier studies.',
                    cake='Same five outer group folds plus separate full fit, no new real dataset or independent real replicates.',
                    timeout_seconds=600, failures='Retain all failures; do not score partial/unconverged selected models.')
    pp = DEST/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == protocol
    dump(pp, protocol)
    hh = {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FROZEN}
    hp = DEST/'code_sha256.json'
    if hp.exists(): assert json.loads(hp.read_text()) == hh
    dump(hp, hh)
    inputs = {str(f.relative_to(DEST)):hashlib.sha256(f.read_bytes()).hexdigest()
              for c in cases for f in [DEST/c['id']/'data.npz', DEST/c['id']/'train.json', DEST/c['id']/'predictors.json']}
    dump(DEST/'input_sha256.json', inputs)
    print(json.dumps(dict(cases=len(cases), cake=6, simulations=40)), flush=True)


def native(a, where):
    rscript = os.environ.get('CSDA_RSCRIPT') or shutil.which('Rscript')
    if not rscript: raise RuntimeError('Install R and glmmsel 1.0.3; set CSDA_RSCRIPT')
    start = time.perf_counter()
    proc = subprocess.run([rscript, '--vanilla', str(ROOT/'run_glmmsel_native.R'),
                           str(where/'train.json'), str(where/'predictors.json'), str(where)],
                          capture_output=True, text=True, timeout=500)
    wall = time.perf_counter()-start
    (where/'R_stdout.txt').write_text(proc.stdout)
    (where/'R_stderr.txt').write_text(proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stderr[-2000:])
    raw = json.loads((where/'native.json').read_text())
    if raw['status'] != 'complete_native': return raw
    beta_raw = np.array(raw['beta'])
    beta = beta_raw*a['train_sd']
    intercept = float(raw['beta0'] + a['train_mean']@beta_raw)
    gamma = np.asarray(raw['gamma'])*a['train_sd']**2
    fixed = np.flatnonzero(np.abs(beta)>1e-7).tolist()
    random = np.flatnonzero(gamma>1e-10).tolist()
    selected = sorted(set(fixed)|set(random))
    out = dict(status='complete_native', selected=selected, fixed_support=fixed,
               random_slope_support=random, estimated_intercept=intercept,
               estimated_beta=beta.tolist(), pipeline_seconds=raw['pipeline_seconds'],
               subprocess_wall_seconds=wall, selected_lambda=raw['selected_lambda'],
               cv_mean_mse=raw['cv_mean_mse'], warnings=raw['warnings'],
               path_length=len(raw['full_path']['lambda']),
               first_support=int(raw['full_path']['nnz'][0]),
               last_support=int(raw['full_path']['nnz'][-1]),
               ineligible_penalties=int(np.sum(~np.asarray(raw['eligible'], dtype=bool))))
    return out


def independent_native_check(a, where):
    raw = json.loads((where/'native.json').read_text())
    x = a['x']*a['train_sd']+a['train_mean']
    residual = a['y']-raw['beta0']-x@np.asarray(raw['beta'])
    gamma = np.array(raw['gamma'])
    assert raw['gamma0'] >= 0 and np.all(gamma >= 0)
    logdet = 0.; quadratic = 0.
    for group in np.unique(a['label0']):
        ind = a['label0']==group
        xx = x[ind]
        v = np.eye(sum(ind)) + raw['gamma0']*np.ones((sum(ind),sum(ind))) + (xx*gamma)@xx.T
        sign, ld = np.linalg.slogdet(v)
        assert sign == 1
        logdet += ld
        rr = residual[ind]
        quadratic += float(rr@np.linalg.solve(v,rr))
    n = len(a['y'])
    dense = logdet+n*np.log(quadratic/n)
    reported = raw['scaled_profile_loss'] + n*np.log(raw['response_scale_squared']) - n*np.log(n)
    gap = abs(dense-reported)
    sigma_gap = abs(quadratic/n - raw['sigma2'])
    xp = a['x_test']*a['train_sd']+a['train_mean']
    native_predictions = np.atleast_1d(np.asarray(raw['predictions'], dtype=float))
    calc = raw['beta0']+xp@np.asarray(raw['beta'])
    pred_gap = float(np.max(np.abs(native_predictions-calc))) if len(calc) else 0.
    assert gap < 1e-5, (gap, dense, reported)
    assert sigma_gap < 1e-7*max(1.,raw['sigma2']), sigma_gap
    assert pred_gap < 1e-8, pred_gap
    return dict(dense_profile_gap=gap, residual_variance_gap=sigma_gap, prediction_gap=pred_gap,
                scope='Independent grouped covariance likelihood and new-group prediction evaluation at native selected parameters; not a global optimum certificate.')


def worker(ident):
    from comparison_methods import LocalFits, coefficients
    from certified_path import BlockGaussian
    from certified_global_path import replay_global_path
    case = next(c for c in json.loads((DEST/'protocol.json').read_text())['cases'] if c['id']==ident)
    where = DEST/ident
    with np.load(where/'data.npz') as f: a={k:f[k] for k in f.files}
    assert array_hash(a) == case['data_sha256']
    training = {k:v for k,v in a.items() if k in ['y','base','x','z0','label0']}
    shift = case['seed'] % len(METHODS)
    order = METHODS[shift:]+METHODS[:shift]
    results = {}
    for method in order:
        gc.collect()
        start = time.perf_counter()
        try:
            result = native(a,where) if method=='glmmsel' else run_one(method, training, case['seed'])
        except Exception as e:
            result = dict(status='method_error', reason=repr(e), pipeline_seconds=time.perf_counter()-start)
        results[method] = result
    ng = results['glmmsel']
    if ng['status']=='complete_native':
        start = time.perf_counter()
        model=BlockGaussian(a['y'],a['base'],a['x'],[a['z0']]); fits=LocalFits(model)
        fit=fits.fit(ng['selected']); b0,beta=coefficients(model,ng['selected'],fit)
        refit_seconds = time.perf_counter()-start
        results['glmmsel_ml'] = dict(status='complete_local',selected=ng['selected'],
            estimated_intercept=float(b0[0]), estimated_beta=beta.tolist(), final_refit=fit,
            pipeline_seconds=ng['pipeline_seconds']+refit_seconds, extra_refit_seconds=refit_seconds)
    metrics, predictions = {}, {}
    for method,r in results.items():
        if r['status'] not in ['complete','complete_local','complete_native']:
            metrics[method]=None; continue
        pred=r['estimated_intercept']+a['x_test']@r['estimated_beta']
        predictions[method]=pred
        if method=='enet_ml':
            predictions['enet_native']=r['native_intercept']+a['x_test']@r['native_coefficients']
        if not len(pred): continue
        m=dict(response_mse=float(np.mean((pred-a['y_test'])**2)),
               mae=float(np.mean(np.abs(pred-a['y_test']))), selected_n=len(r['selected']))
        if 'beta' in a:
            true=set(np.flatnonzero(a['beta'])); selected=set(r['selected'])
            m.update(tpr=len(true&selected)/len(true),fdp=len(selected-true)/max(1,len(selected)),
                     exact=int(selected==true),mean_mse=float(np.mean((pred-a['mean_test'])**2)),
                     both_suppressors=int({0,1}<=selected) if case['scenario']=='suppressor' else None)
        metrics[method]=m
    vr=results['verified_pair']; lr=results['local_pair']; replay=None; agreement=None
    if vr['status']=='complete':
        rr=replay_global_path(a['y'],a['base'],a['x'],[a['z0']],vr)
        replay=dict(valid=rr['valid'],reason=rr['reason'],decisions=len(rr['checks']))
        assert rr['valid']
        if lr['status']=='complete_local':
            agreement=dict(support=vr['selected']==lr['selected'],
                path=[(s['phase'],s['winner']) for s in vr['stages']]==[(s['phase'],s['winner']) for s in lr['stages']])
    check=independent_native_check(a,where) if ng['status']=='complete_native' else None
    if case['dataset']=='cake' and vr['status']=='complete':
        old=json.loads((PREVIOUS/f'{ident}.json').read_text())['results']['verified_pair']
        assert vr['selected']==old['selected']
    np.savez_compressed(where/'predictions.npz',**predictions)
    dump(where/'result.json',dict(case=case,order=order,results=results,metrics=metrics,
                                  replay=replay,agreement=agreement,native_check=check))
    print(json.dumps(dict(id=ident,status={m:r['status'] for m,r in results.items()},
                          seconds={m:round(r['pipeline_seconds'],2) for m,r in results.items()},
                          native_selected=ng.get('selected'),native_random=ng.get('random_slope_support'))),flush=True)


def run():
    config=json.loads((DEST/'protocol.json').read_text())
    frozen=json.loads((DEST/'code_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in frozen.items())
    inputs=json.loads((DEST/'input_sha256.json').read_text())
    assert all(hashlib.sha256((DEST/f).read_bytes()).hexdigest()==h for f,h in inputs.items())
    env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in config['cases']:
        where=DEST/case['id']
        if (where/'result.json').exists(): continue
        try:
            p=subprocess.run([sys.executable,__file__,'worker','--id',case['id']],
                             capture_output=True,text=True,env=env,timeout=600,check=True)
            print(p.stdout.strip(),flush=True)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            dump(where/'result.json',dict(case=case,worker_failure=type(e).__name__,
                                         reason=str(e),stderr=str(getattr(e,'stderr',''))[-5000:]))
            print(json.dumps(dict(id=case['id'],worker_failure=type(e).__name__)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','run','worker'])
    ap.add_argument('--id'); args=ap.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='run': run()
    else: worker(args.id)
