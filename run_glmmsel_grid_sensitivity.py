"""Secondary low-penalty extension on every frozen stage-seven dataset."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
import numpy as np
from run_penalized_lmm_comparison import independent_native_check, dump
from comparison_methods import LocalFits,coefficients
from certified_path import BlockGaussian

ROOT=Path(__file__).resolve().parent
PRIMARY=ROOT/'results/penalized_lmm'
DEST=ROOT/'results/penalized_lmm_grid_sensitivity'

def main():
    DEST.mkdir(exist_ok=True)
    cases=json.loads((PRIMARY/'protocol.json').read_text())['cases']
    frozen=json.loads((PRIMARY/'code_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in frozen.items())
    files=['run_glmmsel_grid_sensitivity.py','run_glmmsel_extended.R']
    protocol=dict(version='CSDA-stage7-grid-sensitivity-v1',cases=[c['id'] for c in cases],
                  reason='11 of 46 primary choices occurred at the last native path value.',
                  extension='All original lambda values plus lambda_last/10, lambda_last/100, and 0; sorted unique descending; fit all 46 cases.',
                  invariant='Same alpha=.8, model class, data, inner folds, convergence gates and validation criterion. No new test-driven tuning.',
                  code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                  primary_outputs_sha256={c['id']:hashlib.sha256((PRIMARY/c['id']/'native.json').read_bytes()).hexdigest() for c in cases})
    pp=DEST/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    dump(pp,protocol)
    rscript=os.environ.get('CSDA_RSCRIPT') or shutil.which('Rscript')
    if not rscript:raise RuntimeError('Rscript required')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in cases:
        where=DEST/case['id'];where.mkdir(exist_ok=True)
        if (where/'result.json').exists():continue
        origin=PRIMARY/case['id']
        start=time.perf_counter()
        try:
            proc=subprocess.run([rscript,'--vanilla',str(ROOT/'run_glmmsel_extended.R'),
                str(origin/'train.json'),str(origin/'predictors.json'),str(where),str(origin/'native.json')],
                env=env,capture_output=True,text=True,timeout=500)
            (where/'R_stdout.txt').write_text(proc.stdout);(where/'R_stderr.txt').write_text(proc.stderr)
            if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
            raw=json.loads((where/'native.json').read_text())
            if raw['status']!='complete_native':raise RuntimeError(raw.get('reason','native fit failed'))
            with np.load(origin/'data.npz') as f:a={k:f[k] for k in f.files}
            beta=np.asarray(raw['beta'])*a['train_sd']
            gamma=np.asarray(raw['gamma'])*a['train_sd']**2
            selected=sorted(set(np.flatnonzero(np.abs(beta)>1e-7))|set(np.flatnonzero(gamma>1e-10)))
            selected=[int(j) for j in selected]
            check=independent_native_check(a,where)
            native_pred=np.asarray(raw['predictions'],dtype=float)
            model=BlockGaussian(a['y'],a['base'],a['x'],[a['z0']]);fits=LocalFits(model)
            fit=fits.fit(selected);b0,b=coefficients(model,selected,fit)
            post=b0[0]+a['x_test']@b
            metrics={}
            for method,pred in [('glmmsel',native_pred),('glmmsel_ml',post)]:
                if not len(a['y_test']):continue
                m=dict(response_mse=float(np.mean((pred-a['y_test'])**2)),
                       mae=float(np.mean(np.abs(pred-a['y_test']))),selected_n=len(selected))
                if 'beta' in a:
                    true=set(np.flatnonzero(a['beta']));s=set(selected)
                    m.update(tpr=len(true&s)/len(true),fdp=len(s-true)/max(1,len(s)),
                             mean_mse=float(np.mean((pred-a['mean_test'])**2)),
                             both_suppressors=int({0,1}<=s) if case['scenario']=='suppressor' else None)
                metrics[method]=m
            np.savez_compressed(where/'predictions.npz',glmmsel=native_pred,glmmsel_ml=post)
            result=dict(case=case,status='complete_native',selected=selected,
                        random_support=np.flatnonzero(gamma>1e-10).tolist(),
                        selected_lambda=raw['selected_lambda'],metrics=metrics,native_check=check,
                        native_seconds=raw['pipeline_seconds'],warnings=raw['warnings'],
                        ineligible_penalties=int(np.sum(~np.asarray(raw['eligible'],dtype=bool))),
                        selected_endpoint=raw['selected_index']==len(raw['full_path']['lambda']))
        except Exception as e:
            result=dict(case=case,status='error',reason=repr(e),elapsed=time.perf_counter()-start)
        dump(where/'result.json',result)
        print(json.dumps(dict(id=case['id'],status=result['status'],selected=result.get('selected'),
                              lambda_value=result.get('selected_lambda'))),flush=True)

if __name__=='__main__':main()
