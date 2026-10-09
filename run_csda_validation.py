"""Stage 6: new-group applications and simultaneous dimension growth.

The stage-five source files and data are left unchanged. All new cases, source
hashes, folds and failed runs are retained. Run with BLAS threads set to one.
"""
from pathlib import Path
import argparse, gc, hashlib, itertools, json, os, subprocess, sys, time, warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/csda_validation'
METHODS=['verified_pair','local_pair','local_single','arss_jps','enet_ml','null_ml','full_ml']

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def partition(labels,seed,folds=3):
    rng=np.random.default_rng(seed)
    return [rng.permutation(np.arange(int(v.max())+1))%folds for v in labels]

def crossed_splits(labels,seed,folds=3):
    assignments=partition(labels,seed,folds)
    rowfolds=[a[v] for a,v in zip(assignments,labels)]
    result=[]
    for cell in itertools.product(range(folds),repeat=len(labels)):
        train=np.logical_and.reduce([v!=k for v,k in zip(rowfolds,cell)])
        test=np.logical_and.reduce([v==k for v,k in zip(rowfolds,cell)])
        ti,vi=np.flatnonzero(train),np.flatnonzero(test)
        if len(ti)<10 or len(vi)<5:raise ValueError('Insufficient fold size')
        for v in labels:assert not set(v[ti])&set(v[vi])
        result.append((ti,vi))
    assert np.array_equal(np.sort(np.concatenate([v for _,v in result])),np.arange(len(labels[0])))
    return result

def real_data(name):
    raw=pd.read_csv(ROOT/f'data/real/{name}.csv')
    if name=='carrots':
        # No imputation of the three missing outcomes; unused columns play no role.
        df=raw.dropna(subset=['Preference','sens1','sens2','Gender','Homesize']).copy()
        s1=df.sens1.to_numpy();s2=df.sens2.to_numpy()
        gender=(df.Gender==2).to_numpy(dtype=float);home=(df.Homesize==3).to_numpy(dtype=float)
        x=np.column_stack([s1,s2,gender,home,s1*gender,s2*gender,s1*home,s2*home])
        names=['sens1','sens2','Gender_2','Homesize_3','sens1:Gender_2','sens2:Gender_2','sens1:Homesize_3','sens2:Homesize_3']
        y=df.Preference.to_numpy(dtype=float)
        labels=[pd.factorize(df[c],sort=True)[0] for c in ['Consumer','Product']]
        outer=crossed_splits(labels,962101,3)
    elif name=='cake':
        df=raw;t=(df.temp.to_numpy()-200.)/25.
        x=np.column_stack([t,t*t,t*t*t,(df.recipe=='B').astype(float),(df.recipe=='C').astype(float)])
        names=['temperature_linear','temperature_quadratic','temperature_cubic','recipe_B','recipe_C']
        y=df.angle.to_numpy(dtype=float)
        group=df.recipe.astype(str)+':'+df.replicate.astype(str)
        labels=[pd.factorize(group,sort=True)[0]]
        # Five folds stratified by recipe, holding complete experimental batches.
        rng=np.random.default_rng(962102);assign=np.zeros(45,dtype=int)
        for recipe in ['A','B','C']:
            ids=np.unique(labels[0][df.recipe==recipe]);assign[ids]=rng.permutation(15)%5
        outer=[(np.flatnonzero(assign[labels[0]]!=k),np.flatnonzero(assign[labels[0]]==k)) for k in range(5)]
    else:raise ValueError(name)
    for tr,te in outer:
        for v in labels:assert not set(v[tr])&set(v[te])
    assert np.array_equal(np.sort(np.concatenate([te for _,te in outer])),np.arange(len(y)))
    return x,y,labels,names,outer,df.index.to_numpy()

def train_test_arrays(x,y,labels,tr,te,row_ids):
    mean=x[tr].mean(0);sd=x[tr].std(0)
    if np.any(sd<=0):raise ValueError('Constant candidate in training fold')
    out={'y':y[tr],'base':np.ones((len(tr),1)),'x':(x[tr]-mean)/sd,
         'x_test':(x[te]-mean)/sd,'y_test':y[te],
         'train_row_ids':row_ids[tr],'test_row_ids':row_ids[te],
         'train_mean':mean,'train_sd':sd}
    assert np.linalg.matrix_rank(np.column_stack([out['base'],out['x']]))==x.shape[1]+1
    for k,lab in enumerate(labels):
        _,codes=np.unique(lab[tr],return_inverse=True)
        out[f'label{k}']=codes;out[f'z{k}']=np.eye(int(codes.max())+1)[codes]
        out[f'original_train_label{k}']=lab[tr];out[f'original_test_label{k}']=lab[te]
    return out

def stress_data(seed,n,p,groups,scenario):
    rng=np.random.default_rng(seed);ntest=1200
    def design(nn,gg):
        labels=[rng.permutation(np.arange(nn)%g) for g in gg]
        x=rng.normal(size=(nn,p))
        if scenario=='suppressor':x[:,1]=.995*x[:,0]+np.sqrt(1-.995**2)*rng.normal(size=nn)
        return x,labels
    x,labels=design(n,groups);xt,lt=design(ntest,(24,32))
    mu=x.mean(0);sd=x.std(0);x=(x-mu)/sd;xt=(xt-mu)/sd
    beta=np.zeros(p)
    if scenario=='signals':beta[[0,2,4]]=[.8,.5,.3]
    else:beta[[0,1,2]]=[6.,-6.,.45]
    y=x@beta+sum(rng.normal(scale=s,size=g)[v] for g,v,s in zip(groups,labels,[.7,.8]))+rng.normal(size=n)
    yt=xt@beta+sum(rng.normal(scale=s,size=g)[v] for g,v,s in zip((24,32),lt,[.7,.8]))+rng.normal(size=ntest)
    d={'y':y,'base':np.ones((n,1)),'x':x,'x_test':xt,'y_test':yt,'mean_test':xt@beta,'beta':beta,'train_mean':mu,'train_sd':sd}
    for k,v in enumerate(labels):d[f'label{k}']=v;d[f'z{k}']=np.eye(groups[k])[v]
    return d

def prepare():
    from run_comparison_experiments import array_hash
    cases=[]
    for name in ['carrots','cake']:
        x,y,labels,names,splits,rowids=real_data(name)
        for fold,(tr,te) in enumerate([*splits,(np.arange(len(y)),np.array([],dtype=int))]):
            kind='full' if fold==len(splits) else f'fold{fold+1}'
            ident=f'{name}_{kind}';a=train_test_arrays(x,y,labels,tr,te,rowids)
            cases.append({'id':ident,'dataset':name,'kind':kind,'seed':962200+len(cases),
                          'n_train':len(tr),'n_test':len(te),'p':x.shape[1],
                          'features':names,'groups':[z.shape[1] for k,z in a.items() if k.startswith('z')],
                          'data_sha256':array_hash(a)})
            np.savez_compressed(DEST/f'{ident}_data.npz',**a)
    # Two fixed configurations; eight independent datasets per signal scenario.
    for config,(n,p,groups) in enumerate([(600,24,(12,18)),(1200,40,(40,60))]):
        for si,scenario in enumerate(['signals','suppressor']):
            for rep in range(8):
                seed=963000+1000*config+100*si+rep;ident=f'stress_{config}_{scenario}_{rep}'
                a=stress_data(seed,n,p,groups,scenario)
                cases.append({'id':ident,'dataset':'stress','scenario':scenario,'kind':'simulation','seed':seed,
                              'n_train':n,'n_test':1200,'p':p,'groups':list(groups),'data_sha256':array_hash(a)})
                np.savez_compressed(DEST/f'{ident}_data.npz',**a)
    config={'version':'CSDA-stage6-v1','cases':cases,'method_order':'rotating by seed',
            'methods':METHODS,'certified':{'bits':128,'splits':2500,'soft_seconds_per_family':45},
            'worker_timeout_seconds':360,'failures':'Retain all; no partial support is scored as completed.',
            'real_evaluation':'Pooled out-of-fold response MSE and MAE; folds are dependent, no fold-wise iid standard errors.',
            'carrots':'1233 complete outcomes; consumer and product intercepts; published sensory PCs treated as externally supplied covariates, not re-estimated PCA.',
            'cake':'270 outcomes; one random intercept per recipe:replicate batch; five stratified group folds.',
            'selection':'Fixed column dictionary, no interaction hierarchy or group-wise factor test claimed.',
            'enet':{'alphas':[*np.logspace(-3,1,17),1e6],'l1_ratio':[.5,.9,1.],
                    'inner_cv':'3^q crossed cells; every inner-training observation validated once, groups absent in all margins',
                    'loss':'Unweighted mean of fold MSEs, scaling inside each inner training fold; post-selection ML refit'},
            'stress':'32 independent new simulations; descriptive small-replicate extension, not high-dimensional asymptotic validation.'}
    pp=DEST/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==config,'Protocol changed'
    dump(pp,config)
    files=['run_csda_validation.py','comparison_methods.py','certified_path.py','certified_global_path.py','certified_crossed_backend.py','reference_arss_policy.py']
    hh={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    hp=DEST/'code_sha256.json'
    if hp.exists():assert json.loads(hp.read_text())==hh,'Code changed after freeze'
    dump(hp,hh)
    print(json.dumps({'prepared_cases':len(cases),'real_folds':14,'full_real':2,'stress_cases':32}),flush=True)

def enet(model,labels,seed):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    from sklearn.exceptions import ConvergenceWarning
    splits=crossed_splits(labels,seed+700000,3)
    pipeline=Pipeline([('scale',StandardScaler()),('enet',ElasticNet(max_iter=30000,tol=1e-7))])
    grid={'enet__alpha':[*np.logspace(-3,1,17),1e6],'enet__l1_ratio':[.5,.9,1.]}
    # At alpha=10^6 every mixing parameter admits the exact zero solution.
    for tr in [*[t for t,_ in splits],np.arange(model.n)]:
        xx=StandardScaler().fit_transform(model.x[tr]);yy=model.y[tr]-model.y[tr].mean()
        assert np.max(np.abs(xx.T@yy))/len(tr)<=.5e6
    cv=GridSearchCV(pipeline,grid,cv=splits,scoring='neg_mean_squared_error',n_jobs=1,error_score='raise')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);cv.fit(model.x,model.y)
    sc=cv.best_estimator_.named_steps['scale'];en=cv.best_estimator_.named_steps['enet']
    coef=en.coef_/sc.scale_;intercept=float(en.intercept_-sc.mean_@coef)
    return {'status':'complete_local','selected':np.flatnonzero(np.abs(coef)>1e-7).tolist(),
            'pair_additions':None,'best_parameters':{k:float(v) for k,v in cv.best_params_.items()},
            'native_coefficients':coef.tolist(),'native_intercept':intercept,
            'cv_mean_mse':float(-cv.best_score_),'cv_warning_count':sum(issubclass(w.category,ConvergenceWarning) for w in caught),
            'inner_splits':[{'train':tr.tolist(),'test':te.tolist()} for tr,te in splits]}

def run_one(method,a,seed):
    from certified_path import BlockGaussian
    from certified_global_path import global_selection_path,TailArbGaussian
    from certified_crossed_backend import CrossedArbGaussian
    from comparison_methods import LocalFits,local_bic_path,arss_path,coefficients
    zs=[a[k] for k in sorted(a) if k.startswith('z')]
    labels=[a[f'label{k}'] for k in range(len(zs))]
    args=(a['y'],a['base'],a['x'],zs);begun=time.perf_counter()
    if method=='verified_pair':
        result=global_selection_path(*args,bits=128,max_splits=2500,time_budget=45.,
                                    verifier_factory=CrossedArbGaussian if len(zs)==2 else TailArbGaussian)
        result['pair_additions']=sum(len(v)==2 for v in result['additions'])
        model=BlockGaussian(*args);fit=LocalFits(model)
    else:
        model=BlockGaussian(*args);fit=LocalFits(model)
        if method=='local_pair':result=local_bic_path(model,fit,True)
        elif method=='local_single':result=local_bic_path(model,fit,False)
        elif method=='arss_jps':result=arss_path(model,fit)
        elif method=='enet_ml':result=enet(model,labels,seed)
        else:result={'status':'complete_local','selected':[] if method=='null_ml' else list(range(model.p)),'pair_additions':None}
    if result['status'] in ('complete','complete_local'):
        final=fit.fit(result['selected']);b0,beta=coefficients(model,result['selected'],final)
        result.update(estimated_intercept=float(b0[0]),estimated_beta=beta.tolist(),final_refit=final,
                      local_stationarity_failures=fit.stationarity_failures)
    result['pipeline_seconds']=time.perf_counter()-begun
    return result

def worker(ident):
    from run_comparison_experiments import array_hash
    from certified_global_path import replay_global_path,TailArbGaussian
    from certified_crossed_backend import CrossedArbGaussian
    case=next(c for c in json.loads((DEST/'protocol.json').read_text())['cases'] if c['id']==ident)
    with np.load(DEST/f'{ident}_data.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    assert array_hash(a)==case['data_sha256']
    train={k:v for k,v in a.items() if k in ['y','base','x'] or k.startswith('z') or k.startswith('label')}
    methods=METHODS if case['dataset']!='stress' else METHODS[:5]
    shift=case['seed']%len(methods);order=methods[shift:]+methods[:shift];results={}
    for method in order:
        gc.collect();begun=time.perf_counter()
        try:r=run_one(method,train,case['seed'])
        except Exception as exc:r={'status':'method_error','reason':repr(exc),'pipeline_seconds':time.perf_counter()-begun}
        results[method]=r
    predictions={};metrics={}
    for m,r in results.items():
        if r['status'] not in ('complete','complete_local'):metrics[m]=None;continue
        pred=r['estimated_intercept']+a['x_test']@r['estimated_beta'];predictions[m]=pred
        if len(pred):
            metrics[m]={'response_mse':float(np.mean((pred-a['y_test'])**2)),'mae':float(np.mean(np.abs(pred-a['y_test']))),'selected_n':len(r['selected'])}
            if 'beta' in a:
                true=set(np.flatnonzero(a['beta']));selected=set(r['selected']);fp=len(selected-true)
                metrics[m].update(tpr=len(selected&true)/len(true),fdp=fp/max(1,len(selected)),
                                  exact=int(selected==true),mean_mse=float(np.mean((pred-a['mean_test'])**2)))
        if 'native_coefficients' in r:predictions['enet_native']=r['native_intercept']+a['x_test']@r['native_coefficients']
    checked=None;r=results['verified_pair'];zs=[a[k] for k in sorted(a) if k.startswith('z')]
    if r['status']=='complete':
        cr=replay_global_path(a['y'],a['base'],a['x'],zs,r,verifier_factory=CrossedArbGaussian if len(zs)==2 else TailArbGaussian)
        checked={'valid':cr['valid'],'reason':cr['reason'],'decisions':len(cr['checks'])}
        assert cr['valid']
    local=results['local_pair'];agreement=None
    if r['status']=='complete' and local['status']=='complete_local':
        agreement={'support':r['selected']==local['selected'],
                   'path':[(s['phase'],s['winner']) for s in r['stages']]==[(s['phase'],s['winner']) for s in local['stages']]}
    import scipy,sklearn,flint,platform
    record={'case':case,'results':results,'metrics':metrics,'replay':checked,'agreement':agreement,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'flint':flint.__version__,'BLAS_threads':1}}
    np.savez_compressed(DEST/f'{ident}_predictions.npz',**predictions)
    dump(DEST/f'{ident}.json',record)
    print(json.dumps({'id':ident,'status':{m:r['status'] for m,r in results.items()},'seconds':{m:round(r['pipeline_seconds'],2) for m,r in results.items()},'agreement':agreement}),flush=True)

def run(kind):
    config=json.loads((DEST/'protocol.json').read_text())
    hh=json.loads((DEST/'code_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hh.items())
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for case in config['cases']:
        if kind=='real' and case['dataset']=='stress':continue
        if kind=='stress' and case['dataset']!='stress':continue
        out=DEST/f"{case['id']}.json"
        if out.exists():continue
        try:
            p=subprocess.run([sys.executable,__file__,'worker','--id',case['id']],env=env,capture_output=True,text=True,timeout=360,check=True)
            print(p.stdout.strip(),flush=True)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            record={'case':case,'worker_failure':type(e).__name__,'reason':str(e),'stderr':str(getattr(e,'stderr',''))[-3000:]}
            dump(out,record);print(json.dumps(record),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run','worker']);ap.add_argument('--kind',choices=['all','real','stress'],default='all');ap.add_argument('--id');args=ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare()
    elif args.command=='worker':worker(args.id)
    else:run(args.kind)
