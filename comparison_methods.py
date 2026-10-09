"""Stage-five comparators on a shared Gaussian profile likelihood.

ARSS retains the exact archived search function with an explicit fit/score
adapter. EN-CV is a marginal elastic-net screen followed by an LMM refit;
it is not a penalized mixed-model estimator. No local fit is called global.
"""
from __future__ import annotations

import itertools
import time
import types
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from certified_path import BlockGaussian
from certified_global_path import global_selection_path
from certified_crossed_backend import CrossedArbGaussian
import reference_arss_policy


METHODS = ['verified_pair', 'local_pair', 'local_single', 'arss_jps', 'enet_ml']
METHOD_LABELS = {'verified_pair':'Verified BIC + pairs', 'local_pair':'Local BIC + pairs',
                 'local_single':'Local BIC singles', 'arss_jps':'ARSS-JPS policy',
                 'enet_ml':'EN-CV + ML'}


class LocalFits:
    """Fresh multistart local optimization; expand an active finite upper bound.

    No cross-support cache. At most six expansions of [0,4]^q, by factors of
    four. An optimum touching the final cap is an explicit unresolved outcome.
    """
    def __init__(self, model):
        self.model=model;self.requests=0;self.expansions=0
        self.stationarity_failures=0

    def fit(self, support, warm=None):
        self.requests+=1
        lo=np.zeros(self.model.q);hi=np.full(self.model.q,4.)
        for expansion in range(7):
            starts=[lo,hi/2,hi*.1,hi*.9]
            if warm is not None:starts=[np.asarray(warm),*starts]
            fit=self.model.fit(tuple(sorted(support)),lo,hi,starts)
            lam=np.asarray(fit['lambda'])
            if np.any(lam>=.99*hi):
                if expansion==6:raise RuntimeError('Local optimum at expansion cap')
                warm=lam;hi*=4;self.expansions+=1;continue
            _,_,d,gg=self.model.evaluate(lam,[tuple(sorted(support))])
            grad=d[0]+gg
            pg=np.where((lam<=1e-7)&(grad>0),0.,grad)
            fit['stationarity_residual']=float(np.max(np.abs(pg)))
            fit['stationary']=bool(np.max(np.abs(pg))<=1e-4)
            self.stationarity_failures+=int(not fit['stationary'])
            return fit
        raise AssertionError('Unreachable')


def residual_scores(model, fit, selected, active):
    _,gh,_,_=model.covariance(fit['lambda'])
    ix=[*range(model.k),*(model.k+j for j in sorted(selected))]
    ai=[model.k+j for j in active]
    ch=cho_factor(gh[np.ix_(ix,ix)],lower=True)
    beta=cho_solve(ch,gh[ix,-1]);by=gh[np.ix_(ix,ai)]
    u=gh[ai,-1]-by.T@beta
    gram=gh[np.ix_(ai,ai)]-by.T@cho_solve(ch,by)
    gram=(gram+gram.T)/2
    if np.any(np.diag(gram)<=0):raise ValueError('Inadmissible score Gram')
    return u,gram,u*u/np.diag(gram)


def local_bic_path(model, fitter, pairs=True):
    current=();stages=[];additions=[]
    while len(current)<model.p:
        remaining=[j for j in range(model.p) if j not in current]
        accepted=False
        for size in ((1,2) if pairs else (1,)):
            if len(remaining)<size:continue
            base=fitter.fit(current)
            family=[current,*[tuple(sorted((*current,*b))) for b in itertools.combinations(remaining,size)]]
            fits=[base,*[fitter.fit(s,base['lambda']) for s in family[1:]]]
            values=[f['f']+(len(s)-len(current))*np.log(model.n) for f,s in zip(fits,family)]
            winner=int(np.argmin(values))
            stages.append({'phase':'single' if size==1 else 'pair','winner':list(family[winner])})
            if winner:
                additions.append([j for j in family[winner] if j not in current])
                current=family[winner];accepted=True;break
        if not accepted:break
    return {'status':'complete_local','selected':list(current),'additions':additions,'stages':stages,
            'pair_additions':sum(len(a)==2 for a in additions)}


def arss_path(model, fitter):
    """Execute unchanged archived search code with shared profile-ML adapters."""
    fit_trace=[];base_warm={}
    def make_fit(selected, warm=None):
        fit=fitter.fit(selected,warm)
        _,gh,_,_=model.covariance(fit['lambda'])
        ix=[*range(model.k),*(model.k+j for j in sorted(selected))]
        beta=cho_solve(cho_factor(gh[np.ix_(ix,ix)],lower=True),gh[ix,-1])
        scale=float((gh[-1,-1]-gh[ix,-1]@beta)/model.n)
        # The archived flag is supplied by a projected-gradient diagnostic here,
        # not by statsmodels. It does not affect the archived selection rule.
        obj=types.SimpleNamespace(llf=-fit['f']/2,converged=fit['stationary'],
                                  vcomp=np.asarray(fit['lambda'])*scale,profile=fit)
        fit_trace.append({'support':sorted(selected),'f':fit['f'],
                          'stationarity_residual':fit['stationarity_residual']})
        return obj
    def fit_model(_data, selected):
        obj=make_fit(selected)
        base_warm[tuple(sorted(selected))]=obj.profile['lambda']
        return obj,False
    def fit_stat(data, selected, base_ll, candidates):
        fit=make_fit(selected+list(candidates),base_warm[tuple(sorted(selected))])
        lr=max(0.,2*(fit.llf-base_ll))
        return {'fit':fit,'ll':fit.llf,'lr':lr,'p':float(chi2.sf(lr,len(candidates))),
                'conv':fit.converged,'retry':False,'singular':bool(np.min(fit.vcomp)<1e-7)}
    def score_system(_data, base, selected, active, *_unused):
        return residual_scores(model,base.profile,selected,active)
    namespace={'np':np,'itertools':itertools,'chi2':chi2,'MAX_STEPS':model.p+1,
               'K_SINGLE':3,'TAU':.60,'L_PAIR':5,'ALPHA':.01,
               'fit_model':fit_model,'fit_stat':fit_stat,'score_system':score_system}
    run=types.FunctionType(reference_arss_policy.arss_family.__code__,namespace)
    selected,cost=run(None,list(range(model.p)),None,None,None,'short')
    assert cost['full_fits']==fitter.requests
    return {'status':'complete_local','selected':sorted(selected),'selection_order':selected,
            'pair_additions':cost['rescues'],'policy_cost':cost,'fit_trace':fit_trace,
            'alpha':.01,'K':3,'gate':.60,'L':5}


def crossed_cv_splits(labels, seed, folds=3):
    """Validation groups are absent on BOTH margins of each training split.

    For fold k, train uses neither margin in k, validation uses both in k;
    off-diagonal cells touching k are omitted from that split. Validation sets
    are disjoint but do not cover all observations. This is intentional.
    """
    rng=np.random.default_rng(seed)
    assignments=[rng.permutation(np.arange(int(v.max())+1))%folds for v in labels]
    rowfolds=[a[v] for a,v in zip(assignments,labels)]
    splits=[]
    for k in range(folds):
        train=np.flatnonzero((rowfolds[0]!=k)&(rowfolds[1]!=k))
        val=np.flatnonzero((rowfolds[0]==k)&(rowfolds[1]==k))
        if len(train)<10 or len(val)<5:raise ValueError('Insufficient crossed CV fold')
        for v in labels:assert not set(v[train])&set(v[val])
        splits.append((train,val))
    return splits


def enet_screen(model, labels, seed):
    splits=crossed_cv_splits(labels,seed+700000)
    pipeline=Pipeline([('scale',StandardScaler()),
                       ('enet',ElasticNet(max_iter=30000,tol=1e-7,selection='cyclic'))])
    grid={'enet__alpha':np.logspace(-3,0,13),'enet__l1_ratio':[.5,.9,1.]}
    cv=GridSearchCV(pipeline,grid,cv=splits,scoring='neg_mean_squared_error',
                    n_jobs=1,error_score='raise',refit=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        cv.fit(model.x,model.y)
    scale=cv.best_estimator_.named_steps['scale']
    enet=cv.best_estimator_.named_steps['enet']
    coef=enet.coef_/scale.scale_
    intercept=float(enet.intercept_-scale.mean_@coef)
    return {'status':'complete_local','selected':np.flatnonzero(np.abs(coef)>1e-7).tolist(),
            'pair_additions':None,'best_parameters':{k:float(v) for k,v in cv.best_params_.items()},
            'cv_mean_mse':float(-cv.best_score_),'cv_folds':[{'train':t.tolist(),'validation':v.tolist()} for t,v in splits],
            'cv_warning_count':sum(issubclass(w.category,ConvergenceWarning) for w in caught),
            'native_coefficients':coef.tolist(),'native_intercept':intercept,
            'grid_fit_count':len(cv.cv_results_['params'])*len(splits)+1}


def coefficients(model, support, fit):
    _,gh,_,_=model.covariance(fit['lambda'])
    ix=[*range(model.k),*(model.k+j for j in sorted(support))]
    selected_beta=cho_solve(cho_factor(gh[np.ix_(ix,ix)],lower=True),gh[ix,-1])
    full=np.zeros(model.p);full[list(sorted(support))]=selected_beta[model.k:]
    return selected_beta[:model.k],full


def run_method(method, arrays, seed, max_splits=2500, time_budget=30.):
    args=arrays['y'],arrays['base'],arrays['x'],[arrays['z0'],arrays['z1']]
    begun=time.perf_counter()
    if method=='verified_pair':
        result=global_selection_path(*args,bits=128,max_splits=max_splits,time_budget=time_budget,
                                     verifier_factory=CrossedArbGaussian)
        result['pair_additions']=sum(len(a)==2 for a in result['additions'])
        selection_fits=result['model_fits'];requests=None
        model=BlockGaussian(*args);fitter=LocalFits(model)
    else:
        model=BlockGaussian(*args);fitter=LocalFits(model)
        if method=='local_pair':result=local_bic_path(model,fitter,True)
        elif method=='local_single':result=local_bic_path(model,fitter,False)
        elif method=='arss_jps':result=arss_path(model,fitter)
        elif method=='enet_ml':result=enet_screen(model,[arrays['origin'],arrays['destination']],seed)
        else:raise ValueError(method)
        selection_fits=model.fit_calls;requests=fitter.requests
    result['selection_seconds']=time.perf_counter()-begun
    result['selection_fit_calls']=selection_fits
    result['local_fit_requests']=requests
    if result['status'] not in ('complete','complete_local'):
        result['pipeline_seconds']=time.perf_counter()-begun
        return result
    # Same post-selection ML refit for every support. It estimates coefficients
    # for prediction only and does not extend the selection certificate.
    refit_started=time.perf_counter();before=model.fit_calls
    fit=fitter.fit(result['selected'])
    beta0,beta=coefficients(model,result['selected'],fit)
    result.update(refit_seconds=time.perf_counter()-refit_started,
                  refit_fit_calls=model.fit_calls-before,estimated_beta=beta.tolist(),
                  estimated_intercept=float(beta0[0]),final_refit=fit,
                  local_stationarity_failures=fitter.stationarity_failures,
                  local_expansions=fitter.expansions,pipeline_seconds=time.perf_counter()-begun)
    return result


def performance(result, arrays, scenario):
    if result['status'] not in ('complete','complete_local'):return None
    selected=set(result['selected']);truth=set(np.flatnonzero(arrays['beta']))
    tp=len(selected&truth);fp=len(selected-truth)
    pred=result['estimated_intercept']+arrays['x_test']@np.asarray(result['estimated_beta'])
    out={'tpr':tp/len(truth) if truth else None,'fdp':fp/max(1,len(selected)),
         'false_selection_event':int(fp>0),'exact_recovery':int(selected==truth),
         'selected_n':len(selected),'tp':tp,'fp':fp,
         'mean_mse':float(np.mean((pred-arrays['mean_test'])**2)),
         'response_mse':float(np.mean((pred-arrays['y_test'])**2)),
         'pair_recovery':int({0,1}<=selected) if scenario=='suppressor' else None}
    if 'native_coefficients' in result:
        native=result['native_intercept']+arrays['x_test']@result['native_coefficients']
        out.update(native_mean_mse=float(np.mean((native-arrays['mean_test'])**2)),
                   native_response_mse=float(np.mean((native-arrays['y_test'])**2)))
    return out
