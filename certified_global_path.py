"""Replayable single/pair BIC decisions over all nonnegative variance ratios.

An exact rational rank calculation identifies a basis of [base, Z]. Arb then
encloses the projection residual for every candidate support. Positive residual
floors and determinant monotonicity exclude the complement of a finite box.
The existing box verifier covers its interior. Failure to prove a positive floor
does not imply nonexistence of ML; it means this sufficient certificate is absent.
"""
from __future__ import annotations

import itertools
import time
import numpy as np
from flint import arb, arb_mat, fmpq, fmpq_mat, ctx
from certified_path import (BlockGaussian, ArbGaussian, submat, family_search,
                            verify_decision)


def exact_column_basis(a):
    """Pivot columns of exact binary input; no floating rank threshold is used."""
    rows = [[fmpq(*float(v).as_integer_ratio()) for v in row] for row in a]
    reduced, rank = fmpq_mat(rows).rref()
    return [next(j for j in range(reduced.ncols()) if reduced[i,j] != 0)
            for i in range(rank)]


class TailArbGaussian(ArbGaussian):
    def __init__(self,model,bits=128):
        super().__init__(model,bits)
        self.model = model
        self.floorcache = {}
        self._floor_ready = False

    def _prepare_floor(self):
        if self._floor_ready:
            return
        # Random intercepts often have structural dependencies with the base.
        # Exact pivoting proves that discarded columns are spanned by this basis.
        w0 = np.column_stack([self.model.base,self.model.z])
        pivots = exact_column_basis(w0)
        self.w0_basis = w0[:,pivots]
        w = np.column_stack([self.w0_basis,self.model.x,self.model.y])
        aw = arb_mat(w.tolist())
        self.floorgram = aw.transpose()*aw
        self.floor_basis_rank = len(pivots)
        rest = list(range(len(pivots),w.shape[1]))
        fixed = list(range(len(pivots)))
        if fixed:
            by = submat(self.floorgram,fixed,rest)
            solved = submat(self.floorgram,fixed,fixed).solve(by)
            self.projected = submat(self.floorgram,rest,rest)-by.transpose()*solved
        else:
            self.projected = self.floorgram
        self._floor_ready = True

    def residual_floor(self,support):
        ctx.prec = self.bits
        support = tuple(support)
        if support in self.floorcache:
            return self.floorcache[support]
        self._prepare_floor()
        try:
            if support:
                by = submat(self.projected,support,[self.p])
                beta = submat(self.projected,support,support).solve(by)
                floor = self.projected[self.p,self.p]-(by.transpose()*beta)[0,0]
            else:
                floor = self.projected[self.p,self.p]
            if not floor > 0 or not floor.is_finite():
                raise ValueError('Positive projection residual not proved')
        except (ValueError,ZeroDivisionError,RuntimeError):
            # A candidate can lie in span(base,Z) without invalidating the floor.
            # Recompute its full exact spanning basis instead of dropping it on
            # the basis of a numerical singular-value threshold.
            w = np.column_stack([self.w0_basis,self.model.x[:,support]])
            pivots = exact_column_basis(w)
            full = [*pivots, w.shape[1]]
            aw = arb_mat(np.column_stack([w,self.model.y]).tolist())
            gram = aw.transpose()*aw
            by = submat(gram,pivots,[full[-1]])
            floor = gram[full[-1],full[-1]]
            if pivots:
                floor -= (by.transpose()*submat(gram,pivots,pivots).solve(by))[0,0]
            if not floor > 0 or not floor.is_finite():
                raise ValueError('Positive projection residual not proved')
        self.floorcache[support] = floor
        return floor

    def component_logdet(self,k,cutoff):
        """log det(I + cutoff Z_k' Z_k), evaluated without square roots."""
        ctx.prec = self.bits
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('Invalid tail cutoff')
        start = sum(self.sizes[:k]); size = self.sizes[k]
        a = submat(self.a,range(start,start+size),range(start,start+size))
        trace = sum((a[i,i] for i in range(size)),arb(0))
        if not trace > 0:
            raise ValueError('Nonzero covariance component not proved')
        mat = arb_mat([[arb(int(i==j))+arb(float(cutoff))*a[i,j]
                        for j in range(size)] for i in range(size)])
        det = mat.det()
        if not det > 0:
            raise ValueError('Positive tail determinant not proved')
        return det.log()

    def tail_constants(self,family,sizes):
        return [arb(self.n)*(self.residual_floor(s)/self.n).log()+d*self.logn
                for s,d in zip(family,sizes)]

    def automatic_box(self,family,sizes,winner,point,margin=1.,max_doublings=50):
        """Powers of two are proposals; only Arb inequalities authorize a box."""
        constants = self.tail_constants(family,sizes)
        target = (self.profile(point,family[winner])[0]+sizes[winner]*self.logn).upper()+arb(margin)
        common = min(c.lower() for c in constants)
        hi=[]
        for k in range(self.q):
            cutoff=4.
            for _ in range(max_doublings+1):
                bound=self.component_logdet(k,cutoff)+common
                if bound.lower() > target:
                    hi.append(cutoff)
                    break
                cutoff *= 2
            else:
                raise ValueError('Tail expansion budget exhausted')
        return np.zeros(self.q),np.asarray(hi)


def verify_global_decision(model,witness,verifier=None):
    """Verify box coverage and all unbounded tails from original arrays."""
    started=time.perf_counter()
    try:
        if witness.get('domain') != 'nonnegative_orthant' or witness.get('tail_method') != 'projection_floor_v1':
            raise ValueError('Missing global-domain certificate')
        if witness['lo'] != [0.]*model.q:
            raise ValueError('Global box must start at zero')
        vg=verifier or TailArbGaussian(model,witness.get('precision_bits',128))
        if vg.data_hash != model.data_hash or witness['data_sha256'] != model.data_hash:
            raise ValueError('Data hash mismatch')
        family=[tuple(s) for s in witness['family']]
        sizes=witness['added_sizes']; winner=witness['winner_index']
        if len(family) != len(sizes) or sizes != [len(s)-len(family[0]) for s in family]:
            raise ValueError('Invalid family penalties')
        box_check=verify_decision(model,witness,vg)
        if not box_check['valid']:
            return {**box_check,'domain':'nonnegative_orthant','seconds':time.perf_counter()-started}
        constants=vg.tail_constants(family,sizes)
        dets=[vg.component_logdet(k,u) for k,u in enumerate(witness['hi'])]
        upper=(vg.profile(witness['winner_lambda'],family[winner])[0]+sizes[winner]*vg.logn).upper()
        bounds=[min((g+c).lower() for g in dets) for c in constants]
        gaps=[bounds[j]-upper for j in range(len(family)) if j != winner]
        margin=min(gaps) if gaps else arb(1)
        valid=all(g>0 for g in gaps)
        return {**box_check,'valid':valid,'domain':'nonnegative_orthant',
                'tail_margin':str(margin),'tail_margin_for_display':float(margin.lower()),
                'tail_families_checked':len(family),'tail_regions_checked':model.q,
                'minimum_residual_floor':str(min(vg.residual_floor(s).lower() for s in family)),
                'projection_basis_rank':vg.floor_basis_rank,
                'reason':'strict separation on nonnegative orthant' if valid else 'tail bounds overlap incumbent',
                'seconds':time.perf_counter()-started}
    except (ValueError,ZeroDivisionError,RuntimeError,IndexError,KeyError,TypeError) as exc:
        return {'valid':False,'reason':str(exc),'seconds':time.perf_counter()-started}


def global_family_search(model,current,blocks,verifier,max_splits=2500,time_budget=60.,geometry='log1p'):
    begun=time.perf_counter(); calls0,fits0=model.calls,model.fit_calls
    current=tuple(sorted(current))
    family=[current,*[tuple(sorted((*current,*b))) for b in blocks]]
    sizes=[0,*[len(b) for b in blocks]]
    # An adaptively enlarged pilot box only proposes a feasible upper bound.
    # Boundary checks guide this search; only the later tail proof sets a domain.
    lo=np.zeros(model.q); pilot_hi=np.full(model.q,4.)
    starts=[lo,pilot_hi/2,pilot_hi*.1,pilot_hi*.9]
    try:
        basefit=model.fit(current,lo,pilot_hi,starts)
        for _ in range(6):
            if not np.any(np.asarray(basefit['lambda'])>=.99*pilot_hi):
                break
            pilot_hi *= 4
            starts=[lo,pilot_hi/2,pilot_hi*.1,pilot_hi*.9]
            basefit=model.fit(current,lo,pilot_hi,[basefit['lambda'],*starts])
        lam0=np.asarray(basefit['lambda'])
        values=model.evaluate(lam0,family)[0]+np.asarray(sizes)*np.log(model.n)
        winner=int(np.argmin(values)); point=lam0
        if winner:
            fit=model.fit(family[winner],lo,pilot_hi,[lam0,*starts])
            point=np.asarray(fit['lambda'])
        lo,hi=verifier.automatic_box(family,sizes,winner,point)
    except (ValueError,ZeroDivisionError,RuntimeError,np.linalg.LinAlgError) as exc:
        return {'status':'unresolved_tail','reason':str(exc),'current':list(current),
                'winner':list(current),'winner_index':0,'splits':0,'verification':None,
                'witness':None,'seconds':time.perf_counter()-begun,
                'model_fits':model.fit_calls-fits0,'evaluation_calls':model.calls-calls0}
    construction_seconds=time.perf_counter()-begun
    result=family_search(model,current,blocks,lo,hi,verifier,max_splits,
                         max(0.,time_budget-construction_seconds),
                         initial={'base_lambda':np.clip(lam0,lo,hi),'winner_index':winner,'winner_lambda':point},
                         decision_verifier=verify_global_decision,
                         witness_extra={'domain':'nonnegative_orthant','tail_method':'projection_floor_v1',
                                        'search_geometry':geometry},geometry=geometry)
    result.update(seconds=time.perf_counter()-begun,model_fits=model.fit_calls-fits0,
                  evaluation_calls=model.calls-calls0,tail_construction_seconds=construction_seconds)
    return result


def global_selection_path(y,base,x,zs,bits=128,max_splits=2500,time_budget=60.,verifier_factory=None):
    """The same single/pair BIC reference, now over lambda in [0,infinity)^q."""
    begun=time.perf_counter();model=BlockGaussian(y,base,x,zs)
    verifier=(verifier_factory or TailArbGaussian)(model,bits)
    current,stages,additions=(),[],[];status='complete'
    while len(current)<model.p:
        remaining=[j for j in range(model.p) if j not in current]
        accepted=False
        for size in (1,2):
            if len(remaining)<size:
                continue
            blocks=list(itertools.combinations(remaining,size))
            step=global_family_search(model,current,blocks,verifier,max_splits,time_budget)
            step['phase']='single' if size==1 else 'pair';stages.append(step)
            if step['status']!='verified':
                status=step['status'];break
            if step['winner_index']:
                new=tuple(step['winner']);additions.append([j for j in new if j not in current])
                current=new;accepted=True;break
        if status!='complete' or not accepted:
            break
    result={'status':status,'selected':list(current),'additions':additions,'stages':stages,
            'seconds':time.perf_counter()-begun,'bits':bits,'domain':'nonnegative_orthant',
            'data_sha256':model.data_hash,'model_fits':model.fit_calls,'evaluation_calls':model.calls}
    if hasattr(verifier,'statistics'):
        result['verifier_statistics']=verifier.statistics()
    return result


def replay_global_path(y,base,x,zs,result,verifier_factory=None):
    """Check full families and phase order; never optimize or trust saved floors."""
    try:
        model=BlockGaussian(y,base,x,zs);verifier=(verifier_factory or TailArbGaussian)(model,result['bits'])
        if result.get('domain')!='nonnegative_orthant' or result['data_sha256']!=model.data_hash:
            raise ValueError('Global path domain or data mismatch')
        current,phase,additions=(),'single',[];checks=[];stopped=(model.p==0)
        for stage in result['stages']:
            if stopped or stage['phase']!=phase or stage['status']!='verified':
                raise ValueError('Invalid phase sequence or unchecked stage')
            remaining=[j for j in range(model.p) if j not in current];size=1 if phase=='single' else 2
            family=[current,*[tuple(sorted((*current,*b))) for b in itertools.combinations(remaining,size)]]
            w=stage['witness']
            if w['family'] != [list(s) for s in family]:
                raise ValueError('Incomplete candidate family')
            checked=verify_global_decision(model,w,verifier);checks.append(checked)
            if not checked['valid']:
                return {'valid':False,'reason':'Global decision not verified','checks':checks}
            winner=w['winner_index']
            if stage['winner']!=list(family[winner]) or stage['current']!=list(current) or stage['winner_index']!=winner:
                raise ValueError('Inconsistent reported decision')
            if winner:
                additions.append([j for j in family[winner] if j not in current])
                current=family[winner];phase='single';stopped=(len(current)==model.p)
            elif phase=='pair' or len(remaining)<2:
                stopped=True
            else:
                phase='pair'
        valid=(stopped and result['status']=='complete' and list(current)==result['selected'] and additions==result['additions'])
        return {'valid':valid,'checks':checks,'selected':list(current),
                'reason':'complete unbounded-domain path verified' if valid else 'Incomplete or inconsistent path'}
    except (ValueError,ZeroDivisionError,RuntimeError,IndexError,KeyError,TypeError) as exc:
        return {'valid':False,'reason':str(exc)}
