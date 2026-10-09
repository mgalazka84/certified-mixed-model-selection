"""BIC forward selection with single additions, pair rescue and Arb witnesses.

The reference is explicitly defined here; it is NOT the published ARSS-JPS
likelihood-ratio protocol. Covariance ratios are restricted to the supplied box.
Float64 proposes a finite branch tree. Arb recomputes every ingredient used to
certify a decision from the original binary input data. No float lower bound is
trusted by the verifier. This is a dense, moderate-p research implementation.
"""
from __future__ import annotations

import hashlib
import itertools
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from flint import arb, arb_mat, ctx


def corners(lo, hi):
    return [np.where(bits, hi, lo) for bits in
            itertools.product([False, True], repeat=len(lo))]


def fingerprint(y, base, x, zs):
    h = hashlib.sha256()
    for a in [y, base, x, *zs]:
        a = np.ascontiguousarray(a, dtype='<f8')
        h.update(str(a.shape).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()


class BlockGaussian:
    """Share low-rank covariance algebra across arbitrary fixed-effect supports.

    Unlike the scalar-only engine, this block implementation forms H.T @ H,
    including a p-by-p predictor Gram matrix. Pair enumeration remains quadratic.
    """
    def __init__(self, y, base, x, zs):
        self.y, self.base, self.x = [np.asarray(a, float) for a in (y, base, x)]
        self.zs = [np.asarray(a, float) for a in zs]
        self.n, self.p = self.x.shape
        self.k = self.base.shape[1]
        if self.y.shape != (self.n,) or self.base.shape[0] != self.n:
            raise ValueError('Inconsistent data dimensions')
        if any(z.shape[0] != self.n for z in self.zs):
            raise ValueError('Inconsistent random designs')
        if not all(np.isfinite(a).all() for a in [self.y,self.base,self.x,*self.zs]):
            raise ValueError('Nonfinite input')
        self.sizes = [z.shape[1] for z in self.zs]
        self.q = len(self.zs)
        self.z = np.column_stack(self.zs)
        self.r = self.z.shape[1]
        self.h = np.column_stack([self.base, self.x, self.y])
        self.a = self.z.T @ self.z
        self.zh = self.z.T @ self.h
        self.hh = self.h.T @ self.h
        self.calls = self.fit_calls = 0
        self._last = None
        self.gcache = {}
        self.data_hash = fingerprint(y,base,x,zs)

    def covariance(self, lam):
        lam = np.asarray(lam, float)
        if lam.shape != (self.q,) or np.any(lam < 0) or not np.isfinite(lam).all():
            raise ValueError('Invalid variance ratios')
        key = tuple(lam)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        sd = np.sqrt(np.repeat(lam,self.sizes))
        ch = cho_factor(np.eye(self.r)+sd[:,None]*self.a*sd[None,:],lower=True)
        g = float(2*np.log(np.diag(ch[0])).sum())
        rhs = sd[:,None]*self.zh
        solved = cho_solve(ch,rhs)
        gh = self.hh-rhs.T @ solved
        eh = self.zh-(self.a*sd[None,:]) @ solved
        aza = self.a-(self.a*sd[None,:]) @ cho_solve(ch,sd[:,None]*self.a)
        gg = []
        start = 0
        for size in self.sizes:
            gg.append(float(np.trace(aza[start:start+size,start:start+size])))
            start += size
        out = (g,gh,eh,np.asarray(gg))
        self._last = (key,out)
        self.gcache[key] = g
        return out

    def logdet(self,lam):
        key = tuple(lam)
        return self.gcache[key] if key in self.gcache else self.covariance(lam)[0]

    def evaluate(self,lam,supports):
        self.calls += 1
        g,gh,eh,gg = self.covariance(lam)
        fs,hs,grads = [],[],[]
        for support in supports:
            ix = [*range(self.k),*(self.k+j for j in support)]
            gram = gh[np.ix_(ix,ix)]
            by = gh[ix,-1]
            beta = cho_solve(cho_factor(gram,lower=True),by)
            s = float(gh[-1,-1]-by @ beta)
            if s <= 1e-12*max(1.,self.hh[-1,-1]):
                raise ValueError('Numerically inadmissible residual')
            residual = eh[:,-1]-eh[:,ix] @ beta
            grad, start = [],0
            for size in self.sizes:
                grad.append(-self.n*float(residual[start:start+size] @
                                        residual[start:start+size])/s)
                start += size
            hv = self.n*np.log(s/self.n)
            fs.append(g+hv); hs.append(hv); grads.append(grad)
        return np.asarray(fs),np.asarray(hs),np.asarray(grads),gg

    def fit(self,support,lo,hi,starts):
        self.fit_calls += 1
        best_f,best_x = np.inf,None
        def fun(lam):
            f,h,d,gg = self.evaluate(lam,[support])
            return f[0], d[0]+gg
        for start in starts:
            start = np.clip(start,lo,hi)
            f0 = fun(start)[0]
            if f0 < best_f:
                best_f,best_x = float(f0),start.copy()
            opt = minimize(fun,start,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),
                           options={'ftol':1e-12,'gtol':1e-7,'maxiter':350,'maxls':40})
            if np.isfinite(opt.fun) and opt.fun < best_f:
                best_f,best_x = float(opt.fun),opt.x.copy()
        return {'f':best_f,'lambda':best_x.tolist()}


def submat(a, rows, cols):
    return arb_mat([[a[i,j] for j in cols] for i in rows])


class ArbGaussian:
    """Verified point objective/gradient enclosures for exact binary input arrays.

    All data cross products, solves and logarithms are recomputed in Arb. The
    precision is a fixed contract for this verifier; inability to prove positivity
    or invertibility raises an exception, never a certificate.
    """
    def __init__(self,model,bits=128):
        if bits < 64:
            raise ValueError('Use at least 64 bits')
        ctx.prec = bits
        self.bits,self.n,self.k = bits,model.n,model.k
        self.p,self.q,self.r = model.p,model.q,model.r
        self.sizes,self.data_hash = model.sizes,model.data_hash
        # arb(float) imports the exact binary value, including signed zeros.
        H = arb_mat(model.h.tolist())
        Z = arb_mat(model.z.tolist())
        self.hh,self.zh,self.a = H.transpose()*H,Z.transpose()*H,Z.transpose()*Z
        self.logn = arb(self.n).log()
        self.covcache,self.evalcache = {},{}

    def covariance(self,lam):
        ctx.prec = self.bits
        key = tuple(lam)
        if key in self.covcache:
            return self.covcache[key]
        if len(key) != self.q or any(not np.isfinite(v) or v < 0 for v in key):
            raise ValueError('Invalid witness ratios')
        sd = [arb(float(v)).sqrt() for v in np.repeat(lam,self.sizes)]
        b = arb_mat([[arb(int(i==j))+sd[i]*self.a[i,j]*sd[j]
                      for j in range(self.r)] for i in range(self.r)])
        det = b.det()
        if not det > 0:
            raise ValueError('Unverified positive covariance determinant')
        g = det.log()
        rhs = arb_mat([[sd[i]*self.zh[i,j] for j in range(self.hh.ncols())]
                       for i in range(self.r)])
        sol = b.solve(rhs)
        gh = self.hh-rhs.transpose()*sol
        az = arb_mat([[self.a[i,j]*sd[j] for j in range(self.r)] for i in range(self.r)])
        eh = self.zh-az*sol
        self.covcache[key] = (g,gh,eh)
        return g,gh,eh

    def profile(self,lam,support):
        ctx.prec = self.bits
        key = (tuple(lam),tuple(support))
        if key in self.evalcache:
            return self.evalcache[key]
        g,gh,eh = self.covariance(lam)
        ix = [*range(self.k),*(self.k+j for j in support)]
        last = self.hh.ncols()-1
        by = submat(gh,ix,[last])
        beta = submat(gh,ix,ix).solve(by)
        s = gh[last,last]-(by.transpose()*beta)[0,0]
        if not s > 0:
            raise ValueError('Unverified positive residual')
        residual = submat(eh,range(self.r),[last])-submat(eh,range(self.r),ix)*beta
        grad,start = [],0
        for size in self.sizes:
            grad.append(-arb(self.n)*sum((residual[i,0]**2 for i in range(start,start+size)),arb(0))/s)
            start += size
        h = arb(self.n)*(s/self.n).log()
        out = (g+h,h,grad)
        if not all(a.is_finite() for a in [out[0],h,*grad]):
            raise ValueError('Nonfinite ball enclosure')
        self.evalcache[key] = out
        return out


def verify_decision(model,witness,verifier=None):
    """Recheck the partition, all bounds and the strict decision independently.

    Serialized float bound values, optimizer success flags and claimed margins
    have no role in the decision. Every finite leaf must be covered.
    """
    started = time.perf_counter()
    try:
        if witness['data_sha256'] != model.data_hash:
            raise ValueError('Data hash mismatch')
        vg = verifier or ArbGaussian(model,witness.get('precision_bits',128))
        if vg.data_hash != model.data_hash:
            raise ValueError('Verifier data mismatch')
        ctx.prec = vg.bits
        fam = [tuple(s) for s in witness['family']]
        sizes = witness['added_sizes']
        if len(fam) != len(sizes) or not fam:
            raise ValueError('Invalid family')
        for s in fam:
            if len(s) != len(set(s)) or any(j < 0 or j >= model.p for j in s):
                raise ValueError('Invalid support')
        if sizes != [len(s)-len(fam[0]) for s in fam]:
            raise ValueError('Invalid BIC penalties')
        lo,hi = np.asarray(witness['lo']),np.asarray(witness['hi'])
        if lo.shape != (model.q,) or np.any(lo < 0) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi<=lo):
            raise ValueError('Invalid root box')
        nodes = witness['nodes']
        seen,leaves = set(),[]
        def visit(idx,a,b,parent_lower=None):
            if idx in seen or idx < 0 or idx >= len(nodes):
                raise ValueError('Invalid tree topology')
            seen.add(idx)
            node = nodes[idx]
            if node['lo'] != list(a) or node['hi'] != list(b):
                raise ValueError('Invalid partition coordinates')
            vv = corners(a,b)
            # A structured verifier may evaluate a determinant without solving
            # all predictor right-hand sides at a vertex. The default is unchanged.
            gg = [vg.logdet(v) if hasattr(vg,'logdet') else vg.covariance(v)[0] for v in vv]
            lower = list(parent_lower) if parent_lower is not None else [None]*len(fam)
            anchors = node['anchors']
            if not anchors:
                raise ValueError('Missing anchor')
            for anchor in anchors:
                for j,s in enumerate(fam):
                    _,h,grad = vg.profile(anchor,s)
                    terms = []
                    for v,g in zip(vv,gg):
                        t = g+h+sizes[j]*vg.logn
                        for k in range(model.q):
                            t += grad[k]*(arb(float(v[k]))-arb(float(anchor[k])))
                        if not t.is_finite():
                            raise ValueError('Invalid tangent enclosure')
                        terms.append(t.lower())
                    lb = min(terms)
                    lower[j] = lb if lower[j] is None else max(lower[j],lb)
            if 'children' not in node:
                leaves.append(lower)
                return
            dim,mid = node['split_dim'],node['split_at']
            if dim not in range(model.q) or not a[dim] < mid < b[dim]:
                raise ValueError('Invalid bisection')
            if len(node['children']) != 2:
                raise ValueError('Invalid child count')
            left_b,right_a = b.copy(),a.copy()
            left_b[dim]=mid; right_a[dim]=mid
            visit(node['children'][0],a,left_b,lower)
            visit(node['children'][1],right_a,b,lower)
        visit(0,lo,hi)
        if len(seen) != len(nodes):
            raise ValueError('Unreachable witness nodes')
        lower = [min(row[j] for row in leaves) for j in range(len(fam))]
        winner = witness['winner_index']
        if winner not in range(len(fam)):
            raise ValueError('Invalid winner')
        point = np.asarray(witness['winner_lambda'])
        if point.shape != lo.shape or np.any(point < lo) or np.any(point > hi):
            raise ValueError('Infeasible incumbent')
        upper = (vg.profile(point,fam[winner])[0]+sizes[winner]*vg.logn).upper()
        gaps = [lower[j]-upper for j in range(len(fam)) if j != winner]
        margin = min(gaps) if gaps else arb(1)
        valid = all(g > 0 for g in gaps)
        return {'valid':valid,'precision_bits':vg.bits,'nodes_checked':len(nodes),
                'leaves_checked':len(leaves),'verified_margin':str(margin),
                'margin_for_display':float(margin.lower()),
                'incumbent_upper':str(upper),'seconds':time.perf_counter()-started,
                'reason':'strict separation' if valid else 'overlapping objective bounds'}
    except (ValueError,ZeroDivisionError,RuntimeError,IndexError,KeyError,TypeError) as exc:
        return {'valid':False,'reason':str(exc),'seconds':time.perf_counter()-started}


def family_search(model,current,blocks,lo,hi,verifier=None,max_splits=2000,time_budget=30.,
                  initial=None,decision_verifier=None,witness_extra=None,geometry='linear'):
    """Choose among current model and one specified family of additions, using BIC."""
    begun = time.perf_counter()
    calls0,fits0 = model.calls,model.fit_calls
    lo,hi = np.asarray(lo,float),np.asarray(hi,float)
    if lo.shape != (model.q,) or np.any(lo<0) or np.any(hi<=lo):
        raise ValueError('Invalid ratio box')
    current = tuple(sorted(current))
    family = [current,*[tuple(sorted((*current,*b))) for b in blocks]]
    sizes = [0,*[len(b) for b in blocks]]
    penalty = np.asarray(sizes)*np.log(model.n)
    if geometry not in ('linear','log1p'):
        raise ValueError('Unknown subdivision geometry')
    def midpoint(a,b):
        return (a+b)/2 if geometry=='linear' else np.expm1((np.log1p(a)+np.log1p(b))/2)
    starts = [lo,(lo+hi)/2,lo+.1*(hi-lo),lo+.9*(hi-lo)]
    if initial is None:
        basefit = model.fit(current,lo,hi,starts)
        lam0 = np.asarray(basefit['lambda'])
        values = model.evaluate(lam0,family)[0]+penalty
        winner = int(np.argmin(values))
        upper,point = float(values[winner]),lam0.copy()
        if winner:
            fit = model.fit(family[winner],lo,hi,[lam0,*starts])
            upper,point = fit['f']+penalty[winner],np.asarray(fit['lambda'])
    else:
        lam0 = np.asarray(initial['base_lambda'],float)
        winner = initial['winner_index']
        point = np.asarray(initial['winner_lambda'],float)
        if np.any(lam0<lo) or np.any(lam0>hi) or np.any(point<lo) or np.any(point>hi):
            raise ValueError('Infeasible initial proposal')
        upper = float(model.evaluate(point,[family[winner]])[0][0]+penalty[winner])
    nodes,lower_by_node = [],[]
    def node(a,b,parent=None,extra=None):
        nonlocal winner,upper,point
        center = midpoint(a,b)
        anchors = [center] if extra is None else [center,extra]
        vv = np.asarray(corners(a,b))
        gg = np.array([model.logdet(v) for v in vv])
        lower = np.full(len(family),-np.inf) if parent is None else parent.copy()
        for c in anchors:
            f,h,grad,_ = model.evaluate(c,family)
            lb = np.min(gg[None,:]+h[:,None]+grad @ (vv-c).T,axis=1)+penalty
            lb -= 2e-8*(1+np.maximum(np.abs(lb),np.abs(f+penalty)))
            lower = np.maximum(lower,lb)
            j = int(np.argmin(f+penalty))
            if f[j]+penalty[j] < upper-1e-7:
                fit = model.fit(family[j],lo,hi,[lam0,*starts,c])
                if fit['f']+penalty[j] < upper:
                    winner,upper,point = j,fit['f']+penalty[j],np.asarray(fit['lambda'])
        idx = len(nodes)
        nodes.append({'lo':a.tolist(),'hi':b.tolist(),'anchors':[c.tolist() for c in anchors]})
        lower_by_node.append(lower)
        return idx
    active = [node(lo.copy(),hi.copy(),extra=lam0)]
    splits,status,verification = 0,'unresolved_budget',None
    while True:
        lower_matrix = np.asarray([lower_by_node[i] for i in active])
        lower = np.min(lower_matrix,axis=0)
        rivals = [j for j in range(len(family)) if j != winner]
        if not rivals or np.min(lower[rivals]-upper) > 1e-6*(1+abs(upper)):
            witness = {'data_sha256':model.data_hash,'lo':lo.tolist(),'hi':hi.tolist(),
                       'family':[list(s) for s in family],'added_sizes':sizes,
                       'nodes':nodes,'winner_index':winner,'winner_lambda':point.tolist(),
                       'precision_bits':verifier.bits if verifier else 128,
                       **(witness_extra or {})}
            verification = (decision_verifier or verify_decision)(model,witness,verifier)
            status = 'verified' if verification['valid'] else 'unresolved_numeric'
            break
        if splits >= max_splits or time.perf_counter()-begun > time_budget:
            break
        row = int(np.argmin(np.min(lower_matrix[:,rivals],axis=1)))
        idx = active.pop(row)
        nd = nodes[idx]
        a,b = np.asarray(nd['lo']),np.asarray(nd['hi'])
        widths=(b-a)/(hi-lo) if geometry=='linear' else np.log1p(b)-np.log1p(a)
        dim = int(np.argmax(widths))
        mid = float(midpoint(a,b)[dim])
        if not a[dim] < mid < b[dim]:
            status = 'unresolved_numeric'
            break
        b0,a1 = b.copy(),a.copy(); b0[dim]=mid; a1[dim]=mid
        children = [node(a.copy(),b0,lower_by_node[idx]),node(a1,b.copy(),lower_by_node[idx])]
        nd.update({'children':children,'split_dim':dim,'split_at':mid})
        active.extend(children)
        splits += 1
    witness = {'data_sha256':model.data_hash,'lo':lo.tolist(),'hi':hi.tolist(),
               'family':[list(s) for s in family],'added_sizes':sizes,'nodes':nodes,
               'winner_index':winner,'winner_lambda':point.tolist(),
               'precision_bits':verifier.bits if verifier else 128,
               **(witness_extra or {})}
    return {'status':status,'current':list(current),'winner':list(family[winner]),
            'winner_index':winner,'splits':splits,'model_fits':model.fit_calls-fits0,
            'evaluation_calls':model.calls-calls0,'seconds':time.perf_counter()-begun,
            'verification':verification,'witness':witness}


def selection_path(y,base,x,zs,lo=None,hi=None,bits=128,max_splits=2000,time_budget=30.):
    """Prefer an improving best single; at single stopping, try all remaining pairs.

    At pair stopping the current support is locally optimal with respect to all
    one- and two-variable additions under the specified BIC and covariance box.
    There is no claim of best-subset optimality, FWER control, or selective inference.
    """
    begun = time.perf_counter()
    model = BlockGaussian(y,base,x,zs)
    verifier = ArbGaussian(model,bits)
    lo = np.zeros(model.q) if lo is None else np.asarray(lo,float)
    hi = np.full(model.q,4.) if hi is None else np.asarray(hi,float)
    current,stages,additions = (),[],[]
    status = 'complete'
    while len(current) < model.p:
        remaining = [j for j in range(model.p) if j not in current]
        blocks = [(j,) for j in remaining]
        step = family_search(model,current,blocks,lo,hi,verifier,max_splits,time_budget)
        step['phase']='single'; stages.append(step)
        if step['status'] != 'verified':
            status=step['status']; break
        if step['winner_index']:
            new = tuple(step['winner'])
            additions.append([j for j in new if j not in current]); current=new
            continue
        if len(remaining) < 2:
            break
        blocks = list(itertools.combinations(remaining,2))
        step = family_search(model,current,blocks,lo,hi,verifier,max_splits,time_budget)
        step['phase']='pair'; stages.append(step)
        if step['status'] != 'verified':
            status=step['status']; break
        if not step['winner_index']:
            break
        new = tuple(step['winner'])
        additions.append([j for j in new if j not in current]); current=new
    return {'status':status,'selected':list(current),'additions':additions,
            'stages':stages,'seconds':time.perf_counter()-begun,'bits':bits,
            'lo':lo.tolist(),'hi':hi.tolist(),'data_sha256':model.data_hash,
            'model_fits':model.fit_calls,'evaluation_calls':model.calls}


def exhaustive_path(y,base,x,zs,lo=None,hi=None):
    """Same selection policy, all-model multistart ML comparator, not global oracle."""
    begun=time.perf_counter()
    model=BlockGaussian(y,base,x,zs)
    lo=np.zeros(model.q) if lo is None else np.asarray(lo,float)
    hi=np.full(model.q,4.) if hi is None else np.asarray(hi,float)
    starts=[lo,(lo+hi)/2,lo+.1*(hi-lo),lo+.9*(hi-lo)]
    current,additions,stages=(),[],[]
    while len(current)<model.p:
        remaining=[j for j in range(model.p) if j not in current]
        accepted=False
        for size in (1,2):
            if len(remaining)<size:
                continue
            basefit=model.fit(current,lo,hi,starts)
            family=[current,*[tuple(sorted((*current,*b))) for b in itertools.combinations(remaining,size)]]
            fits=[basefit,*[model.fit(s,lo,hi,[basefit['lambda'],*starts]) for s in family[1:]]]
            vals=[f['f']+(len(s)-len(current))*np.log(model.n) for f,s in zip(fits,family)]
            winner=int(np.argmin(vals))
            stages.append({'phase':'single' if size==1 else 'pair','winner':list(family[winner]),'gap':float(np.sort(vals)[1]-np.sort(vals)[0])})
            if winner:
                additions.append([j for j in family[winner] if j not in current])
                current=family[winner];accepted=True;break
        if not accepted:
            break
    return {'selected':list(current),'additions':additions,'stages':stages,
            'seconds':time.perf_counter()-begun,'model_fits':model.fit_calls}


def replay_path(y,base,x,zs,result):
    """Validate family completeness, phase order, data, partition and all decisions."""
    model=BlockGaussian(y,base,x,zs)
    verifier=ArbGaussian(model,result['bits'])
    current,phase,additions=(), 'single',[]
    checks=[]
    stopped=(model.p==0)
    for stage in result['stages']:
        if stopped or stage['phase'] != phase:
            return {'valid':False,'reason':'Invalid phase sequence'}
        size=1 if phase=='single' else 2
        remaining=[j for j in range(model.p) if j not in current]
        family=[current,*[tuple(sorted((*current,*b))) for b in itertools.combinations(remaining,size)]]
        w=stage['witness']
        if w['family'] != [list(s) for s in family] or w['lo'] != result['lo'] or w['hi'] != result['hi']:
            return {'valid':False,'reason':'Incomplete family or changed ratio box'}
        checked=verify_decision(model,w,verifier);checks.append(checked)
        if not checked['valid']:
            return {'valid':False,'reason':'Decision not verified','checks':checks}
        winner=w['winner_index']
        if stage['winner'] != list(family[winner]) or stage['current'] != list(current):
            return {'valid':False,'reason':'Inconsistent reported decision'}
        if winner:
            additions.append([j for j in family[winner] if j not in current])
            current=family[winner];phase='single'
            if len(current)==model.p:
                stopped=True
        elif phase=='pair' or len(remaining)<2:
            stopped=True
        else:
            phase='pair'
    valid=(stopped and result['status']=='complete' and list(current)==result['selected'] and additions==result['additions'])
    return {'valid':valid,'checks':checks,'selected':list(current),'reason':'complete path verified' if valid else 'Incomplete or inconsistent path'}
