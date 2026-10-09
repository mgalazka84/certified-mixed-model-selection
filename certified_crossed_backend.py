"""Structured Arb backend for exactly two crossed random-intercept designs.

The float64 search and witness format are unchanged. This backend uses a Schur
complement of dimension min(r1,r2), determinant-only vertex evaluations, shared
gradient Gram matrices, and bounded caches for large covariance/profile objects.
It still uses dense arrays, a dense Schur complement, and exact rank reduction
for tail floors. It is not a general sparse mixed-model implementation.
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
from flint import arb,arb_mat,ctx
from certified_path import submat
from certified_global_path import TailArbGaussian


class BoundedCache(OrderedDict):
    def __init__(self,capacity):
        super().__init__();self.capacity=capacity;self.peak=0;self.evictions=0

    def __getitem__(self,key):
        value=super().__getitem__(key);self.move_to_end(key);return value

    def __setitem__(self,key,value):
        if self.capacity==0:
            return
        super().__setitem__(key,value);self.move_to_end(key)
        if len(self)>self.capacity:
            self.popitem(last=False);self.evictions+=1
        self.peak=max(self.peak,len(self))


def check_crossed_intercepts(model):
    if model.q!=2:
        raise ValueError('Structured backend requires exactly two random components')
    for z in model.zs:
        if z.ndim!=2 or not np.all((z==0.)|(z==1.)) or not np.all(np.count_nonzero(z,axis=1)==1):
            raise ValueError('Structured backend requires exactly one binary group membership per row and component')


class CrossedArbGaussian(TailArbGaussian):
    def __init__(self,model,bits=128,covariance_cache_size=32,profile_cache_size=256):
        check_crossed_intercepts(model)
        super().__init__(model,bits)
        self.large=0 if self.sizes[0]>=self.sizes[1] else 1
        self.small=1-self.large
        offsets=[0,self.sizes[0]]
        self.rows_large=list(range(offsets[self.large],offsets[self.large]+self.sizes[self.large]))
        self.rows_small=list(range(offsets[self.small],offsets[self.small]+self.sizes[self.small]))
        self.count_large=[self.a[i,i] for i in self.rows_large]
        self.count_small=[self.a[i,i] for i in self.rows_small]
        self.cross=submat(self.a,self.rows_large,self.rows_small)
        cols=list(range(self.hh.ncols()))
        self.zhl=submat(self.zh,self.rows_large,cols)
        self.zhs=submat(self.zh,self.rows_small,cols)
        self.covcache=BoundedCache(covariance_cache_size)
        self.evalcache=BoundedCache(profile_cache_size)
        self.gradcache=BoundedCache(covariance_cache_size)
        self.detcache={}
        self.systems_built=self.predictor_solves=self.gram_constructions=0

    def _system(self,lam):
        ctx.prec=self.bits
        key=tuple(lam)
        if len(key)!=2 or any(not np.isfinite(v) or v<0 for v in key):
            raise ValueError('Invalid witness ratios')
        a,b=arb(float(lam[self.large])),arb(float(lam[self.small]))
        inv=[1/(1+a*v) for v in self.count_large]
        weighted=arb_mat([[inv[i]*self.cross[i,j] for j in range(self.cross.ncols())]
                          for i in range(self.cross.nrows())])
        gram=self.cross.transpose()*weighted
        size=len(self.count_small)
        schur=arb_mat([[arb(int(i==j))+(b*self.count_small[i] if i==j else arb(0))-a*b*gram[i,j]
                        for j in range(size)] for i in range(size)])
        self.systems_built+=1
        return a,b,inv,schur

    def _logdet_from_system(self,key,a,schur):
        if key in self.detcache:
            return self.detcache[key]
        det=schur.det()
        if not det>0:
            raise ValueError('Positive Schur determinant not proved')
        g=sum(((1+a*v).log() for v in self.count_large),arb(0))+det.log()
        if not g.is_finite():
            raise ValueError('Nonfinite structured log determinant')
        self.detcache[key]=g
        return g

    def logdet(self,lam):
        key=tuple(lam)
        if key in self.detcache:
            return self.detcache[key]
        a,_,_,schur=self._system(lam)
        return self._logdet_from_system(key,a,schur)

    def covariance(self,lam):
        ctx.prec=self.bits;key=tuple(lam)
        if key in self.covcache:
            return self.covcache[key]
        a,b,inv,schur=self._system(lam)
        g=self._logdet_from_system(key,a,schur)
        scaled=arb_mat([[inv[i]*self.zhl[i,j] for j in range(self.hh.ncols())]
                        for i in range(len(inv))])
        rhs=self.zhs-a*(self.cross.transpose()*scaled)
        es=schur.solve(rhs);self.predictor_solves+=1
        work=self.zhl-b*(self.cross*es)
        el=arb_mat([[inv[i]*work[i,j] for j in range(work.ncols())] for i in range(work.nrows())])
        gh=self.hh-a*(self.zhl.transpose()*el)-b*(self.zhs.transpose()*es)
        pieces=[None,None];pieces[self.large]=el;pieces[self.small]=es
        eh=arb_mat([[pieces[k][i,j] for j in range(gh.ncols())]
                    for k in range(2) for i in range(pieces[k].nrows())])
        self.covcache[key]=(g,gh,eh)
        return g,gh,eh

    def gradient_grams(self,lam,eh):
        key=tuple(lam)
        if key in self.gradcache:
            return self.gradcache[key]
        grams=[];start=0
        for size in self.sizes:
            e=submat(eh,range(start,start+size),range(eh.ncols()))
            grams.append(e.transpose()*e);start+=size
        self.gram_constructions+=1;self.gradcache[key]=grams
        return grams

    def profile(self,lam,support):
        ctx.prec=self.bits;key=(tuple(lam),tuple(support))
        if key in self.evalcache:
            return self.evalcache[key]
        g,gh,eh=self.covariance(lam)
        grams=self.gradient_grams(lam,eh)
        ix=[*range(self.k),*(self.k+j for j in support)]
        last=self.hh.ncols()-1;by=submat(gh,ix,[last])
        beta=submat(gh,ix,ix).solve(by)
        s=gh[last,last]-(by.transpose()*beta)[0,0]
        if not s>0:
            raise ValueError('Unverified positive residual')
        grad=[]
        for gram in grams:
            byq=submat(gram,ix,[last]);xx=submat(gram,ix,ix)
            norm=gram[last,last]-2*(byq.transpose()*beta)[0,0]+(beta.transpose()*xx*beta)[0,0]
            grad.append(-arb(self.n)*norm/s)
        h=arb(self.n)*(s/self.n).log();out=(g+h,h,grad)
        if not all(v.is_finite() for v in [g+h,h,*grad]):
            raise ValueError('Nonfinite structured profile enclosure')
        self.evalcache[key]=out
        return out

    def component_logdet(self,k,cutoff):
        ctx.prec=self.bits
        if not np.isfinite(cutoff) or cutoff<=0:
            raise ValueError('Invalid tail cutoff')
        start=sum(self.sizes[:k]);counts=[self.a[i,i] for i in range(start,start+self.sizes[k])]
        if not sum(counts,arb(0))>0:
            raise ValueError('Nonzero covariance component not proved')
        return sum(((1+arb(float(cutoff))*v).log() for v in counts),arb(0))

    def statistics(self):
        return {'backend':'two_crossed_intercepts_schur_v1','schur_dimension':self.sizes[self.small],
                'original_random_columns':self.r,'systems_built':self.systems_built,
                'predictor_solves':self.predictor_solves,'gradient_gram_constructions':self.gram_constructions,
                'covariance_cache_limit':self.covcache.capacity,'covariance_cache_peak':self.covcache.peak,
                'profile_cache_limit':self.evalcache.capacity,'profile_cache_peak':self.evalcache.peak,
                'gradient_cache_limit':self.gradcache.capacity,'gradient_cache_peak':self.gradcache.peak,
                'covariance_cache_evictions':self.covcache.evictions,
                'profile_cache_evictions':self.evalcache.evictions,'determinant_cache_entries':len(self.detcache)}
