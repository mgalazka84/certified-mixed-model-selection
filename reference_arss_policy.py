"""Unchanged ARSS search policy from the authors' MIT-licensed repository.

Source: code/arss_python_v03_fullrefit_80.py, function arss_family.
Commit: 0cdcf976178be7ed00af4b0737a7ed97540786a4.
The comparison adapter supplies shared ML fitting and score functions; this is
not a run of the original statsmodels fitting engine. See ARSS_POLICY_LICENSE.
"""

def arss_family(d, active0, oi, di, ng, joint='none'):
    sel=[]; active=active0.copy(); fits=0; nonconv=0; retries=0; singular=0; rescues=0; pair_pool=0; pair_full=0
    for _ in range(MAX_STEPS):
        base,ret=fit_model(d,sel); fits+=1; retries+=int(ret)
        if base is None: nonconv+=1; break
        nonconv+=int(not base.converged); singular+=int(np.min(base.vcomp)<1e-7)
        base_ll=float(base.llf)
        u,R,scores=score_system(d,base,sel,active,oi,di,ng)
        order=list(np.argsort(-scores))
        tested=[]; ev={}; batch=min(K_SINGLE,len(order)); sig=[]
        while len(tested)<len(order):
            target=min(len(order),batch)
            new=[i for i in order[:target] if i not in tested]
            for i in new:
                c=active[i]; z=fit_stat(d,sel,base_ll,[c]); fits+=1; retries+=int(z['retry']); nonconv+=int(not z['conv']); singular+=int(z['singular']); ev[i]=z
            tested.extend(new)
            sig=[i for i in tested if ev[i]['p']<ALPHA]
            if sig or len(tested)==len(order): break
            batch=min(len(order),max(batch+1,2*batch))
        if sig:
            best=max(sig,key=lambda i:ev[i]['lr']); sel.append(active[best]); active.pop(best); continue
        if joint=='none' or len(active)<2: break
        # At this point all single candidates have been tested, so their likelihoods can be reused in mutual conditional tests.
        diag=np.diag(R); pairs=[]
        for i,j in itertools.combinations(range(len(active)),2):
            den=np.sqrt(max(diag[i],0)*max(diag[j],0))
            corr=0. if den<=1e-14 else float(R[i,j]/den)
            if np.isfinite(corr) and abs(corr)>=TAU:
                M=R[np.ix_([i,j],[i,j])]; vv=u[[i,j]]
                try: sc=float(vv@np.linalg.solve(M,vv))
                except np.linalg.LinAlgError: sc=float(vv@np.linalg.lstsq(M,vv,rcond=None)[0])
                pairs.append((max(sc,0.),abs(corr),i,j,corr))
        pairs.sort(reverse=True); m=len(pairs); pair_pool+=m
        if m==0: break
        testpairs=pairs if joint=='all' else pairs[:min(L_PAIR,m)]
        found=[]; aj=ALPHA/m
        for sc,ac,i,j,corr in testpairs:
            ci,cj=active[i],active[j]
            z=fit_stat(d,sel,base_ll,[ci,cj]); fits+=1; pair_full+=1; retries+=int(z['retry']); nonconv+=int(not z['conv']); singular+=int(z['singular'])
            if z['fit'] is None: continue
            # mutual conditional LRTs using already fitted single models
            lli=ev[i]['ll']; llj=ev[j]['ll']; llp=z['ll']
            pi=float(chi2.sf(max(0.,2*(llp-llj)),1))
            pj=float(chi2.sf(max(0.,2*(llp-lli)),1))
            if z['p']<aj and pi<ALPHA and pj<ALPHA:
                found.append((z['lr'],i,j))
        if not found: break
        _,i,j=max(found,key=lambda z:z[0]); ci,cj=active[i],active[j]
        sel.extend([ci,cj]); active=[c for k,c in enumerate(active) if k not in (i,j)]; rescues+=1
    return sel,dict(full_fits=fits,nonconv=nonconv,retries=retries,singular=singular,rescues=rescues,pair_pool=pair_pool,pair_full_fits=pair_full)
