"""A top-three heuristic comparator, NOT the complete published ARSS procedure."""
from pathlib import Path
import csv,json,time
import numpy as np
from covariance_screening import CrossedGaussian
from run_experiments import generate


def main():
    rows=list(csv.DictReader(Path('results/pilot.csv').open()))
    output=[]
    for row in rows:
        args=generate(int(row['seed']),int(row['n']),int(row['p']),row['scenario'])
        start=time.perf_counter()
        m=CrossedGaussian(*args)
        lo=np.zeros(2);hi=np.ones(2)*4
        starts=[lo,(lo+hi)/2,lo+.1*(hi-lo),lo+.9*(hi-lo)]
        base=m.fit(0,lo,hi,starts)
        lam0=np.array(base['lambda']);ev=m.evaluate(lam0)
        ids=np.argsort(ev.f[1:])[:3]
        fits=[m.fit(int(j)+1,lo,hi,[lam0,*starts]) for j in ids]
        winner=int(ids[np.argmin([x['f'] for x in fits])])
        secs=time.perf_counter()-start
        output.append({'scenario':row['scenario'],'seed':int(row['seed']),
                       'top3_winner':winner,'reference_winner':int(row['reference_winner']),
                       'agreement':winner==int(row['reference_winner']),
                       'top3_seconds':secs,'certified_seconds':float(row['screen_seconds']),
                       'certified_over_top3_time':float(row['screen_seconds'])/secs,
                       'note':'Fixed top-three rank-and-refit diagnostic; no significance expansion or rescue. Separate timing run.'})
    with Path('results/top3_diagnostic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
    print(json.dumps({'cases':len(output),'agreements':sum(r['agreement'] for r in output),
                      'median_certified_over_top3_time':float(np.median([r['certified_over_top3_time'] for r in output]))},indent=2))


if __name__=='__main__':main()
