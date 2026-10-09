"""Targeted follow-ups: scale, box range, ties, rank reversals and tail exclusion."""
from pathlib import Path
import csv,json,time
import numpy as np
from covariance_screening import CrossedGaussian,screen_step,exhaustive_reference,automatic_tail_box,box_lower
from run_experiments import generate,run_case


def write_rows(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    out=Path('results');out.mkdir(exist_ok=True)
    rows=[]
    for p in [100,300]:
        for scenario in ['weak','group_aligned','null']:
            seed=420000+p+['weak','group_aligned','null'].index(scenario)
            row,detail=run_case(seed,1000,p,scenario,max_splits=1500,time_budget=20)
            rows.append(row)
            print('SCALE',json.dumps(row),flush=True)
    write_rows(out/'scaling.csv',rows)
    rows=[]
    for hi in [1.,4.,16.]:
        row,detail=run_case(202609340,400,30,'group_aligned',hi=hi,max_splits=2000,time_budget=20)
        rows.append(row)
        print('BOX',json.dumps(row),flush=True)
    write_rows(out/'box_sensitivity.csv',rows)
    row,detail=run_case(47771,300,12,'exact_tie',max_splits=250,time_budget=5)
    (out/'exact_tie.json').write_text(json.dumps({'row':row,'detail':detail},indent=2))
    print('TIE',json.dumps(row),flush=True)
    # A deterministic case in the main pilot already requests two local refits.
    main_details=json.loads((out/'details.json').read_text())
    with (out/'pilot.csv').open() as f:main_rows=list(csv.DictReader(f))
    reversals=[]
    for row,detail in zip(main_rows,main_details):
        initial=int(np.argmax(detail['screen']['q_scores']))
        final=detail['screen']['winner']
        if initial!=final:
            reversals.append({'scenario':row['scenario'],'seed':int(row['seed']),
                              'initial_GLS_winner':initial,'certified_winner':final,
                              'reference_winner':detail['reference']['winner'],
                              'q_scores':detail['screen']['q_scores'],
                              'full_refit_objectives':detail['reference']['f'],
                              'screen':detail['screen']})
    (out/'rank_reversals.json').write_text(json.dumps(reversals,indent=2))
    print('REVERSALS',len(reversals),flush=True)
    tail_rows=[]
    for scenario,seed in [('weak',202609240),('group_aligned',202609340)]:
        args=generate(seed,400,30,scenario)
        model=CrossedGaussian(*args)
        initial=screen_step(model,[0,0],[4,4],max_splits=1000,time_budget=10)
        tail=automatic_tail_box(model,initial['best_feasible_f'],margin=1.)
        before=time.perf_counter()
        screened=screen_step(CrossedGaussian(*args),tail['lo'],tail['hi'],
                             max_splits=3000,time_budget=30)
        reference=exhaustive_reference(CrossedGaussian(*args),tail['lo'],tail['hi'])
        # Independent points beyond each face check the analytic tail inequality.
        minimum_tail_slack=np.inf
        for k in range(2):
            for factor in [1.0001,2.,10.]:
                point=np.array(tail['hi'])*.1
                point[k]=tail['hi'][k]*factor
                ev=model.evaluate(point)
                minimum_tail_slack=min(minimum_tail_slack,float(ev.f[1:].min()-tail['outside_objective_lower']))
        record={'scenario':scenario,'seed':seed,'tail':tail,'screen':screened,
                'reference':reference,'minimum_sampled_tail_slack':minimum_tail_slack,
                'agreement':screened['winner']==reference['winner'],
                'overall_tail_certificate':screened['status']=='separated' and screened['best_feasible_f']<tail['outside_objective_lower']}
        tail_rows.append(record)
        print('TAIL',scenario,tail['hi'],screened['status'],screened['splits'],record['overall_tail_certificate'],flush=True)
    (out/'tail_extension.json').write_text(json.dumps(tail_rows,indent=2))


if __name__=='__main__':main()
