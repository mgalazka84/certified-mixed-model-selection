"""Report the secondary penalty extension, retaining the primary study."""
from pathlib import Path
import hashlib,json
import numpy as np
from summarize_penalized_lmm import estimate,dump

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results/penalized_lmm'
DEST=ROOT/'results/penalized_lmm_grid_sensitivity'

def main():
    protocol=json.loads((DEST/'protocol.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['code_sha256'].items())
    assert all(hashlib.sha256((BASE/k/'native.json').read_bytes()).hexdigest()==h for k,h in protocol['primary_outputs_sha256'].items())
    records=[json.loads((DEST/ident/'result.json').read_text()) for ident in protocol['cases']]
    complete=[r for r in records if r['status']=='complete_native']
    changed=[];extra=[];zeros=[];warnings=[];excluded=[]
    for r in complete:
        ident=r['case']['id'];old=json.loads((BASE/ident/'result.json').read_text())
        if r['selected']!=old['results']['glmmsel']['selected']:changed.append(ident)
        if r['selected_lambda']==0:zeros.append(ident)
        before=json.loads((BASE/ident/'native.json').read_text())
        if r['selected_lambda'] not in before['full_path']['lambda']:extra.append(ident)
        if r['warnings']:warnings.append(dict(id=ident,warnings=r['warnings']))
        if r['ineligible_penalties']:excluded.append(dict(id=ident,count=r['ineligible_penalties']))
    cake={}
    for method in ['glmmsel','glmmsel_ml']:
        predictions=[];responses=[];sizes=[]
        folds=[r for r in complete if r['case']['dataset']=='cake' and r['case']['kind']!='full']
        for r in folds:
            ident=r['case']['id']
            with np.load(DEST/ident/'predictions.npz') as f:predictions.extend(f[method])
            with np.load(BASE/ident/'data.npz') as f:responses.extend(f['y_test'])
            sizes.append(len(r['selected']))
        e=np.asarray(predictions)-responses
        cake[method]=dict(folds=len(folds),rows=len(e),mse=float(np.mean(e**2)),
                          mae=float(np.mean(np.abs(e))),mean_selected=float(np.mean(sizes)))
    simulation={}
    for scenario in ['signals','suppressor']:
        group=[r for r in complete if r['case']['scenario']==scenario]
        simulation[scenario]={}
        for method in ['glmmsel','glmmsel_ml']:
            ms=[r['metrics'][method] for r in group]
            item={k:estimate([m[k] for m in ms], interval=k not in ['fdp','tpr']) for k in ['mean_mse','response_mse','fdp','tpr','selected_n']}
            if scenario=='suppressor':item['both_suppressors']=sum(m['both_suppressors'] for m in ms)
            deltas=[]
            for r in group:
                old=json.loads((BASE/r['case']['id']/'result.json').read_text())
                deltas.append(r['metrics'][method]['mean_mse']-old['metrics']['verified_pair']['mean_mse'])
            item['paired_mean_mse_minus_verified']=estimate(deltas)
            simulation[scenario][method]=item
    overview=dict(cases=len(records),complete=len(complete),changed_supports=changed,
                  selected_added_lambda=extra,selected_zero_lambda=zeros,warnings=warnings,excluded_penalties=excluded,
                  failures=[dict(id=r['case']['id'],reason=r.get('reason')) for r in records if r['status']!='complete_native'],
                  max_profile_gap=max(r['native_check']['dense_profile_gap'] for r in complete))
    summary=dict(overview=overview,cake=cake,simulations=simulation)
    dump(DEST/'summary.json',summary)
    ss=simulation['suppressor']['glmmsel'];si=simulation['signals']['glmmsel']
    latex=(r'\paragraph{Sensitivity to the lower end of the penalty path.}'+ '\n'
           'Because 11 primary selections occur at the path endpoint, we repeat native tuning on all '
           '46 stored inputs after adding the last penalty divided by 10 and 100, and zero, to each original grid. '
           'The hierarchy parameter, group splits, convergence gates and loss are unchanged; this is a secondary '
           'analysis motivated by the endpoint diagnostic, not a new independent study. '
           f"All {len(complete)}/{len(records)} analyses complete; {len(extra)} choose an added penalty, "
           f"{len(zeros)} choose zero and {len(changed)} change selected predictor sets. "
           f"On cake the extended native MSE is {cake['glmmsel']['mse']:.3f}, "
           f"and the post-selection ML MSE is {cake['glmmsel_ml']['mse']:.3f}. "
           f"The extended native mean-MSE values are {si['mean_mse']['mean']:.4f} for ordinary signals and "
           f"{ss['mean_mse']['mean']:.4f} for suppression, with FDR {si['fdp']['mean']:.3f} and {ss['fdp']['mean']:.3f}. "
           f"Both suppressed predictors are recovered in {ss['both_suppressors']}/20 cases. "
           'Primary and extended-grid outputs are retained separately; the sensitivity check does not optimize over '
           'other hierarchy parameters or establish uniform superiority.\n')
    (DEST/'findings.tex').write_text(latex)
    pl=['','## Sprawdzenie dolnego końca siatki kar','',
        'Ponieważ w 11 analizach wybór kończył się na ostatniej karze, powtórzono strojenie **dla wszystkich 46 danych**. '+
        'Dodano ostatnią karę podzieloną przez 10 i 100 oraz karę równą 0. Pozostałe ustawienia i podziały są identyczne. '+
        'To analiza wrażliwości na tych samych danych, przeprowadzona po diagnostyce wyników głównych.','',
        f"Ukończono **{len(complete)}/{len(records)}** analiz; dodatkową karę wybrano w {len(extra)}, "+
        f"karę 0 w {len(zeros)}, a zbiór zmiennych zmienił się w {len(changed)}. "+
        f"Ostrzeżenia: {len(warnings)} analiz; punkty wykluczone przez kryterium zbieżności: {len(excluded)} analiz.",'',
        '| Wynik | Natywny glmmsel: rozszerzona siatka |', '|---|---:|',
        f"| Cake: MSE odpowiedzi | {cake['glmmsel']['mse']:.4f} |",
        f"| Zwykłe sygnały: MSE średniej | {si['mean_mse']['mean']:.4f} |",
        f"| Zwykłe sygnały: FDR | {si['fdp']['mean']:.4f} |",
        f"| Supresja: MSE średniej | {ss['mean_mse']['mean']:.4f} |",
        f"| Supresja: FDR | {ss['fdp']['mean']:.4f} |",
        f"| Supresja: obie zmienne odzyskane | {ss['both_suppressors']}/20 |",'',
        f"MSE dla cake po dodatkowym dopasowaniu naszego ML wynosi {cake['glmmsel_ml']['mse']:.4f}. "+
        'Pełne wyniki są w `results/penalized_lmm_grid_sensitivity/`. Badanie nie obejmuje strojenia parametru hierarchii α.','']
    text='\n'.join(pl)
    for name in ['CSDA_penalized_LMM_comparison_PL.md','CSDA_wyniki_i_plan_PL.md']:
        p=ROOT/name;s=p.read_text().split('\n## Sprawdzenie dolnego końca siatki kar')[0]
        p.write_text(s.rstrip()+'\n'+text)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
