"""Summarize the secondary penalty-grid check without replacing primary data."""
from pathlib import Path
import json
from summarize_comparison_results import estimate,SCENARIOS,PL,EN


def main():
    root=Path(__file__).parent;dest=root/'results/enet_sensitivity'
    rows=json.loads((dest/'rows.json').read_text());assert len(rows)==300
    summary=[];mdrows=[]
    tex=[r'\begin{table}[H]\centering\small',r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Design & \shortstack{FDR\\base / wider} & \shortstack{Refit mean MSE\\base / wider} & \shortstack{Native mean MSE\\base / wider} & \shortstack{Changed\\supports} \\',r'\midrule']
    for scenario in SCENARIOS:
        rr=[r for r in rows if r['scenario']==scenario];assert len(rr)==50
        old=[json.loads((root/'results/comparison'/f"{r['scenario']}_{r['seed']}.json").read_text()) for r in rr]
        ss={'scenario':scenario,'n':len(rr),'larger_penalty_selected':sum(r['grid_changed_choice'] for r in rr),
            'changed_supports':sum(not r['same_support_primary'] for r in rr),'paired':{}}
        for metric in ['tpr','fdp','false_selection_event','exact_recovery','mean_mse','response_mse','native_mean_mse','native_response_mse']:
            primary=[r['metrics']['enet_ml'][metric] for r in old];wide=[r['metrics'][metric] for r in rr]
            ss[metric]={'primary':estimate(primary),'wider':estimate(wide)}
            ss['paired'][metric]=estimate([b-a for a,b in zip(primary,wide) if a is not None and b is not None])
        ss['pair_recovery_primary']=sum(r['metrics']['enet_ml']['pair_recovery'] for r in old) if scenario=='suppressor' else None
        ss['pair_recovery_wider']=sum(r['metrics']['pair_recovery'] for r in rr) if scenario=='suppressor' else None
        summary.append(ss)
        cells=[]
        for metric in ['fdp','mean_mse','native_mean_mse']:
            aa=ss[metric]['primary']['mean'];bb=ss[metric]['wider']['mean'];cells.append(f'{aa:.3f} / {bb:.3f}')
        tex.append(EN[scenario]+' & '+' & '.join(cells)+f" & {ss['changed_supports']}/50"+r' \\')
        mdrows.append(f"| {PL[scenario]} | {' | '.join(cells)} | {ss['changed_supports']}/50 |")
    overview={'datasets':300,'larger_penalty_selected':sum(r['grid_changed_choice'] for r in rows),
              'changed_supports':sum(not r['same_support_primary'] for r in rows),
              'convergence_warnings':sum(r['result']['warning_count'] for r in rows),
              'difference_direction':'wider grid minus primary grid',
              'by_scenario':summary}
    (dest/'summary.json').write_text(json.dumps(overview,indent=2)+'\n')
    tex += [r'\bottomrule',r'\end{tabular}',r'\caption{Secondary elastic-net grid sensitivity on the same datasets and identical inner splits. Every primary penalty is retained; the wider grid extends to 10 and includes an intercept-only candidate. Base and wider results use the same test data. The primary five-method results are unchanged. Paired differences and Monte Carlo standard errors are stored separately.}\label{tab:enet_sensitivity}',r'\end{table}']
    (dest/'sensitivity_table.tex').write_text('\n'.join(tex)+'\n')
    g=next(r for r in summary if r['scenario']=='group_aligned')
    d=g['paired']['mean_mse']
    finding=(f"The wider grid selected a penalty above 1 in {overview['larger_penalty_selected']}/300 datasets "
             f"and changed {overview['changed_supports']}/300 selected supports. "
             f"For group-aligned predictors, the paired refit population-mean MSE difference (wider minus primary) was {d['mean']:.3f}, "
             f"with Monte Carlo standard error {d['mcse']:.3f}. "
             "This is a secondary sensitivity result, not a replacement of the frozen primary comparison. "
             "Validation-loss improvement on a larger candidate grid does not itself establish improved test performance.\n")
    (dest/'sensitivity_findings.tex').write_text(finding)
    text=f'''## Wrażliwość na zakres strojenia elastic netu

Podstawowa siatka kończyła się na karze 1. Przy dużej wariancji odpowiedzi mogła nie obejmować modelu bez predyktorów. W osobnej analizie na tych samych 300 zbiorach zachowano wszystkie pierwotne kary, rozszerzono siatkę do 10 i dodano karę 10⁶ jako wariant z samym wyrazem wolnym. Warunek optymalności zera sprawdzono dla tego wariantu w każdym podziale treningowym i w całym treningu. Nie zmieniono trzech podziałów walidacyjnych, udziałów L1 ani danych testowych.

Szersza siatka wybrała karę powyżej 1 w **{overview['larger_penalty_selected']}/300** przypadków i zmieniła wybrany zbiór w **{overview['changed_supports']}/300** przypadków. To analiza wrażliwości uzupełniająca główny eksperyment. Nie zastępuje wyników pierwotnych i nie jest częścią porównania czasów pięciu metod.

W komórkach tabeli podano wynik siatki podstawowej / szerszej. FDR jest średnim udziałem fałszywych wyborów; oba MSE odnoszą się do prawdziwej średniej populacyjnej w nowych grupach.

| Scenariusz | FDR | MSE po reficie ML | Natywny MSE elastic netu | Zmienione zbiory |
|---|---:|---:|---:|---:|
{chr(10).join(mdrows)}

Dla predyktorów grupowych sparowana zmiana MSE po reficie ML (szersza minus podstawowa siatka) wyniosła **{d['mean']:.3f}**, z błędem Monte Carlo **{d['mcse']:.3f}**. Mniejszy błąd walidacyjny po rozszerzeniu zbioru kar nie musi oznaczać mniejszego błędu na niezależnym teście. Wszystkie sparowane różnice są w `results/enet_sensitivity/summary.json`; wyniki pierwotne pozostają w `results/comparison/`.

Nie odnotowano tu automatycznie poprawy wszystkich miar: porównanie należy odczytywać oddzielnie dla selekcji, skurczonych prognoz i prognoz po reficie. Zwiększenie zakresu kar usuwa konkretną wątpliwość dotyczącą doboru siatki, lecz nie przekształca marginalnego elastic netu w penalizowany LMM.
'''
    (root/'CSDA_enet_sensitivity_PL.md').write_text(text)
    print(json.dumps({k:v for k,v in overview.items() if k!='by_scenario'},indent=2))


if __name__=='__main__':main()
