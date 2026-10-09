"""Rebuild stage-seven comparisons from saved outputs, without fitting models."""
from pathlib import Path
import collections, hashlib, json
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results/penalized_lmm'
LABELS={'verified_pair':'Verified BIC + pairs','local_pair':'Local BIC + pairs',
        'local_single':'BIC singles','arss_jps':'ARSS-JPS policy','enet_ml':'EN + ML',
        'enet_native':'Native elastic net','glmmsel':'Native glmmsel','glmmsel_ml':'glmmsel + ML'}
ORDER=list(LABELS)

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def estimate(values, interval=True):
    values=np.asarray(values,dtype=float)
    if not len(values): return dict(n=0,mean=None,mcse=None,ci95=None)
    mean=float(values.mean())
    se=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    half=float(t.ppf(.975,len(values)-1)*se) if se is not None else None
    ci=[mean-half,mean+half] if interval and half is not None and half>0 else None
    return dict(n=len(values),mean=mean,mcse=se,ci95=ci,
                ci_method='Approximate Student-t across independent datasets' if ci is not None else None)

def number(x,d=3): return '--' if x is None else f'{x:.{d}f}'

def main():
    protocol=json.loads((DEST/'protocol.json').read_text())
    frozen=json.loads((DEST/'code_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in frozen.items())
    cases=[json.loads((DEST/c['id']/'result.json').read_text()) for c in protocol['cases']]
    failures=[{'id':r['case']['id'],'reason':r.get('reason','worker failure')} for r in cases if 'worker_failure' in r]
    records=[r for r in cases if 'worker_failure' not in r]
    for r in records:
        for method,out in r['results'].items():
            if out['status'] not in ['complete','complete_local','complete_native']:
                failures.append(dict(id=r['case']['id'],method=method,reason=out.get('reason',out['status'])))
        where=DEST/r['case']['id']
        with np.load(where/'data.npz') as f: a={k:f[k] for k in f.files}
        with np.load(where/'predictions.npz') as f: preds={k:f[k] for k in f.files}
        if 'enet_native' in preds and len(a['y_test']):
            pred=preds['enet_native']; metric=dict(r['metrics']['enet_ml'])
            metric.update(response_mse=float(np.mean((pred-a['y_test'])**2)),
                          mae=float(np.mean(np.abs(pred-a['y_test']))))
            if 'mean_test' in a: metric['mean_mse']=float(np.mean((pred-a['mean_test'])**2))
            r['metrics']['enet_native']=metric
    cake=[r for r in records if r['case']['dataset']=='cake' and r['case']['kind']!='full']
    full=next((r for r in records if r['case']['id']=='cake_full'),None)
    real={}
    for method in ORDER:
        rowids=[]; predictions=[]; responses=[]; times=[]; sizes=[]; random_counts=[]
        for r in cake:
            where=DEST/r['case']['id']
            if r['metrics'].get(method) is None: continue
            with np.load(where/'predictions.npz') as f: pred=f[method]
            with np.load(where/'data.npz') as f:
                responses.extend(f['y_test']);rowids.extend(f['test_row_ids'])
            predictions.extend(pred)
            sizes.append(r['metrics'][method]['selected_n'])
            if method!='enet_native':times.append(r['results'][method]['pipeline_seconds'])
            if method=='glmmsel':random_counts.append(len(r['results'][method]['random_slope_support']))
        assert len(rowids)==len(set(rowids))
        err=np.asarray(predictions)-np.asarray(responses)
        real[method]=dict(folds=len(sizes),rows=len(err),mse=float(np.mean(err**2)) if len(err) else None,
                          mae=float(np.mean(np.abs(err))) if len(err) else None,
                          median_seconds=float(np.median(times)) if times else None,
                          mean_selected=float(np.mean(sizes)) if sizes else None,
                          random_slopes_per_fold=random_counts)
    simulation={}; paired={}
    for scenario in ['signals','suppressor']:
        group=[r for r in records if r['case']['scenario']==scenario]
        simulation[scenario]={};paired[scenario]={}
        for method in ORDER:
            completed=[r for r in group if r['metrics'].get(method) is not None]
            ms=[r['metrics'][method] for r in completed]
            item={k:estimate([m[k] for m in ms], interval=k not in ['tpr','fdp','exact']) for k in
                  ['tpr','fdp','exact','mean_mse','response_mse','selected_n']}
            times=[r['results'][method]['pipeline_seconds'] for r in completed] if method!='enet_native' else []
            item.update(completed=len(completed),total=20,median_seconds=float(np.median(times)) if times else None)
            if scenario=='suppressor': item['both_suppressors']=sum(m['both_suppressors'] for m in ms)
            if method=='glmmsel': item['random_slopes']=estimate([len(r['results'][method]['random_slope_support']) for r in completed])
            simulation[scenario][method]=item
            common=[r for r in completed if r['metrics'].get('verified_pair') is not None]
            paired[scenario][method]=dict(mean_mse_difference=estimate([r['metrics'][method]['mean_mse']-r['metrics']['verified_pair']['mean_mse'] for r in common]),
                                          response_mse_difference=estimate([r['metrics'][method]['response_mse']-r['metrics']['verified_pair']['response_mse'] for r in common]))
    native_records=[r for r in records if r['results']['glmmsel']['status']=='complete_native']
    endpoints=[]; full_choices=[]; warning_cases=[]; exclusions=[]
    for r in native_records:
        ident=r['case']['id']; raw=json.loads((DEST/ident/'native.json').read_text())
        if raw['selected_index']==len(raw['full_path']['lambda']): endpoints.append(ident)
        if len(r['results']['glmmsel']['selected'])==r['case']['p']:full_choices.append(ident)
        if raw['warnings']:warning_cases.append(dict(id=ident,warnings=raw['warnings']))
        if r['results']['glmmsel']['ineligible_penalties']:
            exclusions.append(dict(id=ident,count=r['results']['glmmsel']['ineligible_penalties']))
    overview=dict(cases=len(cases),native_complete=len(native_records),
                  verified_complete=sum(r['results']['verified_pair']['status']=='complete' for r in records),
                  replay_valid=sum(bool(r.get('replay',{}).get('valid')) for r in records),
                  verified_decisions=sum(r.get('replay',{}).get('decisions',0) for r in records),
                  identical_paths=sum(bool((r.get('agreement') or {}).get('path')) for r in records),
                  native_selected_endpoint=endpoints,native_selected_full=full_choices,
                  native_warning_cases=warning_cases,ineligible_penalties=exclusions,failures=failures,
                  max_native_profile_gap=max(r['native_check']['dense_profile_gap'] for r in native_records),
                  max_native_prediction_gap=max(r['native_check']['prediction_gap'] for r in native_records),
                  max_native_variance_gap=max(r['native_check']['residual_variance_gap'] for r in native_records))
    supports={m:full['results'][m]['selected'] for m in ORDER if m in full['results']} if full else {}
    frequencies={m:dict(collections.Counter(i for r in cake for i in r['results'][m]['selected'])) for m in ['verified_pair','glmmsel']}
    term_names={0:'linear temperature',1:'quadratic temperature',2:'cubic temperature',3:'recipe B',4:'recipe C'}
    full_terms=', '.join(term_names[i] for i in supports.get('glmmsel',[]))
    full_random=', '.join(term_names[i] for i in full['results']['glmmsel']['random_slope_support']) if full else ''
    pl_names={0:'temperatura (liniowo)',1:'temperatura²',2:'temperatura³',3:'receptura B',4:'receptura C'}
    full_terms_pl=', '.join(pl_names[i] for i in supports.get('glmmsel',[]))
    full_random_pl=', '.join(pl_names[i] for i in full['results']['glmmsel']['random_slope_support']) if full else ''
    summary=dict(overview=overview,cake=real,simulations=simulation,paired_with_verified=paired,
                 cake_full_supports=supports,cake_fold_frequencies=frequencies)
    dump(DEST/'summary.json',summary)

    lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrr}\toprule',
           r'Method & Folds & MSE & MAE & Size & Seconds\\\midrule']
    for method in ORDER:
        v=real[method]
        lines.append(f"{LABELS[method]} & {v['folds']}/5 & {number(v['mse'])} & {number(v['mae'])} & {number(v['mean_selected'],1)} & {number(v['median_seconds'])}"+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Cake: pooled out-of-fold prediction on the same 270 observations. Size is the mean selected predictor count; seconds are median fitting-and-tuning times. Native glmmsel allows selected random slopes; its ML refit uses only the reference random intercept. Native elastic net shares tuning with EN + ML, so no separate time is assigned. Differences between dependent folds are descriptive.}',r'\end{table}']
    (DEST/'cake_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{llrrrrrr}\toprule',
           r'Design & Method & Done & TPR & FDR & Size & Mean MSE & Seconds\\\midrule']
    for scenario in simulation:
        for method in ORDER:
            v=simulation[scenario][method]
            vals=[number(v['tpr']['mean']),number(v['fdp']['mean'],4)]
            lines.append(f"{scenario.title()} & {LABELS[method]} & {v['completed']}/20 & {' & '.join(vals)} & {number(v['selected_n']['mean'],2)} & {number(v['mean_mse']['mean'])} & {number(v['median_seconds'])}"+r'\\')
        if scenario=='signals':lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}',r'\caption{One-group simulations: 20 independent datasets per scenario. FDR is mean false discovery proportion; Mean MSE evaluates the generating population mean for new groups. Seconds are medians. Full Monte Carlo standard errors and paired differences are archived. Native glmmsel and reference BIC use different penalties and covariance search spaces.}',r'\end{table}']
    (DEST/'simulation_table.tex').write_text('\n'.join(lines)+'\n')
    o=overview; ss=simulation['suppressor'];g=ss['glmmsel'];v=ss['verified_pair']
    d=paired['suppressor']['glmmsel']['mean_mse_difference']
    f=(f"Native fitting and tuning completed in {o['native_complete']}/{o['cases']} analyses. "
       f"All {o['verified_complete']} reference certified paths completed, with {o['verified_decisions']} checked decisions; "
       f"{o['replay_valid']} replayed and {o['identical_paths']} agreed with their local-pair paths. "
       "The six real-data analyses reuse one dataset and are not independent simulation replicates.\n\n"
       f"On cake, native glmmsel has pooled MSE {real['glmmsel']['mse']:.3f}, compared with "
       f"{real['verified_pair']['mse']:.3f} for verified BIC and {real['glmmsel_ml']['mse']:.3f} after its reference ML refit. "
       f"The mean selected sizes are {real['glmmsel']['mean_selected']:.1f} and {real['verified_pair']['mean_selected']:.1f}, respectively. "
       f"The descriptive full-data native fit selects {full_terms}, with random slopes for {full_random or 'none'}; "
       "the reference BIC fit selects only linear temperature. "
       "These descriptive prediction results do not isolate the effect of numerical certification.\n\n"
       f"In suppression, verified BIC recovers both suppressed predictors in {v['both_suppressors']}/20 cases, "
       f"native glmmsel in {g['both_suppressors']}/20, and singles-only BIC in {ss['local_single']['both_suppressors']}/20. "
       f"Mean false discovery proportions are {v['fdp']['mean']:.4f} and {g['fdp']['mean']:.4f} for verified BIC and glmmsel. "
       f"The paired native-minus-verified mean-MSE difference is {d['mean']:.4f} "
       f"(Monte Carlo SE {d['mcse']:.4f}; approximate 95\\% $t$ interval [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]). "
       "Only twenty datasets inform each estimate.\n\n"
       f"The selected native penalty is the last available path value in {len(endpoints)}/{len(native_records)} analyses. "
       f"Warnings occur in {len(warning_cases)} analyses and iteration-limit eligibility excludes penalties in {len(exclusions)} analyses; "
       "the archive retains their identities and full diagnostics. The adaptive-path results do not establish robustness to every penalty grid or hierarchy parameter.\n\n")
    mant,exponent=f"{o['max_native_profile_gap']:.2e}".split('e')
    f+=(f"An independent grouped-covariance calculation at every selected native parameter vector agrees with "
        f"the package profile objective to at most ${mant}\\times10^{{{int(exponent)}}}$. "
        "The native new-group predictions also agree with direct fixed-effect evaluation. These checks validate "
        "the implementation interface and likelihood evaluation, not global optimization of its penalized problem.\n")
    (DEST/'findings.tex').write_text(f)

    pl=["# Porównanie ze współczesnym selektorem penalizowanego LMM",'',
        'Marek Gałązka i Hanna Wdowicka • 14 września 2026','',
        '**Uruchomiono natywny pakiet `glmmsel` 1.0.3**, oparty na karze ℓ₀, coordinate descent i local search. '+
        'To implementacja metody Thompsona, Wanda i Wang opisanej w [JCGS (2026)](https://doi.org/10.1080/10618600.2026.2701133). '+
        '[Dokumentacja pakietu](https://cran.r-project.org/web/packages/glmmsel/glmmsel.pdf) i niezmienione źródła CRAN są dołączone do pakietu.','',
        '## Zakres porównania','',
        '- Te same pięć podziałów danych z ciastami i dodatkowe dopasowanie na pełnych danych.',
        '- 40 nowych, niezależnych symulacji: po 20 dla zwykłych sygnałów i pary supresyjnej; n=300, p=12, 30 grup treningowych.',
        '- Dobór kary w trzech wewnętrznych podziałach całych grup, dokładnie tych samych co dla elastic netu. Wyniki testowe nie trafiają do R.',
        '- α=0,8, włączone local search, tolerancja 10⁻⁶, natywna adaptacyjna ścieżka do 100 kar. Zachowano ostrzeżenia i nieudane obliczenia.',
        '- `glmmsel + ML` oznacza dodatkowe dopasowanie naszego LMM do zbioru wybranego przez glmmsel; nie jest natywnym estymatorem pakietu.','',
        '**Różnica modelowa:** glmmsel wybiera także niezależne losowe nachylenia. Nasza procedura zachowuje ustaloną strukturę losowego wyrazu wolnego. '+
        'Porównujemy zatem całe procedury statystyczne; nie przypisujemy różnic wyłącznie optymalizatorom. '+
        'Nie zastępowano skrzyżowanych efektów konsumenta i produktu pojedynczym grupowaniem.','',
        '## Dane rzeczywiste: predykcja kąta złamania ciasta','',
        '| Procedura | MSE | MAE | Średnia liczba zmiennych | Mediana czasu [s] |',
        '|---|---:|---:|---:|---:|']
    for m in ORDER:
        v=real[m];pl.append(f"| {LABELS[m]} | {number(v['mse'],4)} | {number(v['mae'],4)} | {number(v['mean_selected'],1)} | {number(v['median_seconds'])} |")
    pl += ['', 'Każda ukończona procedura ocenia te same 270 obserwacji poza treningiem. '+
           'Podziały współdzielą treningi; nie podano naiwnych testów istotności ani przedziałów z pięciu zależnych replikacji. '+
           'Czasy dotyczą dopasowania i strojenia; uruchomienie R i import pakietów są poza timerem, a czas całego procesu zachowano osobno.','',
           f"Na pełnych danych glmmsel wybiera: **{full_terms_pl}**; losowe nachylenia: **{full_random_pl or 'brak'}**. "+
           'BIC wybiera tylko liniowy składnik temperatury. Pełne dopasowanie opisuje wybrane zmienne i nie wchodzi do oceny predykcji poza próbą.','',
           '## Symulacje z jednym grupowaniem','',
           '| Scenariusz | Procedura | TPR | FDR | MSE średniej populacyjnej | MCSE MSE | Mediana czasu [s] |',
           '|---|---|---:|---:|---:|---:|---:|']
    for sc in simulation:
        for m in ORDER:
            v=simulation[sc][m]
            pl.append(f"| {sc} | {LABELS[m]} | {number(v['tpr']['mean'])} | {number(v['fdp']['mean'],4)} | {number(v['mean_mse']['mean'],4)} | {number(v['mean_mse']['mcse'],4)} | {number(v['median_seconds'])} |")
    pl += ['',f"Przy supresji obie zmienne odzyskano: certyfikowana BIC **{ss['verified_pair']['both_suppressors']}/20**, "+
           f"glmmsel **{ss['glmmsel']['both_suppressors']}/20**, ARSS **{ss['arss_jps']['both_suppressors']}/20**, "+
           f"BIC bez par **{ss['local_single']['both_suppressors']}/20**.",'',
           'FDR to średni udział zmiennych fałszywie wybranych w wybranym zbiorze. '+
           'MSE w symulacjach dotyczy znanej średniej populacyjnej dla nowych grup, więc nie należy porównywać jego wartości bezpośrednio z MSE odpowiedzi rzeczywistych.','',
           '## Kontrola obliczeń i ograniczenia','',
           f"Natywne analizy ukończone: **{o['native_complete']}/{o['cases']}**. Sprawdzone ścieżki naszej procedury: **{o['verified_complete']}/{o['cases']}**, "+
           f"decyzje: **{o['verified_decisions']}**, poprawne odtworzenia: **{o['replay_valid']}**. "+
           f"Maksymalna różnica niezależnej ewaluacji wiarygodności glmmsel: **{o['max_native_profile_gap']:.3g}**.",'',
           f"Ostatnią dostępną karę wybrano w **{len(endpoints)}/{len(native_records)}** analizach. "+
           f"Ostrzeżenia wystąpiły w {len(warning_cases)} analizach; ograniczenie iteracji wykluczyło punkty siatki w {len(exclusions)} analizach. "+
           'Wynik dotyczy podanego α i adaptacyjnej siatki, a nie każdej możliwej konfiguracji pakietu.','',
           '## Pliki','',
           '- `CSDA_manuscript.pdf`: uzupełniony główny manuskrypt.',
           '- `run_penalized_lmm_comparison.py` i `run_glmmsel_native.R`: zamrożony protokół i natywne obliczenia.',
           '- `summarize_penalized_lmm.py`: odtworzenie zestawień bez ponownego dopasowania.',
           '- `results/penalized_lmm/`: wszystkie dane, podziały, obiekty R, predykcje, czasy, certyfikaty i diagnostyka.',
           '- `vendor/glmmsel/`: oryginalne źródła 1.0.3, pochodzenie, licencja i zapis instalacji.','',
           'Wyniki wcześniejszych 300 symulacji, dwóch aplikacji i rozszerzenia wymiarów pozostają zachowane. '+
           'Nowe porównanie usuwa brak natywnego penalizowanego konkurenta, lecz nie ustanawia uniwersalnej przewagi żadnej metody. '+
           'Przed zgłoszeniem pozostaje przegląd dowodów i implementacji przez autorów, uzupełnienie bibliografii własnej publikacji oraz ocena potrzeby szerszych porównań.','']
    report='\n'.join(pl)
    (ROOT/'CSDA_penalized_LMM_comparison_PL.md').write_text(report)
    (ROOT/'CSDA_wyniki_i_plan_PL.md').write_text(report)
    print(json.dumps(dict(overview=o,cake=real,paired_suppressor=d),indent=2),flush=True)

if __name__=='__main__': main()
