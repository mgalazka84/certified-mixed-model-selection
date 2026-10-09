"""Build the CSDA application/extension results directly from retained records."""
from pathlib import Path
import hashlib,json
import numpy as np
from run_csda_validation import ROOT,DEST,METHODS,dump

LABELS={'verified_pair':'Verified BIC + pairs','local_pair':'Local BIC + pairs',
        'local_single':'BIC singles','arss_jps':'ARSS-JPS policy','enet_ml':'EN + ML',
        'enet_native':'Native elastic net','null_ml':'Intercept-only ML','full_ml':'Full-dictionary ML'}

def stat(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'mcse':float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
            'n':len(x),'median':float(np.median(x)),'p90':float(np.quantile(x,.9))} if len(x) else None

def main():
    protocol=json.loads((DEST/'protocol.json').read_text())
    hashes=json.loads((DEST/'code_sha256.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in hashes.items())
    records=[]
    for c in protocol['cases']:
        d=json.loads((DEST/f"{c['id']}.json").read_text());assert d['case']==c;records.append(d)
    real={};real_records=[r for r in records if r['case']['dataset']!='stress']
    order=['verified_pair','local_pair','local_single','arss_jps','enet_ml','enet_native','null_ml','full_ml']
    for dataset,nfold in [('carrots',9),('cake',5)]:
        folds=[r for r in real_records if r['case']['dataset']==dataset and r['case']['kind'].startswith('fold')]
        allres={}
        for method in order:
            actual='enet_ml' if method=='enet_native' else method
            errors=[];seconds=[];supports=[];test_ids=[];done=0
            for r in folds:
                rr=r.get('results',{}).get(actual,{})
                if rr.get('status') not in ('complete','complete_local'):continue
                ident=r['case']['id'];done+=1;seconds.append(rr['pipeline_seconds']);supports.append(rr['selected'])
                with np.load(DEST/f'{ident}_data.npz') as a,np.load(DEST/f'{ident}_predictions.npz') as p:
                    err=p[method]-a['y_test'];errors.extend(err.tolist());test_ids.extend(a['test_row_ids'].tolist())
            assert len(test_ids)==len(set(test_ids))
            n=len(errors)
            allres[method]={'completed_folds':done,'total_folds':nfold,'tested_rows':n,
                            'mse':float(np.mean(np.square(errors))) if n else None,
                            'mae':float(np.mean(np.abs(errors))) if n else None,
                            'median_pipeline_seconds':float(np.median(seconds)) if seconds else None,
                            'selected_supports':supports,
                            'selection_count':{str(j):sum(j in s for s in supports) for j in range(folds[0]['case']['p'])}}
        full=next(r for r in real_records if r['case']['id']==dataset+'_full')
        real[dataset]={'methods':allres,'full_supports':{m:r.get('selected') for m,r in full.get('results',{}).items()},
                       'features':full['case']['features'],'n':full['case']['n_train'],'groups':full['case']['groups'],
                       'qualification':'One partition per dataset; dependent folds. No iid fold uncertainty. Native EN and EN + ML use the same selected support.'}
    stress=[r for r in records if r['case']['dataset']=='stress'];groups={}
    for p in [24,40]:
        for scenario in ['signals','suppressor']:
            rr=[r for r in stress if r['case']['p']==p and r['case']['scenario']==scenario]
            methods={}
            for m in METHODS[:5]:
                good=[r for r in rr if r.get('results',{}).get(m,{}).get('status') in ('complete','complete_local')]
                methods[m]={'complete':len(good),'total':len(rr),
                            'metrics':{k:stat([r['metrics'][m][k] for r in good]) for k in ['tpr','fdp','exact','mean_mse','response_mse']},
                            'seconds':stat([r['results'][m]['pipeline_seconds'] for r in good])}
            groups[f'{p}_{scenario}']={'p':p,'n':rr[0]['case']['n_train'],'groups':rr[0]['case']['groups'],'scenario':scenario,'methods':methods}
    def counts(rr):
        completed=[r for r in rr if r.get('results',{}).get('verified_pair',{}).get('status')=='complete']
        comparable=[r for r in completed if r.get('agreement') is not None]
        return {'cases':len(rr),'verified_complete':len(completed),
                'replay_valid':sum(bool(r.get('replay',{}).get('valid')) for r in completed),
                'decisions':sum(len(r['results']['verified_pair']['stages']) for r in completed),
                'pair_additions':sum(r['results']['verified_pair']['pair_additions'] for r in completed),
                'comparable':len(comparable),'identical_paths':sum(r['agreement']['path'] for r in comparable),
                'failures':[{'case':r['case']['id'],'reason':r.get('worker_failure') or {m:z['status'] for m,z in r.get('results',{}).items() if z['status'] not in ('complete','complete_local')}} for r in rr if r.get('worker_failure') or any(z['status'] not in ('complete','complete_local') for z in r.get('results',{}).values())]}
    summary={'real':real,'stress':groups,'overview':{'all':counts(records),'real':counts(real_records),'stress':counts(stress)}}
    dump(DEST/'summary.json',summary)
    macro='\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in {
       'ValidationCases':48,'ValidationComplete':summary['overview']['all']['verified_complete'],
       'ValidationDecisions':summary['overview']['all']['decisions'],
       'RealComplete':summary['overview']['real']['verified_complete'],
       'StressComplete':summary['overview']['stress']['verified_complete']}.items())
    (DEST/'manuscript_macros.tex').write_text(macro+'\n')
    rows=[]
    for ds in ['carrots','cake']:
        for m in order:
            r=real[ds]['methods'][m]
            tt='--' if m=='enet_native' else f"{r['median_pipeline_seconds']:.3f}"
            rows.append(f"{ds.title()} & {LABELS[m]} & {r['completed_folds']}/{r['total_folds']} & {r['tested_rows']} & {r['mse']:.3f} & {r['mae']:.3f} & {tt} \\\\")
        rows.append(r'\midrule')
    text=r'''\begin{table}[H]\centering\small
\begin{tabular}{llrrrrr}\toprule
Data & Method & Folds & Rows & MSE & MAE & Seconds\\\midrule
'''+ '\n'.join(rows[:-1])+r'''
\bottomrule\end{tabular}
\caption{Pooled out-of-fold prediction for new groups. Seconds are median
complete-pipeline times. Native EN uses the same tuned support as EN + ML;
its separate timing is not inferred from that pipeline. All rows are evaluated
once for each completed method. Differences are descriptive.}\label{tab:real}
\end{table}
'''
    (DEST/'real_table.tex').write_text(text)
    a=summary['overview']['real']
    text=(f"All {a['verified_complete']}/{a['cases']} verified real-data paths completed, including 14 training-fold\n"
          f"paths and two full-data descriptions, with {a['decisions']} checked decisions. Every completed\n"
          f"path replayed successfully, and all {a['identical_paths']} comparable local-pair paths agreed.\n")
    for ds in ['carrots','cake']:
        v=real[ds]['methods']['verified_pair'];null=real[ds]['methods']['null_ml'];native=real[ds]['methods']['enet_native']
        feats=[real[ds]['features'][j].replace('_',r'\_') for j in real[ds]['full_supports']['verified_pair']]
        text+=(f"\nFor {ds}, verified out-of-fold MSE was {v['mse']:.3f}, compared with {null['mse']:.3f}\n"
               f"for the intercept-only baseline and {native['mse']:.3f} for native elastic-net predictions.\n"
               f"The full-data verified support was \\texttt{{{', '.join(feats)}}}.\n")
    carrot_counts=real['carrots']['methods']['verified_pair']['selection_count']
    cake_counts=real['cake']['methods']['verified_pair']['selection_count']
    text+=(f"\nAcross the consumer-training folds, the verified procedure selected \\texttt{{sens2}}\n"
           f"in {carrot_counts['1']}/9 fits, \\texttt{{sens1}} in {carrot_counts['0']}/9 and\n"
           f"\\texttt{{Homesize\\_3}} in {carrot_counts['3']}/9. The linear temperature column\n"
           f"was selected in {cake_counts['0']}/5 cake-training fits. These dependent-fold\n"
           "frequencies describe stability within this partition, not selection probabilities.\n")
    text+=('\nThe observed prediction gains are modest for consumer preferences and larger\n'
           'for the cake experiment. They do not establish a predictive advantage of\n'
           'certification: local BIC paths agree, and native elastic net has the lowest\n'
           'pooled MSE among the tested procedures for carrots.\n')
    (DEST/'real_findings.tex').write_text(text)
    rows=[]
    for key,g in groups.items():
        for m,v in g['methods'].items():
            def val(metric):return f"{v['metrics'][metric]['mean']:.3f}" if v['metrics'][metric] else '--'
            sec=f"{v['seconds']['median']:.2f}" if v['seconds'] else '--'
            rows.append(f"{g['p']} & {g['scenario'].title()} & {LABELS[m]} & {v['complete']}/{v['total']} & {val('tpr')} & {val('fdp')} & {val('mean_mse')} & {sec} \\\\")
        rows.append(r'\midrule')
    (DEST/'stress_table.tex').write_text(r'''\begin{table}[H]\centering\small
\begin{tabular}{rllrrrrr}\toprule
$p$ & Design & Method & Done & TPR & FDR & Mean MSE & Seconds\\\midrule
'''+ '\n'.join(rows[:-1])+r'''
\bottomrule\end{tabular}
\caption{Simultaneous-dimension extension. At $p=24$, $n=600$ and group counts
are 12 and 18; at $p=40$, $n=1200$ and group counts are 40 and 60. Each design has
eight independent replicates. Seconds are median complete-pipeline times.
Mean MSE targets the generating population mean in new groups; full Monte
Carlo standard errors and failures are retained in the raw summary.}
\end{table}
''')
    a=summary['overview']['stress']
    (DEST/'stress_findings.tex').write_text(f"The verified method completed {a['verified_complete']}/{a['cases']} extension paths, with {a['decisions']}\n"
          f"checked decisions and {a['pair_additions']} pair additions. All {a['replay_valid']} completed paths\n"
          f"passed replay; {a['identical_paths']}/{a['comparable']} comparable full local-pair paths agreed.\n"
          "Eight replicates per design are insufficient for precise rare-error estimates.\n"
          "These cases extend the tested dimension range jointly but do not establish\n"
          "a general high-dimensional scaling law.\n")
    native=json.loads((DEST/'native_validation.json').read_text())
    nrec=len(native['records']);points=sum(len(r['points']) for r in native['records'])
    maxgap=native['max_profile_loglike_gap'];maxdev=max(abs(r['native_minus_shared']) for r in native['records'])
    maxpred=max(r['max_prediction_difference'] for r in native['records']);conv=sum(r['native_converged'] for r in native['records'])
    def sci(x):
        m,e=f'{x:.2e}'.split('e')
        return '$'+m+r'\times10^{'+str(int(e))+'}$'
    (DEST/'native_findings.tex').write_text(f"An independent implementation check uses native statsmodels 0.14.6\n"
          f"mixed-model likelihoods~\\cite{{statsmodels}} on {nrec} full-data supports. Across {points}\n"
          f"common positive variance-ratio points, the largest log-likelihood discrepancy\n"
          f"was {sci(maxgap)}. Native local refits, initialized at the common engine's fitted\n"
          f"point, reported convergence in {conv}/{nrec} cases. The largest absolute\n"
          f"profile-deviance difference was {sci(maxdev)}, and the largest fitted-value\n"
          f"difference was {sci(maxpred)}. Warnings and all point comparisons are archived.\n"
          "This checks independent likelihood implementations and local refits; it is\n"
          "not a native selection-method speed comparison or an additional global-ML certificate.\n")
    figure(real,order)
    polish(summary,native,order)
    print(json.dumps(summary['overview'],indent=2))

def figure(real,order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),sharey=True,layout='constrained')
    for ax,ds,title in zip(axes,['carrots','cake'],['Consumer preferences','Cake breakage angle']):
        null=real[ds]['methods']['null_ml']['mse']
        vals=[real[ds]['methods'][m]['mse']/null for m in order]
        colors=['#087f8c' if m=='verified_pair' else '#496079' for m in order]
        ax.scatter(vals,np.arange(len(order)),c=colors,s=45,zorder=3)
        ax.axvline(1,color='#a3a3a3',ls='--',lw=1)
        ax.set_title(title,fontsize=11);ax.set_xlabel('MSE / intercept-only MSE',fontsize=10)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
        ax.set_yticks(range(len(order)),[LABELS[m] for m in order],fontsize=9.5)
        ax.tick_params(axis='x',labelsize=9)
    axes[0].invert_yaxis()
    fig.savefig(ROOT/'figures/csda_real_prediction.png',dpi=210)
    fig.savefig(ROOT/'figures/csda_real_prediction.svg')
    plt.close(fig)

def polish(summary,native,order):
    o=summary['overview'];real=summary['real']
    text=f'''# Manuskrypt w kierunku CSDA: wyniki i stan pracy

Marek Gałązka i Hanna Wdowicka • 14 września 2026

Przygotowano osobny manuskrypt **Verified likelihood-based screening for Gaussian mixed-model selection** w klasie Elsevier `elsarticle`. Poprzednia nota zachowuje pełną historię rozwoju i pozostaje uzupełnieniem technicznym. Główny tekst jest ułożony problemowo: motywacja, granice wiarygodności, certyfikowana ścieżka, implementacja, symulacje, zastosowania i ograniczenia.

## Wkład w stosunku do Applied Sciences

ARSS-JPS wybiera krótką listę kandydatów i stosuje testy ilorazu wiarygodności. Nowa praca bada inną, jawną regułę BIC, z kompletną rodziną pojedynczych dodatków i par. Nowością, której bronimy, jest konstrukcja sprawdzalnych decyzji przy ponownym estymowaniu wariancji: granice dla całej rodziny, wykluczenie nieograniczonych ogonów dziedziny oraz świadkowie pozwalający odtworzyć ścieżkę.

Dodano wniosek o **istnieniu skończonego certyfikatu przy dodatniej luce między najlepszym kandydatem a rywalami**. Wynik łączy dodatnie reszty projekcyjne, skończone pokrycie dziedziny, dokładność granic przy podziale i wystarczającą precyzję arytmetyki. Nie stanowi gwarancji zakończenia konkretnej implementacji w zadanym limicie czasu.

Nie przypisujemy nowości samemu profilowaniu ML, tożsamościom macierzowym ani ogólnej idei branch-and-bound. Gwarancja dotyczy wykonania reguły obliczeniowej, a nie prawdziwości wybranych zmiennych, globalnie najlepszego podzbioru czy kalibracji p-value po selekcji.

## Co wykonano w tym etapie

- Dwie analizy rzeczywiste: 1233 oceny konsumentów i 270 pomiarów kąta złamania ciasta.
- 14 analiz z selekcją wyłącznie wewnątrz treningu; każdy obserwowany wynik trafia dokładnie raz do testu.
- Dwa oddzielne dopasowania na pełnych danych do opisu wybranych zmiennych.
- 32 nowe symulacje przy jednoczesnym zwiększeniu liczby predyktorów i grup.
- Niezależne sprawdzenie natywnym silnikiem statsmodels na {len(native['records'])} wybranych modelach.

Certyfikowane nowe ścieżki: **{o['all']['verified_complete']}/{o['all']['cases']}**; sprawdzone decyzje: **{o['all']['decisions']}**. Wszystkie ukończone ścieżki przeszły odtwarzanie. Zgodność całych ścieżek z lokalną BIC z parami: **{o['all']['identical_paths']}/{o['all']['comparable']}**.

## Predykcja na danych rzeczywistych

| Procedura | Oceny konsumentów: MSE | Kąt złamania ciasta: MSE |
|---|---:|---:|
'''
    for m in order:text+=f"| {LABELS[m]} | {real['carrots']['methods'][m]['mse']:.4f} | {real['cake']['methods'][m]['mse']:.4f} |\n"
    text+='''
MSE obliczono z przewidywań poza próbą, w oryginalnych skalach odpowiedzi. W tabeli każdy model ocenia 1233 i 270 wierszy. Podziały mają wspólne treningi i grupy, dlatego nie traktujemy ich jako niezależnych replikacji i nie dopisujemy naiwnych przedziałów ufności.

**Oceny konsumentów:** losowe wyrazy wolne konsumenta i produktu; dziewięć przecięć trzech podziałów na każdym wymiarze. Nowe grupy testowe nie występują w treningu na żadnym wymiarze. Ośmiokolumnowy słownik obejmuje dwa zewnętrznie dostarczone deskryptory sensoryczne, płeć, wielkość gospodarstwa i ich wskazane interakcje. Dane źródłowe zawierają trzy brakujące preferencje; pominięto wyłącznie te wiersze, bez imputacji wyniku. Opublikowane składowe PCA traktujemy jako dostępne deskryptory produktu, a nie PCA ponownie trenowane w walidacji.

Model Gaussa dla skali 1–7 jest modelem roboczym. Certyfikat dotyczy tego celu obliczeniowego, nie trafności przybliżenia rozkładu. Różnice predykcyjne są niewielkie, a natywny elastic net ma najniższy MSE. Na pełnych danych BIC i ARSS wybierają `sens2`.

**Eksperyment z ciastami:** jeden efekt losowy dla `recipe:replicate`, zgodnie z dokumentacją doświadczenia; 45 partii. Pięć podziałów pozostawia całe partie poza treningiem, po trzy partie każdej receptury. Kandydaci to trzy potęgi temperatury i dwa wskaźniki receptury. Nie tworzono sztucznego drugiego efektu losowego. Na pełnych danych wszystkie procedury selekcji wskazują liniowy składnik temperatury. Reguły nie wymuszają hierarchii wielomianów i interakcji.

W obu zadaniach tuning elastic netu używa rozszerzonej siatki kar. Wszystkie wewnętrzne podziały i ich skalowanie są zapisane. Prognozy natywne i prognozy po ML dla tego samego wybranego zbioru pokazujemy osobno.

## Większe symulacje

| n | p | Liczby grup | Scenariusze | Powtórzenia |
|---:|---:|---|---|---:|
| 600 | 24 | 12 i 18 | sygnały, para supresyjna | po 8 |
| 1200 | 40 | 40 i 60 | sygnały, para supresyjna | po 8 |

'''
    text+=f"Zakończono {o['stress']['verified_complete']}/{o['stress']['cases']} ścieżek certyfikowanych; liczba sprawdzonych decyzji: {o['stress']['decisions']}. To małe rozszerzenie zakresu wymiarów, a nie badanie asymptotyczne. Pełne metryki, błędy Monte Carlo, czasy i ewentualne niepowodzenia są w `results/csda_validation/summary.json`.\n\n"
    text+='''## Wcześniejsze badanie pozostaje zachowane

300 głównych symulacji, 1186 decyzji, sześć scenariuszy i analiza wrażliwości siatki elastic netu pozostają bez zmian. W scenariuszu supresji obie zmienne odzyskano 50/50 razy z parami, 0/50 bez par i 50/50 w ARSS-JPS. Certyfikowana wersja miała medianowy czas 2,60 razy większy od ARSS, a lokalna BIC z parami potrzebowała medianowo 1,64 razy więcej czasu niż wersja certyfikowana. To różne porównania implementacyjne; nie łączymy ich z późniejszą siatką w jeden pozornie jednorodny wynik.

## Dopasowanie do CSDA

[Profil CSDA opisany przez CMStatistics](https://www.cmstatistics.org/CSDA_publications.php) obejmuje rozwój, ocenę i walidację algorytmów statystycznych oraz metody optymalizacji. To właściwe uzasadnienie wyboru czasopisma dla obecnej pracy. [Wskazówki dla autorów](https://www.cmstatistics.org/submission.php) zalecają `elsarticle`; użyto oryginalnej klasy z CTAN. Szczegółowa strona wydawcy Guide for Authors była niedostępna podczas sprawdzania, więc nie przypisano czasopismu niezweryfikowanego limitu stron ani szczegółowego trybu anonimizacji.

Pozostałe najważniejsze kwestie naukowe to natywny współczesny selektor penalizowanego LMM na wspólnej klasie modeli oraz niezależny przegląd dowodów i implementacji przez autorów. Nowy przykład jednego grupowania ułatwia uczciwe porównanie z `glmmsel`, ale tego pakietu jeszcze nie uruchomiono. Czasopismo może uznać obecny zakres numeryczny za zbyt wąski; nie podajemy pozornej procentowej szansy akceptacji.

## Pliki i odtworzenie

- `CSDA_manuscript.pdf` oraz `CSDA_manuscript.tex`: nowy główny manuskrypt.
- `CSDA_covariance_screening_note.pdf`: wcześniejsza nota z pełnymi szczegółami i historią pilotów.
- `run_csda_validation.py`: komplet nowych danych, podziałów i analiz.
- `verify_csda_native.py`: niezależne porównanie natywnego celu ML.
- `summarize_csda_validation.py`: automatyczne tabele, rysunek i ten raport.
- `CSDA_editorial_materials.md`: argumentacja dla redakcji, highlights i pozostałe konkretne uzupełnienia.

Autorstwo i afiliacje przepisano z dostarczonego artykułu. Końcowe dane bibliograficzne publikacji Applied Sciences wymagają uzupełnienia z wersji wydawniczej; nie zgadywano DOI. Dodano opis rzeczywistego zakresu pomocy ChatGPT/Codex: obejmowała tekst, rozumowanie matematyczne, kod i wykonanie eksperymentów, więc nie ograniczono deklaracji do samej korekty językowej. Ostateczne sprawdzenie i zatwierdzenie treści naukowej należy do autorów. Manuskrypt nie został wysłany do czasopisma.
'''
    (ROOT/'CSDA_wyniki_i_plan_PL.md').write_text(text)

if __name__=='__main__':main()
