"""Generate stage-five summaries from every recorded outcome, including failures."""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS=['verified_pair','local_pair','local_single','arss_jps','enet_ml']
LABELS={'verified_pair':'Verified BIC + pairs','local_pair':'Local BIC + pairs',
        'local_single':'Local BIC singles','arss_jps':'ARSS-JPS policy','enet_ml':'EN-CV + ML'}
SHORT={'verified_pair':'Verified','local_pair':'Local pairs','local_single':'Singles',
       'arss_jps':'ARSS-JPS','enet_ml':'EN + ML'}
SCENARIOS=['signals','weak','suppressor','null','correlated_null','group_aligned']
PL={'signals':'Sygnały','weak':'Słabe sygnały','suppressor':'Para supresyjna',
    'null':'Brak sygnału','correlated_null':'Brak sygnału, korelacja','group_aligned':'Predyktory grupowe'}
EN={'signals':'Signals','weak':'Weak signals','suppressor':'Suppressor',
    'null':'Null','correlated_null':'Correlated null','group_aligned':'Group-aligned'}
EVENTS={'false_selection_event','exact_recovery','pair_recovery'}
METRICS=['tpr','fdp','false_selection_event','exact_recovery','selected_n','mean_mse','response_mse','pair_recovery']


def estimate(values,event=False):
    v=np.asarray([x for x in values if x is not None],float);n=len(v)
    if not n:return {'n':0,'mean':None,'mcse':None}
    mean=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(n)) if n>1 else None
    out={'n':n,'mean':mean,'mcse':se}
    if event:
        z=1.959963984540054;den=1+z*z/n
        center=(mean+z*z/(2*n))/den
        half=z*math.sqrt(mean*(1-mean)/n+z*z/(4*n*n))/den
        out.update(events=int(v.sum()),wilson95=[max(0.,center-half),min(1.,center+half)])
    return out


def number(x,digits=3):return '--' if x is None else f'{x:.{digits}f}'


def main():
    root=Path(__file__).parent;dest=root/'results/comparison'
    protocol=json.loads((dest/'protocol.json').read_text())
    sensitivity_text=(root/'CSDA_enet_sensitivity_PL.md').read_text()
    frozen=json.loads((dest/'code_sha256.json').read_text())
    for name,digest in frozen.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,'Frozen benchmark code changed'
    records=[json.loads(f.read_text()) for f in sorted(dest.glob('*_[0-9]*.json'))]
    expected=len(protocol['scenarios'])*protocol['replicates_per_scenario']
    assert len(records)==expected,(len(records),expected)
    summary=[];paired=[]
    for scenario in SCENARIOS:
        data=[r for r in records if r['scenario']==scenario]
        assert len(data)==protocol['replicates_per_scenario']
        for method in METHODS:
            good=[r for r in data if r.get('metrics',{}).get(method) is not None]
            failed=[r for r in data if r not in good]
            row={'scenario':scenario,'method':method,'n_total':len(data),'n_complete':len(good),
                 'failure_statuses':dict(collections.Counter(r.get('results',{}).get(method,{}).get('status',r.get('worker_failure','unknown')) for r in failed))}
            for key in METRICS:row[key]=estimate([r['metrics'][method].get(key) for r in good],key in EVENTS)
            for key in ('native_mean_mse','native_response_mse'):
                row[key]=estimate([r['metrics'][method].get(key) for r in good])
            times=[r['results'][method]['pipeline_seconds'] for r in good]
            row['median_seconds']=statistics.median(times) if times else None
            row['p90_seconds']=float(np.quantile(times,.9)) if times else None
            row['min_seconds']=min(times) if times else None
            row['max_seconds']=max(times) if times else None
            row['mean_seconds']=estimate(times)
            row['mean_selection_fit_calls']=estimate([r['results'][method]['selection_fit_calls'] for r in good])
            row['pair_additions']=sum(r['results'][method].get('pair_additions') or 0 for r in good)
            row['datasets_with_stationarity_flags']=sum(r['results'][method].get('local_stationarity_failures',0)>0 for r in good)
            row['enet_convergence_warnings']=sum(r['results'][method].get('cv_warning_count',0) for r in good)
            summary.append(row)
        for other in METHODS[1:]:
            good=[r for r in data if r.get('metrics',{}).get('verified_pair') is not None and r.get('metrics',{}).get(other) is not None]
            row={'scenario':scenario,'other_method':other,'paired_n':len(good),'difference_direction':'verified minus other'}
            for key in METRICS:
                row[key]=estimate([r['metrics']['verified_pair'][key]-r['metrics'][other][key]
                                   for r in good if r['metrics']['verified_pair'][key] is not None and r['metrics'][other][key] is not None])
            ratios=[r['results'][other]['pipeline_seconds']/r['results']['verified_pair']['pipeline_seconds'] for r in good]
            row['median_other_over_verified_time']=statistics.median(ratios) if ratios else None
            row['same_support']=sum(r['results']['verified_pair']['selected']==r['results'][other]['selected'] for r in good)
            paired.append(row)
    complete=[r for r in records if r.get('results',{}).get('verified_pair',{}).get('status')=='complete']
    compared=[r for r in records if r.get('local_comparison') is not None]
    overview={'datasets':len(records),'verified_complete':len(complete),
              'verified_replay_valid':sum(r['verified_replay']['valid'] for r in complete),
              'verified_decisions':sum(len(r['results']['verified_pair']['stages']) for r in complete),
              'verified_pair_additions':sum(r['results']['verified_pair']['pair_additions'] for r in complete),
              'null_pair_additions':sum(r['results']['verified_pair']['pair_additions'] for r in complete if r['scenario'] in ('null','correlated_null')),
              'local_pair_comparisons':len(compared),
              'same_local_pair_full_paths':sum(r['local_comparison']['same_full_path'] for r in compared),
              'same_local_pair_supports':sum(r['local_comparison']['same_support'] for r in compared),
              'method_complete':{m:sum(r.get('metrics',{}).get(m) is not None for r in records) for m in METHODS},
              'median_time_ratios':{m:statistics.median(r['results'][m]['pipeline_seconds']/r['results']['verified_pair']['pipeline_seconds']
                    for r in records if r.get('metrics',{}).get(m) is not None and r.get('metrics',{}).get('verified_pair') is not None) for m in METHODS[1:]},
              'definition_time_ratios':'other method divided by verified, paired complete runs',
              'median_verified_over_time':{m:statistics.median(r['results']['verified_pair']['pipeline_seconds']/r['results'][m]['pipeline_seconds']
                    for r in records if r.get('metrics',{}).get(m) is not None and r.get('metrics',{}).get('verified_pair') is not None) for m in METHODS[1:]},
              'total_seconds':{m:sum(r.get('results',{}).get(m,{}).get('pipeline_seconds',0.) for r in records) for m in METHODS},
              'qualification':'300 new fixed-grid simulations. Local agreement is an observation, not a global-ML oracle. Completion denominators are explicit.'}
    (dest/'summary.json').write_text(json.dumps({'overview':overview,'by_scenario_method':summary,'paired':paired},indent=2)+'\n')
    lookup={(r['scenario'],r['method']):r for r in summary}
    sup=lookup['suppressor','verified_pair'];single=lookup['suppressor','local_single'];arss=lookup['suppressor','arss_jps']
    matched=next(r for r in paired if r['scenario']=='suppressor' and r['other_method']=='local_single')
    interpretation=(
        f"In the suppressor scenario, both active members of the pair were recovered in {sup['pair_recovery']['events']}/{sup['pair_recovery']['n']} verified runs, "
        f"{single['pair_recovery']['events']}/{single['pair_recovery']['n']} local-single runs and {arss['pair_recovery']['events']}/{arss['pair_recovery']['n']} ARSS-JPS runs. "
        f"The paired population-mean MSE difference, verified minus singles, was {matched['mean_mse']['mean']:.3f} "
        f"with Monte Carlo standard error {matched['mean_mse']['mcse']:.3f}. "
        "This contrast assesses the benefit of the pair policy; it does not isolate a statistical benefit from numerical certification.\n\n"
        f"Across jointly completed datasets, median verified/ARSS-JPS elapsed time was {overview['median_verified_over_time']['arss_jps']:.2f}. "
        "The archived shortlist policy and the all-family verifier therefore have different measured costs as well as different guarantees and stopping thresholds. "
        "The numerical evidence should not be summarized as universal superiority of one selector.\n"
    )
    (dest/'comparison_interpretation.tex').write_text(interpretation)
    tables=[]
    for scenarios,title,label in [(SCENARIOS[:3],'Selection and prediction on signal designs','tab:comparison_signal'),
                                  (SCENARIOS[3:],'Null and group-aligned designs','tab:comparison_null')]:
        tex=[r'\begin{table}[H]\centering\small',r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{llrrrrr}',r'\toprule',
             r'Design & Method & Done & TPR & FDR & Exact & Mean MSE \\',r'\midrule']
        for scenario in scenarios:
            for j,method in enumerate(METHODS):
                r=lookup[scenario,method]
                tex.append(f"{EN[scenario] if j==0 else ''} & {SHORT[method]} & {r['n_complete']}/{r['n_total']} & {number(r['tpr']['mean'],2)} & {number(r['fdp']['mean'],2)} & {number(r['exact_recovery']['mean'],2)} & {number(r['mean_mse']['mean'],3)}"+r' \\')
            if scenario!=scenarios[-1]:tex.append(r'\midrule')
        tex += [r'\bottomrule',r'\end{tabular}',r'\caption{'+title+r'. Entries use completed runs; TPR is undefined under the null. FDR is mean false-discovery proportion, Exact is full-support recovery, and Mean MSE targets the fixed population mean in entirely new groups. All selected supports receive the same type of ML coefficient refit. Full Monte Carlo standard errors and event intervals are retained in the raw summary.}\label{'+label+'}',r'\end{table}']
        tables.append('\n'.join(tex))
    (dest/'comparison_selection_tables.tex').write_text('\n\n'.join(tables)+'\n')
    tex=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',
         r'Design & Verified & Local pairs & Singles & ARSS-JPS & EN + ML \\',r'\midrule']
    for scenario in SCENARIOS:
        tex.append(EN[scenario]+' & '+' & '.join(number(lookup[scenario,m]['median_seconds'],3) for m in METHODS)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\caption{Median end-to-end seconds on completed runs: selection, tuning where applicable, all certificate construction/checking, and a final ML coefficient refit. Independent replay, held-out scoring, imports and file writes are excluded. These procedures have different search/stopping policies; the ratios are descriptive, not isolated backend speedups.}\label{tab:comparison_time}',r'\end{table}']
    (dest/'comparison_time_table.tex').write_text('\n'.join(tex)+'\n')
    macros={'ComparisonCases':str(overview['datasets']),'ComparisonComplete':str(overview['verified_complete']),
            'ComparisonSteps':str(overview['verified_decisions']),'ComparisonPairs':str(overview['verified_pair_additions']),
            'ComparisonAgreed':str(overview['same_local_pair_full_paths']),
            'ComparisonLocalRatio':number(overview['median_time_ratios']['local_pair'],2)}
    (dest/'comparison_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    # Scientific figure: uncertainty calculated over independent datasets.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#087f8c','#55748d','#8866a7','#ba6139','#607e48'];markers=['o','s','^','D','v']
    show=['verified_pair','local_single','arss_jps','enet_ml']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.2))
    panels=[('tpr',SCENARIOS[:3],'A  Signal recovery','Mean TPR',False),
            ('false_selection_event',SCENARIOS[3:5],'B  Any false selection','Event rate',True),
            ('mean_mse',SCENARIOS,'C  Prediction in new groups','Population-mean MSE',False),
            ('median_seconds',SCENARIOS,'D  Complete pipeline','Median seconds',False)]
    for ax,(key,scenarios,title,ylabel,event) in zip(axes.flat,panels):
        positions=np.arange(len(scenarios));offsets=np.linspace(-.22,.22,len(show))
        for off,method in zip(offsets,show):
            i=METHODS.index(method);rr=[lookup[sc,method] for sc in scenarios]
            if key=='median_seconds':
                yy=[r[key] for r in rr];err=None
            else:
                yy=[r[key]['mean'] for r in rr]
                if event:err=[[v-r[key]['wilson95'][0] for v,r in zip(yy,rr)],[r[key]['wilson95'][1]-v for v,r in zip(yy,rr)]]
                else:err=[r[key]['mcse'] for r in rr]
            ax.errorbar(positions+off,yy,yerr=err,fmt=markers[i],color=colors[i],ms=4,capsize=2,label=SHORT[method],lw=1)
        ax.set_xticks(positions,[EN[s] for s in scenarios],rotation=20 if len(scenarios)>3 else 0,ha='right' if len(scenarios)>3 else 'center')
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        if key in ('tpr','false_selection_event'):ax.set_ylim(-.03,1.06)
        else:ax.set_ylim(bottom=0)
        if key=='tpr':ax.legend(loc='lower left',frameon=False,fontsize=8)
    fig.tight_layout(h_pad=2.2,w_pad=2)
    fig.savefig(root/'figures/comparison_methods.png',dpi=210)
    fig.savefig(root/'figures/comparison_methods.svg');plt.close(fig)
    # Current Polish narrative is rebuilt from the recorded numbers.
    def r(sc,m):return lookup[sc,m]
    suppressor_rows=[]
    for m in METHODS:
        z=r('suppressor',m)
        suppressor_rows.append(f"| {LABELS[m]} | {z['pair_recovery']['events']}/{z['pair_recovery']['n']} | {number(z['tpr']['mean'],3)} | {number(z['fdp']['mean'],3)} | {number(z['mean_mse']['mean'],3)} |")
    allrows=[]
    for sc in SCENARIOS:
        for m in METHODS:
            z=r(sc,m)
            allrows.append(f"| {PL[sc]} | {LABELS[m]} | {z['n_complete']}/{z['n_total']} | {number(z['tpr']['mean'],3)} | {number(z['fdp']['mean'],3)} | {number(z['exact_recovery']['mean'],3)} | {number(z['response_mse']['mean'],3)} | {number(z['median_seconds'],3)} |")
    nullrows=[]
    for sc in ['null','correlated_null']:
        for m in METHODS:
            z=r(sc,m)['false_selection_event'];lo,hi=z['wilson95']
            nullrows.append(f"| {PL[sc]} | {LABELS[m]} | {z['events']}/{z['n']} | {100*z['mean']:.1f}% [{100*lo:.1f}%, {100*hi:.1f}%] |")
    nativerows=[]
    native_tex=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrr}',r'\toprule',
                r'Design & Native EN mean MSE & EN + ML mean MSE \\',r'\midrule']
    for sc in SCENARIOS:
        z=r(sc,'enet_ml');a=z['native_mean_mse'];b=z['mean_mse']
        nativerows.append(f"| {PL[sc]} | {a['mean']:.3f} ± {a['mcse']:.3f} | {b['mean']:.3f} ± {b['mcse']:.3f} |")
        native_tex.append(EN[sc]+f" & {a['mean']:.3f} ({a['mcse']:.3f}) & {b['mean']:.3f} ({b['mcse']:.3f})"+r' \\')
    native_tex += [r'\bottomrule',r'\end{tabular}',r'\caption{Native shrunken elastic-net predictions and the same selected support after ML refitting. Values are population-mean MSE; parentheses give one Monte Carlo standard error. The support and its tuning are the same in both columns.}\label{tab:comparison_native}',r'\end{table}']
    (dest/'comparison_native_table.tex').write_text('\n'.join(native_tex)+'\n')
    md=f'''# Rozwinięcie artykułu Applied Sciences w kierunku CSDA — etap 5

Materiał roboczy dla Marka Gałązki • 14 września 2026

**Wygenerowano i przeanalizowano {overview['datasets']} nowych zbiorów, porównując pięć procedur.** Badanie obejmuje po 50 powtórzeń sześciu scenariuszy. Oceniamy poprawność wykonania reguły selekcji, wykrywanie efektów, fałszywe wybory, predykcję na nowych grupach oraz czas całej procedury.

Certyfikowana metoda zakończyła **{overview['verified_complete']}/{overview['datasets']} ścieżek**, obejmujących **{overview['verified_decisions']} decyzji** i **{overview['verified_pair_additions']} dodań par**. Odtworzenie wszystkich ukończonych ścieżek potwierdziło ich świadków. W **{overview['same_local_pair_full_paths']}/{overview['local_pair_comparisons']}** porównywalnych przypadkach pełna sekwencja decyzji zgadzała się z lokalną procedurą BIC z parami. Ta zgodność jest wynikiem eksperymentu; obliczenia lokalne nie stają się przez nią niezależnym wzorcem globalnego ML.

Najbardziej czytelny kontrast dotyczy pary supresyjnej: oba składniki odzyskano w **{sup['pair_recovery']['events']}/{sup['pair_recovery']['n']}** przebiegów certyfikowanych, **{single['pair_recovery']['events']}/{single['pair_recovery']['n']}** przebiegów BIC bez par i **{arss['pair_recovery']['events']}/{arss['pair_recovery']['n']}** przebiegów ARSS-JPS. Sparowana różnica błędu względem prawdziwej średniej (wersja certyfikowana minus BIC bez par) wyniosła **{matched['mean_mse']['mean']:.3f}**, z błędem Monte Carlo **{matched['mean_mse']['mcse']:.3f}**. To miara korzyści z rozpatrywania par, a nie z samego certyfikowania rachunków.

Mediana czasu wersji certyfikowanej podzielonego przez czas ARSS-JPS wyniosła **{overview['median_verified_over_time']['arss_jps']:.2f}**. Wkład pracy należy więc opierać na sprawdzalnej gwarancji decyzji i jawnej ocenie jej kosztu; te wyniki nie uzasadniają obietnicy, że certyfikowana selekcja zawsze jest najszybsza lub statystycznie najlepsza.

## Porównywane procedury

| Metoda | Co porównujemy |
|---|---|
| Verified BIC + pairs | Nasza pełna ścieżka BIC, wszystkie pojedyncze dodatki i wszystkie pary, dziedzina wszystkich nieujemnych ilorazów wariancji, świadkowie Arb. |
| Local BIC + pairs | Ta sama reguła wyboru, z osobnym lokalnym dopasowaniem każdego kandydata. Służy do porównania kosztu i zgodności decyzji. |
| Local BIC singles | BIC bez etapu par. Pozwala ocenić skutki ratowania zamaskowanych efektów. |
| ARSS-JPS policy | Niezmieniona funkcja wyszukiwania z repozytorium artykułu, ze wspólnym adapterem ML: α=0,01, K=3, próg korelacji 0,60, L=5. |
| EN-CV + ML | Elastic net wybiera zmienne po walidacji wewnętrznej; następnie wybrany zbiór jest dopasowany jako LMM z oboma efektami losowymi. |

ARSS zachowuje reguły z publikacji, ale **nie jest tu uruchomieniem oryginalnego silnika statsmodels**. Wykorzystano niezmienioną funkcję `arss_family` z commitu `0cdcf976178be7ed00af4b0737a7ed97540786a4`; zapisano skróty kodu i licencję. Wszystkie dopasowania ML w tym porównaniu korzystają z tego samego celu i podstawowego silnika numerycznego. Użycie profilu R zamiast V skaluje system wyników przesiewania wspólną dodatnią stałą, bez zmiany rankingów i korelacji.

Elastic net jest zewnętrznym punktem odniesienia opartym na scikit-learn, z kryterium nastawionym na predykcję. Sam etap przesiewania nie modeluje kowariancji efektów losowych. Określenie „EN-CV + ML” oznacza tę konkretną procedurę dwuetapową; nie przedstawiamy jej jako pakietu penalizowanego LMM. Surowe wyniki zawierają również błędy natywnych, skurczonych prognoz elastic netu.

## Protokół

Każdy zbiór treningowy ma n=300, p=12 oraz 6 i 8 grup. Zbiór testowy ma 1200 obserwacji i całkowicie nowe efekty losowe dla 24 i 32 grup. Predyktory są centrowane i skalowane wyłącznie parametrami treningowymi; współczynniki generujące odpowiedź podano w tej skali. Scenariusze obejmują zwykłe sygnały, słabe sygnały, parę supresyjną, dwa projekty bez sygnału i predyktory silnie powiązane z grupowaniem.

Selekcja i dopasowanie kończą się przed użyciem odpowiedzi testowych. Strojenie elastic netu ma trzy wewnętrzne podziały: w walidacji występują nowe grupy **na obu marginesach**. Komórki łączące grupę walidacyjną z treningową są pomijane w danym podziale. Zbiory walidacyjne nie pokrywają więc wszystkich obserwacji. Skalowanie powtarza się wewnątrz każdego podziału; stroimy 13 wartości kary i trzy udziały L1. Nie są wykorzystywane dane testowe.

Każda wybrana struktura otrzymuje końcowy refit ML dla prognoz. Prognoza jest marginalna: wyraz wolny i efekty stałe, bez oszacowań nieznanych nowych efektów losowych. `Mean MSE` porównuje ją z prawdziwą średnią populacyjną; `Response MSE` dodatkowo zawiera efekty nowych grup i szum. Końcowy refit współczynników jest lokalny i nie rozszerza zakresu certyfikatu wyboru.

Jeden nowy proces obsługuje zbiór, metody mają osobne obiekty modeli i zmienianą kolejność wykonania. BLAS używa jednego wątku. Czas obejmuje selekcję, strojenie, certyfikację oraz końcowy refit. Dodatkowe odtwarzanie świadków, ocena na teście, importy i zapis plików pozostają poza pomiarem. W tym etapie nie raportujemy osobnego szczytu pamięci każdej metody.

## Pary supresyjne

| Metoda | Odzyskane oba składniki pary | Średnia TPR | FDR | Mean MSE |
|---|---:|---:|---:|---:|
{chr(10).join(suppressor_rows)}

Obecność obu zmiennych w wyniku i dodanie ich jednocześnie to różne miary. Pierwszą podaje tabela; liczbę faktycznych ratowań parami zapisano osobno. Porównanie z wersją bez par pozwala ocenić wpływ samego rozszerzenia rodziny modeli.

## Brak sygnału: fałszywe wybory

Poniżej częstość wybrania co najmniej jednej zmiennej oraz 95-procentowy przedział Wilsona. Przy 50 powtórzeniach nawet zero zdarzeń nie dowodzi zerowego ryzyka; górna granica wynosi około 7,1%.

| Scenariusz | Metoda | Liczba zdarzeń | Częstość i przedział 95% |
|---|---|---:|---:|
{chr(10).join(nullrows)}

W scenariuszach bez żadnego sygnału liczba akceptacji par przez certyfikowaną procedurę wyniosła **{overview['null_pair_additions']}**. Poprawny certyfikat porównania BIC nie oznacza więc, że dodane efekty są prawdziwe.

ARSS używa α=0,01, a BIC kary log(300); te progi nie oznaczają tego samego poziomu błędu. Elastic net minimalizuje błąd walidacyjny. Różnice mocy i fałszywych wyborów należy interpretować razem z tymi regułami, bez deklarowania jednakowej kalibracji FWER.

## Pełne zestawienie

FDR oznacza średnią z FP/max(1, liczba wybranych zmiennych), TPR — udział wykrytych prawdziwych efektów, a Exact — odsetek dokładnego odzyskania zbioru. TPR pod hipotezą całkowitego braku sygnału pozostaje niezdefiniowana. Wyniki jakości dotyczą ukończonych przebiegów; kolumna Done podaje mianownik.

| Scenariusz | Metoda | Done | TPR | FDR | Exact | Response MSE | Mediana sekund |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(allrows)}

Dla elastic netu osobno pokazujemy wpływ końcowego refitu. Poniżej błąd wobec prawdziwej średniej ± jeden błąd Monte Carlo. Obie kolumny używają tego samego wybranego zbioru; zmienia się estymacja współczynników i uwzględnienie struktury mieszanej.

| Scenariusz | Natywne prognozy elastic netu | Prognozy po reficie ML |
|---|---:|---:|
{chr(10).join(nativerows)}

Mediana sparowanego ilorazu czasu lokalnej procedury z parami do wersji certyfikowanej wynosi **{overview['median_time_ratios']['local_pair']:.2f}**. Jest to porównanie kompletnych implementacji: lokalne dopasowania rozszerzają obszar, gdy rozwiązanie dotyka granicy, a weryfikator konstruuje obszary na podstawie ograniczeń ogonowych. Nie jest to wyizolowany efekt samej arytmetyki Arb.

Pełny `results/comparison/summary.json` zawiera błędy Monte Carlo, przedziały Wilsona, sparowane różnice miar między metodami, czasy i wszystkie mianowniki. Jednostką niepewności jest niezależny zbiór symulacyjny, nie pojedyncza obserwacja testowa. Żaden wynik nierozstrzygnięty nie został potraktowany jako poprawnie zakończona selekcja ani usunięty z liczby zaplanowanych zadań.

{sensitivity_text}

## Weryfikacja i ograniczenia

Porównano funkcję ARSS ze źródłowym skrótem oraz z niezależnym zapisem reguły na małych przykładach, w tym z ratowaniem pary. Wyniki przesiewania i współczynniki GLS porównano z pełną macierzą kowariancji i niezależnym wybielaniem. Sprawdzono rozdzielenie grup w walidacji, definicje miar oraz niewykorzystywanie danych testowych przez funkcje selekcji. Każda ukończona ścieżka certyfikowana przeszła ponowne sprawdzenie świadków poza pomiarem czasu.

To większe badanie symulacyjne, ale nadal ma p=12 i niewiele grup. Nie łączy ono powtarzanej oceny statystycznej z największymi rozmiarami z etapu 4. Pięćdziesiąt powtórzeń na scenariusz nie wystarcza do dokładnego oszacowania rzadkich błędów. Nie jest to prerejestracja ani niezależna walidacja zewnętrzna; sześć prób technicznych o innych seedach wyłączono z badania. Metod i progów nie zmieniano na podstawie wyników tej siatki.

Współczesny `glmmsel` sprawdzono w [instrukcji pakietu](https://cran.r-project.org/web/packages/glmmsel/glmmsel.pdf): interfejs używa jednego wektora klastrów oraz selekcji efektów stałych i losowych. Bezpośrednie podstawienie dwóch skrzyżowanych grup do tego interfejsu nie odtwarza naszego modelu. Pakietu nie uruchamiano i nie umieszczono go w tabelach. Dokumentację bazowego estymatora opisuje [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).

**Najważniejszy dalszy krok do CSDA:** zastosowanie całej selekcji wewnątrz treningowych podziałów danych rzeczywistych oraz porównanie z wyspecjalizowanym penalizowanym LMM na wspólnej, obsługiwanej klasie modeli. Obecne wyniki pozwalają oceniać cenę certyfikacji i rolę par; nie uzasadniają automatycznie przewagi nad wszystkimi współczesnymi metodami.

## Odtworzenie

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 verify_comparison_methods.py
python3 run_comparison_experiments.py
python3 run_enet_sensitivity.py
python3 summarize_enet_sensitivity.py
python3 summarize_comparison_results.py
```

Runner wznawia istniejące pliki wyłącznie przy zgodności protokołu, danych i zamrożonych skrótów kodu. Surowe NPZ zawierają dane treningowe i testowe, a JSON — wyniki wszystkich metod oraz świadków. Starsze wyniki zachowano w osobnych katalogach i historycznych podsumowaniach.
'''
    (root/'CSDA_wyniki_i_plan_PL.md').write_text(md)
    print(json.dumps(overview,indent=2))


if __name__=='__main__':main()
