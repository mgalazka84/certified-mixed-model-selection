"""Rebuild tables, figures, and the Polish summary from the recorded pilot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent
    results=root/'results';figures=root/'figures';figures.mkdir(exist_ok=True)
    p=pd.read_csv(results/'pilot.csv',keep_default_na=False)
    scale=pd.read_csv(results/'scaling.csv',keep_default_na=False)
    top3=pd.read_csv(results/'top3_diagnostic.csv',keep_default_na=False)
    labels={'separated':'Strong signal','weak':'Weak signals','group_aligned':'Group-aligned',
            'unbalanced':'Unbalanced groups','boundary':'Zero variance','null':'Complete null','near_tie':'Near tie'}
    order=list(labels)
    summary=p.groupby('scenario',sort=False).agg(cases=('status','size'),
        resolved=('status',lambda x:(x=='separated').sum()),fits=('full_candidate_fits','mean'),
        splits=('splits','median'),speed=('speedup','median'),
        screen_time=('screen_seconds','median'),reference_time=('reference_seconds','median')).reindex(order)
    lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',
           r'Scenario & Resolved & Mean fits & Median splits & Median time ratio\\',r'\midrule']
    # Five displayed columns.
    lines[1]=r'\begin{tabular}{lrrrr}'
    for key,row in summary.iterrows():
        lines.append(f"{labels[key]} & {int(row.resolved)}/{int(row.cases)} & {row.fits:.1f} & {row.splits:.0f} & {row.speed:.2f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Main pilot: $n=400$, $p=30$, five datasets per scenario. Time ratio is all-candidate multi-start time divided by screening time. The comparator fits 30 candidates per dataset.}',r'\end{table}']
    (results/'main_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Scenario & $p$ & Candidate fits & Splits & Time ratio\\',r'\midrule']
    for _,row in scale.iterrows():
        lines.append(f"{labels[row.scenario]} & {int(row.p)} & {int(row.full_candidate_fits)} & {int(row.splits)} & {row.speedup:.2f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Larger diagnostics, $n=1000$. One dataset per configuration; all six selections agreed with the all-candidate comparator.}',r'\end{table}']
    (results/'scaling_table.tex').write_text('\n'.join(lines)+'\n')
    ratio=top3.certified_over_top3_time.median()
    grouped=top3.groupby('scenario').certified_over_top3_time.median()
    paragraph=(f"A separate diagnostic refitted the three highest-scoring GLS candidates and selected the best among them. "
      f"This fixed top-three heuristic also agreed with all 35 comparator choices. Its median screening-to-heuristic time ratio was {ratio:.3f} "
      f"over the pilot, but the certificate method was slower in the difficult subsets: median ratios were {grouped['near_tie']:.2f} "
      f"for near ties, {grouped['null']:.2f} for the complete null, and {grouped['group_aligned']:.2f} for group-aligned predictors. "
      "This is not the full published ARSS algorithm: it has no significance-based expansion or pair rescue. "
      "It was timed in a separate run, so these ratios are descriptive overhead diagnostics rather than matched performance estimates. "
      "The formal exclusion guarantee is the proposed distinction; universal speed superiority over a short heuristic is not claimed.\n")
    (results/'top3_text.tex').write_text(paragraph)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#19384c','text.color':'#19384c','savefig.bbox':'tight'})
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[.92,1.25]})
    yy=np.arange(len(order))
    ax[0].barh(yy-.15,np.repeat(30,len(order)),height=.28,color='#ced8de',label='All candidates')
    ax[0].barh(yy+.15,summary.fits,height=.28,color='#087f8c',label='With bounds')
    ax[0].set_yticks(yy,[labels[k] for k in order]);ax[0].invert_yaxis()
    ax[0].set_xlabel('Separate candidate optimizations')
    ax[0].set_title('35 one-step datasets, 30 candidates each',loc='left',fontweight='bold',fontsize=10)
    ax[0].legend(frameon=False,fontsize=8,loc='lower right')
    for j,k in enumerate(order):
        vals=p.loc[p.scenario==k,'speedup'].to_numpy()
        ax[1].scatter(vals,np.repeat(j,len(vals))+np.linspace(-.12,.12,len(vals)),s=23,color='#087f8c',alpha=.75)
        ax[1].plot([np.median(vals)],[j],'|',color='#19384c',markersize=13,markeredgewidth=2)
    ax[1].set_yticks(yy,[]);ax[1].invert_yaxis();ax[1].axvline(1,color='#9aa9b2',ls='--',lw=1)
    ax[1].set_xlabel('All-candidate time / time with bounds')
    ax[1].set_title('Recorded runtime ratios',loc='left',fontweight='bold',fontsize=10)
    ax[1].set_xlim(0,18);ax[1].grid(axis='x',color='#e6ecef',lw=.7)
    fig.tight_layout(w_pad=2)
    fig.savefig(figures/'pilot_summary.pdf');fig.savefig(figures/'pilot_summary.png',dpi=170)
    plt.close(fig)
    reversal=json.loads((results/'rank_reversals.json').read_text())[0]
    q=np.array(reversal['q_scores'])[:2]
    base=reversal['screen']['base_feasible_f']
    gain=base-np.array(reversal['full_refit_objectives'])[:2]
    fig,ax=plt.subplots(1,2,figsize=(7.5,2.9))
    for a,values,title in zip(ax,[q,gain],['Fixed-covariance GLS score','Gain after covariance refitting']):
        a.bar(['$x_1$','$x_2$'],values,color=['#087f8c','#c59032'],width=.55)
        for j,v in enumerate(values):a.text(j,v+.025*max(values),f'{v:.3f}',ha='center',fontsize=9)
        a.set_ylim(0,1.2*max(values));a.set_title(title,fontsize=10);a.set_ylabel('Larger is better')
    fig.tight_layout(w_pad=2)
    fig.savefig(figures/'rank_reversal.pdf');fig.savefig(figures/'rank_reversal.png',dpi=170)
    plt.close(fig)
    env=json.loads((results/'environment.json').read_text())
    validation=json.loads((results/'validation.json').read_text())
    md=f'''# CSDA: ograniczenie błędu screeningu i pierwszy prototyp

Stan na 14 września 2026 r. Materiał roboczy do rozwinięcia ARSS-JPS.

## Co zostało zrobione

Wyprowadzono i opisano pełne dowody dla modeli Gaussa o kowariancji
`V = sigma² (I + lambda_1 A_1 + ... + lambda_q A_q)`, przy ustalonych dodatnio
półokreślonych macierzach `A_k`. Obejmują one krzyżowe efekty losowe.

1. Udowodniono wypukłość logarytmu resztowej sumy kwadratów GLS po dopasowaniu efektów stałych.
2. Połączono tę własność z wklęsłością logarytmu wyznacznika kowariancji. Powstało obliczalne ograniczenie funkcji celu na całym prostokątnym zakresie wariancji.
3. Otrzymano żądane ograniczenie `Q_C <= D_C <= Q_C + B_C`, gdy aktualny model jest globalnie dopasowany w rozważanej dziedzinie. Dla dopasowania lokalnego analogiczne ograniczenie odnosi się do jawnie wskazanego, wykonalnego modelu bazowego.
4. Udowodniono, że błąd granicy na pojedynczym prostokącie jest ograniczony przez stałą razy kwadrat jego średnicy, przy podanym ograniczeniu hesjanu.
5. Udowodniono dodatkowy warunek pozwalający wykluczyć wszystkie nieograniczone zakresy wariancji. Wymaga on dodatniej reszty po rzutowaniu odpowiedzi na przestrzeń efektów stałych danego kandydata i wszystkich kolumn efektów losowych.
6. Zaimplementowano wybór pojedynczego najlepszego kandydata z opcją jawnego wyniku `unresolved`, gdy dowód rozdzielenia nie mieści się w budżecie.

Pełne sformułowania, założenia i dowody znajdują się w `CSDA_covariance_screening_note.pdf`.

## Jak działa algorytm

Najpierw oblicza tani ranking GLS. Dopasowuje osobno najlepszego kandydata,
a dla pozostałych wyznacza dolne granice funkcji celu po reestymacji wariancji.
Jeśli nawet najlepszy dopuszczalny wynik rywala jest gorszy od już uzyskanego
wyniku, rywal zostaje wykluczony. W niejasnych przypadkach algorytm dzieli zakres
wariancji na mniejsze części i zacieśnia granice. Obliczenia kowariancji są
współdzielone przez kandydatów.

## Wyniki głównego pilotażu

- 35 zbiorów: 7 scenariuszy po 5 realizacji, 400 obserwacji i 30 kandydatów.
- {int(p.agreement.sum())}/35 wyborów zgodnych z dopasowaniem wszystkich kandydatów z wielu punktów startowych.
- 36 odrębnych dopasowań modeli kandydujących zamiast 1050 (redukcja 96,57%). Nie jest to liczba wszystkich obliczeń wiarygodności: screening wykonuje wspólne obliczenia w wielu punktach wariancji.
- Mediana przyspieszenia względem tej implementacji pełnego przeglądu: {p.speedup.median():.2f} razy; zakres {p.speedup.min():.2f}-{p.speedup.max():.2f} razy.
- Brak wykrytych naruszeń granic. {validation['dense_box_bound_checks']} sprawdzeń korzystało z niezależnego obliczenia z pełną macierzą kowariancji.

| Scenariusz | Rozstrzygnięte | Średnia liczba odrębnych dopasowań | Mediana przyspieszenia względem pełnego przeglądu |
|---|---:|---:|---:|
'''
    pl_labels={'separated':'Wyraźny sygnał','weak':'Słabsze sygnały','group_aligned':'Predyktory związane z grupami',
               'unbalanced':'Niezrównoważone grupy','boundary':'Zerowa wariancja','null':'Brak efektów','near_tie':'Prawie równe wyniki'}
    for key,row in summary.iterrows():md+=f'| {pl_labels[key]} | {int(row.resolved)}/5 | {row.fits:.1f} | {row.speed:.2f} |\n'
    md+='''
W sześciu większych próbach (1000 obserwacji, 100 lub 300 kandydatów) również
uzyskano rozstrzygnięcia zgodne z pełnym przeglądem. Są to pojedyncze próby dla
każdej konfiguracji, nie ustalona charakterystyka skalowalności. Liczba kolumn
efektów losowych wciąż wynosiła tylko 22.

W jednym przypadku lider GLS zmienił się po reestymacji wariancji:
`Q(x_2)=9,61577 > Q(x_1)=7,99677`, ale po pełnym dopasowaniu lepszy był `x_1`.
Prototyp znalazł tę zmianę i rozstrzygnął wybór poprawnie. Zwykła lista trzech
kandydatów również obejmowała obie te zmienne.

Sprawdzono także dokładny remis. Program nie ogłosił nieistniejącej ścisłej
przewagi: po 250 podziałach pozostawił dwóch kandydatów i zwrócił `unresolved`.

W dwóch przykładach zakres wariancji wyznaczony z danych pozwolił objąć argumentem
całą nieograniczoną dziedzinę nieujemnych ilorazów wariancji. Górne granice wyniosły
około (249,881; 65,039) i (289,702; 72,687). Potrzeba było odpowiednio 427 i 183 podziałów.

## Ile kosztuje gwarancja

'''
    md+=f'''Dodatkowa heurystyka dopasowująca tylko trzech liderów GLS również wskazała poprawnego
kandydata we wszystkich 35 próbach. Mediana stosunku czasu metody z ograniczeniami
do czasu tej heurystyki wyniosła {ratio:.3f}. W trudniejszych scenariuszach metoda z
ograniczeniami była jednak wolniejsza: około {grouped['near_tie']:.2f} raza przy prawie
równych wynikach i {grouped['null']:.2f} raza przy braku efektów. Te dodatkowe pomiary
wykonano w osobnej serii; nie stanowią porównania pełnych procedur ARSS-JPS.

Wyróżnikiem rozwijanej metody jest możliwość uzasadnienia pominięcia kandydatów.
Nie wykazano powszechnej przewagi szybkości nad dobrze dobraną krótką listą.

## Granice obecnego wyniku

- Dowody są matematyczne, dla dokładnej arytmetyki. Kod używa float64 i buforów numerycznych; nie jest formalnie zweryfikowany arytmetyką przedziałową.
- Główny pilotaż sprawdza **jeden krok wyboru**, a nie całą procedurę selekcji ani FDR/FWER.
- Porównanie wszystkich kandydatów używa optymalizacji z wielu startów, bez odrębnego dowodu globalności każdego takiego dopasowania. Dowód wykluczenia konkurentów tego nie wymaga.
- Ogólne twierdzenia dotyczą również bloków, ale kod selekcji par i wzajemnych testów warunkowych nie został jeszcze wdrożony.
- Nie przeprowadzono jeszcze pełnego audytu nowości. Wypukłość, nierówności macierzowe i branch-and-bound są znanymi składnikami; nowość należy ustalić dla dokładnej konstrukcji i jej użycia do selekcji modeli.
- Duże liczby grup wymagają rzadkiej algebry liniowej. Pilotaż używa gęstej macierzy o rozmiarze liczby kolumn efektów losowych.
- Nie zmieniano pierwotnego artykułu ani repozytorium ARSS-JPS.

## Najbliższy etap pracy do CSDA

1. Sprawdzić nowość dokładnego ograniczenia i sposobu współdzielenia obliczeń.
2. Ustalić rygor numeryczny gwarancji; dodać certyfikowane decyzje o zatrzymaniu, także gdy model bazowy jest dopasowany tylko lokalnie.
3. Zaimplementować całą ścieżkę selekcji i pary kandydatów.
4. Porównać z oryginalnym ARSS-JPS oraz współczesnymi metodami mieszanymi z karami i selekcją podzbiorów, na zgodnych klasach modeli.
5. Przeprowadzić większe eksperymenty i dwie aplikacje z ponowną selekcją w zbiorach treningowych.

Materiał stanowi sprawdzony pierwszy etap badawczy. Nie jest jeszcze gotowym
artykułem do wysłania do CSDA.

## Źródła do dalszego pozycjonowania

- [Thompson, Wand, Wang: Scalable Subset Selection in Linear Mixed Models](https://arxiv.org/html/2506.20425v3).
- [Ghosh, Thoresen: Robust and Scalable Sure Screening...](https://arxiv.org/abs/2606.27789).
- [Rügamer, Baumann, Greven: Selective inference for additive and linear mixed models, CSDA](https://doi.org/10.1016/j.csda.2021.107350).
- [Zhou i współautorzy: MM Algorithms for Variance Components Models](https://arxiv.org/abs/1509.07426).
- [Boyd, Vandenberghe: Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).

Wersje wykonawcze: Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}.
'''
    (root/'CSDA_stage1_summary_PL.md').write_text(md)
    (results/'summary.json').write_text(json.dumps({'main_cases':len(p),'main_resolved':int((p.status=='separated').sum()),
        'main_agreements':int(p.agreement.sum()),'candidate_fits':int(p.full_candidate_fits.sum()),
        'comparator_fits':int(p.p.sum()),'median_time_ratio':float(p.speedup.median()),
        'time_ratio_range':[float(p.speedup.min()),float(p.speedup.max())],
        'scale_cases':len(scale),'scale_agreements':int(scale.agreement.sum()),
        'heuristic_cases':len(top3),'heuristic_agreements':int(top3.agreement.sum()),
        'median_certified_to_top3_time':float(ratio)},indent=2))
    print((results/'summary.json').read_text())


if __name__=='__main__':main()
