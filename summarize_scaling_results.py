"""Rebuild the matched scaling tables, scientific figures and Polish report."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent;dest=root/'results/scaling_path'
    rows=json.loads((dest/'rows.json').read_text());s=json.loads((dest/'summary.json').read_text())
    checks=json.loads((dest/'verification_checks.json').read_text())
    good=[r for r in rows if not r['failures'] and r['generic_status']=='complete' and r['crossed_status']=='complete']
    order=['groups25','groups50','groups100','groups200','candidates40','candidates60']
    grouped={name:[r for r in good if r['configuration']==name] for name in order}
    median=lambda rr,key:statistics.median(r[key] for r in rr)
    tex=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{rrrrrr}',r'\toprule',
         r'$r_1+r_2$ & $p$ & Paths & \shortstack{Seconds\\generic / new} & \shortstack{Time\\ratio} & \shortstack{Peak MiB\\generic / new} \\',r'\midrule']
    mdrows=[]
    for name in order:
        rr=grouped[name]
        if not rr:continue
        r=rr[0]
        gt,nt=median(rr,'generic_seconds'),median(rr,'crossed_seconds')
        gm,nm=median(rr,'generic_peak_rss_mib'),median(rr,'crossed_peak_rss_mib')
        ratio=median(rr,'generic_over_crossed_time')
        tex.append(f"{r['random_columns']} & {r['p']} & {len(rr)} & {gt:.2f} / {nt:.2f} & {ratio:.2f} & {gm:.1f} / {nm:.1f}"+r' \\')
        mdrows.append(f"| {r['groups'][0]} + {r['groups'][1]} | {r['p']} | {gt:.2f} → {nt:.2f} s | {ratio:.2f}× | {gm:.1f} → {nm:.1f} MiB |")
    tex += [r'\bottomrule',r'\end{tabular}',r'\caption{Matched stage-four backend comparison, $n=800$. Each row has two signal datasets and one suppressor dataset. Entries are medians; the time ratio is the median of paired ratios. RSS is total Linux process high-water memory including imports and data.}\label{tab:scalingmatched}',r'\end{table}']
    (dest/'scaling_table.tex').write_text('\n'.join(tex)+'\n')
    macros={
        'ScalingCases':str(s['cases']),'ScalingComplete':str(s['both_complete']),
        'ScalingStages':str(s['verified_stages_per_backend']),'ScalingPairs':str(s['pair_additions_per_backend']),
        'ScalingRatio':f"{s['median_generic_over_crossed_time']:.2f}",
        'ScalingMinRatio':f"{s['minimum_generic_over_crossed_time']:.2f}",
        'ScalingMaxRatio':f"{s['maximum_generic_over_crossed_time']:.2f}",
        'ScalingMemoryReduction':f"{s['median_peak_rss_reduction_percent']:.1f}",
        'ScalingGenericPeak':f"{s['maximum_generic_peak_rss_mib']:.1f}",
        'ScalingNewPeak':f"{s['maximum_crossed_peak_rss_mib']:.1f}",
    }
    (dest/'scaling_macros.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(9.3,6.2))
    sweeps=[(['groups25','groups50','groups100','groups200'],'random_columns','Random-effect columns; p = 20'),
            (['groups50','candidates40','candidates60'],'p','Candidate variables; random-effect columns = 50')]
    for row,(names,xkey,xlabel) in enumerate(sweeps):
        for col,(key,ylabel) in enumerate([('seconds','Seconds per complete path'),('peak_rss_mib','Peak process RSS (MiB)')]):
            ax=axes[row,col]
            for backend,label,color,marker in [('generic','Generic Arb','#a64730','s'),('crossed','Structured Arb','#087f8c','o')]:
                xx=[grouped[name][0][xkey] for name in names]
                yy=[median(grouped[name],backend+'_'+key) for name in names]
                lows=[min(r[backend+'_'+key] for r in grouped[name]) for name in names]
                highs=[max(r[backend+'_'+key] for r in grouped[name]) for name in names]
                ax.errorbar(xx,yy,yerr=[[v-l for v,l in zip(yy,lows)],[h-v for h,v in zip(highs,yy)]],
                            color=color,marker=marker,ms=4,lw=1.3,capsize=3,label=label)
            ax.set_xlabel(xlabel,fontsize=9);ax.set_ylabel(ylabel)
            ax.set_xticks(xx);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
            ax.set_title(('A','B','C','D')[row*2+col]+'  '+('Time' if col==0 else 'Memory'),loc='left',fontweight='bold')
            if row==0 and col==0:ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.tight_layout(h_pad=2)
    fig.savefig(root/'figures/scaling_path_pilot.png',dpi=200)
    fig.savefig(root/'figures/scaling_path_pilot.svg')
    plt.close(fig)
    r200=grouped['groups200'];p60=grouped['candidates60']
    md=f'''# Rozwinięcie artykułu Applied Sciences w kierunku CSDA — etap 4

Materiał roboczy dla Marka Gałązki • 14 września 2026

**Rozszerzyłem badanie do 200 kolumn efektów losowych oraz 60 kandydatów i dodałem sprawdzany silnik obliczeń dla dwóch skrzyżowanych efektów losowych.** Nowa wersja zachowuje kryterium BIC i certyfikację całej nieujemnej dziedziny wariancji. Porównanie obejmuje czas i pamięć pełnych ścieżek selekcji.

Na {s['cases']} nowych zbiorach obie wersje zakończyły {s['both_complete']} par ścieżek. Uzyskano {s['same_full_paths']} zgodnych przebiegów i {s['identical_witnesses']} par **identycznych świadków decyzji**, obejmujących po {s['verified_stages_per_backend']} sprawdzonych kroków. Mediana przyspieszenia wyniosła **{s['median_generic_over_crossed_time']:.2f}×**, a mediana spadku szczytowego zużycia pamięci całego procesu **{s['median_peak_rss_reduction_percent']:.1f}%**.

Proponowany tytuł pozostaje: **Verified covariance-aware screening for Gaussian mixed-model selection**. Mamy teraz wynik metodologiczny z kontrolą całej dziedziny i lepiej udokumentowane koszty jego wykonania.

## Co zmieniło się w algorytmie

Profilowanie wskazało, że przy większej liczbie grup zasadniczy koszt stanowią obliczenia macierzy w weryfikatorze Arb. Wprowadziłem cztery powiązane zmiany:

1. **Mniejszy układ równań.** Dla dwóch skrzyżowanych efektów losowych z jedną przynależnością do grupy w każdym wierszu wykorzystujemy diagonalną strukturę bloków. Po eliminacji rozwiązujemy układ o wymiarze `min(r1,r2)`; przy 80 i 120 grupach jest to 80 zamiast 200.
2. **Wspólne iloczyny dla gradientów.** W danym punkcie wariancji przygotowujemy macierze Grama raz dla całej rodziny. Kolejny kandydat wymaga małego podukładu, co ogranicza powtarzanie operacji zależnych od liczby grup.
3. **Mniej pracy w wierzchołkach obszaru.** Gdy ograniczenie wymaga jedynie logarytmu wyznacznika, pomijamy rozwiązywanie układów dla wszystkich kolumn kandydatów.
4. **Ograniczone duże pamięci podręczne.** Zachowujemy najwyżej po 32 duże stany kowariancji i iloczynów gradientów oraz 256 wyników profilowania. Usunięte wyniki są w razie potrzeby ponownie obliczane z kontrolą błędu.

Te zmiany dotyczą weryfikowanych obliczeń. Silnik propozycji float64, reguła dodawania pojedynczych zmiennych i par, punkty startowe oraz budżety pozostały takie same w porównywanych wariantach.

## Porównanie na tych samych zadaniach

Każdy zbiór ma `n=800`. Zwiększaliśmy osobno liczbę grup przy `p=20` oraz liczbę kandydatów przy 20 i 30 grupach. **Nie badano jednocześnie 200 kolumn efektów losowych i 60 kandydatów.** Na konfigurację przypadają dwa zbiory z pojedynczymi sygnałami i jeden z parą supresyjną. To trzy przykłady rozwojowe, zbyt mało do precyzyjnej oceny rozkładów czasu.

| Grupy w dwóch komponentach | Kandydaci | Mediana czasu: stary → nowy | Mediana ilorazów czasu | Mediana szczytowej pamięci: stary → nowy |
|---|---:|---:|---:|---:|
{chr(10).join(mdrows)}

Dla 200 kolumn efektów losowych przyspieszenie w trzech próbach wyniosło **{min(r['generic_over_crossed_time'] for r in r200):.2f}–{max(r['generic_over_crossed_time'] for r in r200):.2f}×**. Przy 60 kandydatach największe znaczenie ma pamięć: mediany wyniosły **{median(p60,'generic_peak_rss_mib'):.1f} MiB** i **{median(p60,'crossed_peak_rss_mib'):.1f} MiB**.

Pełny zakres ilorazów czasu wynosi **{s['minimum_generic_over_crossed_time']:.2f}–{s['maximum_generic_over_crossed_time']:.2f}×**. Mała liczba grup daje mniejsze oszczędności. W tym badaniu obie wersje wykonały te same dopasowania i kroki podziału; poprawa wynika z ich realizacji numerycznej.

## Dlaczego to porównanie jest mocniejsze od wcześniejszego

W etapie 3 porównanie lokalne używało stałego prostokąta `[0,1024]²`, a certyfikowany algorytm innych, zależnych od danych obszarów. Dzisiejsze porównanie używa **tej samej procedury i tego samego problemu optymalizacji**. Dodatkowo porównano skróty dokładnie zapisanych świadków: rodzin modeli, skończonych obszarów, całych podziałów, punktów styczności i osiągalnych punktów zwycięzców.

Identyczny świadek został sprawdzony przez dwa silniki stosujące różne wzory obliczeń kowariancji. To silniejszy sprawdzian niż sama zgodność końcowego zestawu zmiennych. Jest to obserwacja dla zapisanych prób; różne szerokości przedziałów mogą prowadzić do różnej rozstrzygalności w innych przykładach i przy innych budżetach.

Każdy pomiar odbywał się w nowym procesie, z jednym wątkiem BLAS. Czas obejmuje przygotowanie modelu, poszukiwanie, dokładne wyliczanie przestrzeni projekcji, kontrolę ogonów oraz weryfikację wszystkich kroków. Pamięć to szczytowe RSS całego procesu Linux do zakończenia ścieżki, łącznie z interpreterem, bibliotekami i danymi. Dodatkowe odtwarzanie oraz zapis wyników są poza pomiarem. Nie jest to sam rozmiar tablic ani pomiar pamięci wszystkich procesów komputera.

## Sprawdzenie poprawności

- **{sum(r['support_point_checks'] for r in checks['exact_checks'])} kombinacji modelu i punktu wariancji:** dokładna arytmetyka wymierna sprawdza macierze kowariancji po kompresji i gradienty; logarytmy porównano z niezależnym obliczeniem przy 256 bitach.
- Uwzględniono oba porządki wielkości komponentów, zerowe ilorazy wariancji, niewykorzystane kolumny grup i rozłączną strukturę powiązań między grupami.
- Wymuszono usuwanie stanów z pamięci podręcznej i ponowne obliczenie wcześniejszego punktu.
- Nowy silnik odtworzył trzy wcześniejsze certyfikaty z dużą wariancją. Nowy świadek z większego modelu został odczytany z plików i ponownie sprawdzony również ogólnym silnikiem.
- Odrzucono niewłaściwe dane, niepełne pokrycie obszaru, błędnego zwycięzcę oraz nieobsługiwane struktury przynależności do grup.
- Dokładny remis i zerowa dolna granica reszty nadal dają jawny wynik nierozstrzygnięty.

Do noty dodano wzory eliminacji blokowej, ich dowód oraz wniosek o zachowaniu poprawności certyfikatów przez nowy silnik. Warunek struktury dwóch efektów jest sprawdzany na wejściu. Weryfikacja nadal dotyczy dokładnych wartości binarnych danych, z zaufaniem do operacji Arb i implementacji wzorów.

## Co pozostaje ograniczeniem

To specjalizacja dla **dwóch skrzyżowanych losowych wyrazów wolnych**, z dokładnie jedną binarną przynależnością do grupy w każdym komponencie. Model z wagami, wieloma przynależnościami, losowymi nachyleniami lub inną liczbą komponentów wymaga ogólnego silnika albo kolejnej specjalizacji.

Nadal tworzymy gęste macierze i sprawdzamy wszystkie pary. Przy zrównoważonej liczbie grup zasadniczy koszt gęstej algebry pozostaje sześcienny. Ograniczenie pamięci podręcznej nie ogranicza całkowitej pamięci niezależnie od liczby kandydatów i rozmiaru świadków. W obecnym porównaniu oceniono cztery zmiany razem; nie przypisujemy całego przyspieszenia jednemu składnikowi.

Wynik obliczeniowy nie gwarantuje odzyskania wyłącznie prawdziwych efektów. Niektóre poprawnie wykonane ścieżki dodają zmienne bez sygnału. BIC pozostaje osobno określoną regułą i nie odtwarza wszystkich testów LRT z opublikowanego ARSS-JPS.

## Znaczenie dla zgłoszenia do CSDA

Potencjalny wkład obejmuje obecnie: ograniczenia dla rodzin modeli po ponownej estymacji kowariancji, sprawdzane zatrzymanie, pełne ścieżki z parami, kontrolę całej dziedziny przy dodatniej granicy reszty oraz sprawdzaną specjalizację obliczeń dla dwóch skrzyżowanych efektów.

Eliminacja Schura, iloczyny Grama i ograniczanie pamięci podręcznej są znanymi technikami. Obliczenia z macierzami rzadkimi w LMM mają rozbudowaną literaturę, np. [Bates i in. (2015)](https://www.jstatsoft.org/article/view/v067i01). W tym etapie sprawdzono stronę publikacji i abstrakt; odsyłacz do pełnego PDF nie otworzył się poprawnie. Użyte operacje z kontrolą błędu opisuje [dokumentacja macierzy Arb](https://python-flint.readthedocs.io/en/latest/arb_mat.html). Uzupełniono `NOVELTY_AUDIT_PL.md` bez deklarowania pierwszeństwa dla tych składników.

**Kolejnym priorytetem powinno być badanie porównujące metody, a następnie zastosowanie na danych rzeczywistych.** Dzisiejszy eksperyment porównuje dwa sposoby obliczenia tej samej certyfikowanej ścieżki. Przed zgłoszeniem potrzebne są powtarzane symulacje z mocą i fałszywymi odkryciami, porównanie z ARSS-JPS i współczesnymi metodami na zgodnej klasie modeli oraz selekcja wewnątrz podziałów treningowych w zastosowaniach. Nie zastępuje tego kolejna niewielka poprawa czasu we własnym silniku.

## Odtworzenie

W środowisku Linux z pakietami z `requirements.txt`, w katalogu archiwum:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 run_scaling_experiments.py
python3 verify_crossed_backend.py
python3 summarize_scaling_results.py
python3 replay_saved_path.py results/scaling_path/groups100_917200_crossed.json --data results/scaling_path/groups100_917200_data.npz
```

Testy dodatkowe uruchamiamy po pomiarach, aby nie konkurowały o procesor. Ostatnie polecenie sprawdza nowy świadek wzorami ogólnego silnika, bez optymalizacji. Opcja `--backend crossed` wybiera nowy weryfikator. Surowe dane i wyniki obu procesów są w `results/scaling_path/`. Zachowano wcześniejsze rezultaty i podsumowania etapów 2 oraz 3.
'''
    (root/'CSDA_stage4_summary_PL.md').write_text(md)


if __name__=='__main__':main()
