"""Build stage-three tables, a scientific figure and its historical Polish report."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent;dest=root/'results/global_path'
    rows=json.loads((dest/'rows.json').read_text());summary=json.loads((dest/'summary.json').read_text())
    checks=json.loads((dest/'verification_checks.json').read_text())
    groups=['signals','suppressor','boundary','null','near_tie','high_variance']
    labels=['Signals','Suppressor','Boundary','Null','Near tie','Large variance']
    table=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Scenario & Verified paths & Decisions & Fits / comparator & Median seconds \\',r'\midrule']
    for scenario,label in zip(groups,labels):
        rr=[r for r in rows if r['scenario']==scenario]
        table.append(f"{label} & {sum(r['verified_replay'] for r in rr)} / {len(rr)} & {sum(r['stages'] for r in rr)} & {sum(r['model_fits'] for r in rr)} / {sum(r['reference_model_fits'] for r in rr)} & {statistics.median(r['seconds'] for r in rr):.3f}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\caption{Third-stage full-orthant BIC paths. All certificate construction is timed. The comparator fits all models on $[0,1024]^2$; its search domain and initialization differ. These are exploratory timings, not a matched-domain speedup study.}',r'\end{table}']
    (dest/'global_table.tex').write_text('\n'.join(table)+'\n')
    high=[r for r in rows if r['scenario']=='high_variance']
    path_string=lambda blocks:r'$'+r'\to'.join(r'\{'+','.join(str(j+1) for j in b)+r'\}' for b in blocks)+r'$'
    table=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{llll}',r'\toprule',
           r'Seed & Additions on $[0,4]^2$ & Additions on $\mathbb R_+^2$ & Final support changes \\',r'\midrule']
    mdrows=[]
    for r in high:
        raw=json.loads((dest/f"high_variance_{r['seed']}.json").read_text())
        old=raw['box4_verified_path'];new=raw['verified_path'];changed=old['selected']!=new['selected']
        table.append(f"{r['seed']} & {path_string(old['additions'])} & {path_string(new['additions'])} & {'Yes' if changed else 'No'}"+r' \\')
        text_path=lambda a:' → '.join('+'.join(f'x{j+1}' for j in b) for b in a)
        mdrows.append(f"| {r['seed']} | {text_path(old['additions'])} | {text_path(new['additions'])} | {'Tak' if changed else 'Nie; zmienia się kolejność'} |")
    table += [r'\bottomrule',r'\end{tabular}',r'\caption{The same BIC policy on two domains. Entries use one-based variable indices and show successive additions. Both columns are certified on their stated domains.}\label{tab:domainchange}',r'\end{table}']
    (dest/'domain_comparison_table.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.35))
    for i,(scenario,label) in enumerate(zip(groups,labels)):
        rr=[r for r in rows if r['scenario']==scenario]
        offsets=[-.13,0,.13]
        axes[0].scatter([i+o for o in offsets],[r['seconds'] for r in rr],s=29,color='#087f8c',label='Verified full orthant' if i==0 else None)
        axes[0].scatter([i+o for o in offsets],[r['reference_seconds'] for r in rr],s=29,marker='x',color='#a64730',label='Local fits, box 1024' if i==0 else None)
        axes[1].scatter([i+o for o in offsets],[r['largest_cutoff'] for r in rr],s=30,color='#19384c')
    short=['Signals','Suppr.','Bound.','Null','Tie','Large var.']
    for ax in axes:
        ax.set_xticks(range(6),short,rotation=30,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Seconds per complete path');axes[0].set_ylim(0,3.6)
    axes[0].set_title('A  Observed total runtime',loc='left',fontweight='bold',fontsize=11)
    axes[0].legend(fontsize=9,loc='upper right',frameon=False)
    axes[1].set_yscale('log');axes[1].set_ylabel('Largest certified cutoff across path')
    axes[1].axhline(4,color='#a64730',ls='--',lw=.8)
    axes[1].set_title('B  Data-dependent finite cover',loc='left',fontweight='bold',fontsize=11)
    fig.tight_layout();fig.savefig(root/'figures/global_path_pilot.png',dpi=200);plt.close(fig)
    md=f'''# Rozwinięcie artykułu Applied Sciences w kierunku CSDA — etap 3

Materiał roboczy dla Marka Gałązki • 14 września 2026

**Mamy teraz certyfikowaną pełną ścieżkę BIC dla wszystkich nieujemnych ilorazów wariancji, przy sprawdzonych dodatnich dolnych granicach reszty.** W nowym pilotażu zakończyło się i niezależnie odtworzyło {summary['complete']}/{summary['cases']} ścieżek, obejmujących {summary['total_verified_stages']} decyzji. Usunięcie arbitralnej granicy `[0,4]²` zmieniło przebieg trzech ścieżek, a w dwóch przypadkach również końcowy zestaw zmiennych.

Proponowany tytuł pozostaje: **Verified covariance-aware screening for Gaussian mixed-model selection**. Nowe wyniki wzmacniają argument za takim tematem pracy do CSDA: znaczenie ma poprawność porównania po ponownej estymacji kowariancji na całej zadanej dziedzinie.

## Co dodałem

1. **Kontrolę nieograniczonych ogonów dla każdej rodziny modeli.** Rodzina zawiera model bieżący i wszystkie rozpatrywane pojedyncze zmienne albo pary. Każdy konkurent zostaje wykluczony zarówno wewnątrz skończonego obszaru, jak i poza nim.
2. **Ścisłe ustalanie przestrzeni projekcji.** Zależne kolumny macierzy efektów losowych i wyrazu wolnego obsługuje dokładna arytmetyka wymierna. Następnie Arb wyznacza przedział dla kwadratu odległości odpowiedzi od przestrzeni `[X_M+C, Z]`. Weryfikacja nie opiera się na umownym progu SVD.
3. **Podział szerokich obszarów w skali `log(1+λ)`.** Ograniczenia funkcji nadal wykorzystują oryginalne współrzędne `λ`. Zmieniły się położenia punktów podziału i punktów styczności, co pomaga przy bardzo szerokich zakresach wariancji.
4. **Odtwarzanie pełnej gwarancji z zapisanych danych.** Weryfikator ponownie liczy przestrzeń projekcji, dolne granice reszty, ograniczenia ogonów i wszystkie decyzje. Nie ufa zapisanym oszacowaniom tych wielkości.

## Wyniki nowego pilotażu

| Miara | Wynik |
|---|---:|
| Pełne ścieżki sprawdzone na całej nieujemnej dziedzinie | 18/18 |
| Decyzje potwierdzone przedziałowo | 76 |
| Zgodność całej ścieżki z dopasowaniem wszystkich modeli w `[0,1024]²` | 18/18 |
| Dodania par po zatrzymaniu pojedynczych dodatków | 3 |
| Osobne dopasowania wielostartowe, z modelami bieżącymi | 137 wobec 1150 |
| Zmienione ścieżki względem `[0,4]²` | 3 |
| Zmienione końcowe zestawy zmiennych | 2 |

Wykorzystano te same 15 zbiorów z poprzedniego pilotażu i dodano trzy z góry ustalone ziarna dla większych wariancji. Nadal jest to `n=240`, `p=10`, dwa skrzyżowane komponenty i łącznie dziewięć kolumn efektów losowych. Każdy scenariusz ma trzy powtórzenia; precyzja Arb wynosi 128 bitów. Wyniki nie są dużym badaniem statystycznych własności selekcji.

Nowy scenariusz ma odchylenia standardowe efektów losowych 4 i 6 przy odchyleniu resztowym 1. Zmienne `x1` i `x2` są zgodne ze strukturą grupowania, a prawdziwe stałe współczynniki niezerowe dotyczą `x3` i `x5`. W poniższej tabeli pokazuję kolejność dodatków; numeracja zaczyna się od 1, podczas gdy pliki JSON używają indeksów od 0.

| Ziarno | Ścieżka w `[0,4]²` | Ścieżka na całej dziedzinie | Inny model końcowy? |
|---|---|---|---|
{chr(10).join(mdrows)}

**Obie kolumny są certyfikowane dla swoich dziedzin.** Różnicy nie wyjaśnia wyłącznie niedokładny lokalny optymalizator: ograniczenie dziedziny zmienia dokładnie określony problem selekcji. Dodatkowe porównanie w `[0,2048]²` dało te same pełne ścieżki co nowy algorytm we wszystkich trzech próbach. Porównania wielostartowe pozostają metodami lokalnymi; dowód decyzji pochodzi z certyfikatu.

W ziarnie 915261 pełna dziedzina również prowadzi do dodania `x2`, która nie ma prawdziwego stałego sygnału. Certyfikat potwierdza wykonanie kryterium, a nie brak fałszywych odkryć.

## Koszt i usprawnienie podziału

Zaobserwowana mediana czasu porównania / czasu nowej metody wyniosła **{summary['median_reference_over_verified_time']:.2f}**, a zakres **{summary['minimum_reference_over_verified_time']:.2f}–{summary['maximum_reference_over_verified_time']:.2f}**. Wszystkie koszty tworzenia certyfikatów są wliczone. Dodatkowe odtwarzanie świadków oraz zapis plików są poza czasem benchmarku. **Nie jest to porównanie szybkości przy identycznych dziedzinach i punktach startowych:** algorytm ma adaptacyjne obszary i dodatkowe dopasowania przy rozszerzaniu, a porównanie lokalne ma ustalone `[0,1024]²`. Dlatego głównym wynikiem tego etapu jest rozszerzenie gwarancji i wykazana zależność selekcji od dziedziny.

Automatyczne granice są zachowawcze: największa wyniosła **524 288**, mimo że największe ilorazy wariancji w zapisanych punktach zwycięzców w nowych próbach wynosiły około 35–48. Granica służy do pokrycia dziedziny w dowodzie; nie jest oszacowaniem parametru ani przedziałem ufności.

W jednym zapisanym porównaniu dla modelu bieżącego `x3,x5` i ziarna 915260 podział liniowy pozostał nierozstrzygnięty po **2500 podziałach**, natomiast wariant `log(1+λ)` potwierdził decyzję po **98**. Geometrię dopracowano przy użyciu tego ziarna. Są to przykłady rozwojowe; wynik nie dowodzi uniwersalnej przewagi tej geometrii.

## Co obejmuje dowód

Dla każdego modelu `C` obliczamy dodatnią dolną granicę `ℓ_C` na kwadrat odległości odpowiedzi od przestrzeni jego efektów stałych i wszystkich efektów losowych. Pozwala to ograniczyć profilowaną resztę dla dowolnych nieujemnych wariancji. Monotoniczność wyznacznika daje następnie dolną granicę kryterium poza prostokątem. Do tej granicy dodajemy właściwą karę BIC dla konkretnego modelu.

Jeżeli konkurent nie może pobić osiągalnej wartości zwycięzcy ani w prostokącie, ani poza nim, jest wykluczony na całej dziedzinie. Tak samo certyfikujemy zatrzymanie, ponieważ model bez dodatku uczestniczy w porównaniu. Prostokąty mogą być inne w kolejnych krokach: kryterium referencyjne zawsze dotyczy całej nieujemnej dziedziny.

Do noty dodano twierdzenie o takim łączeniu ograniczeń i jego dowód. Numeryczny kontrakt dotyczy dokładnych wartości binarnych danych, przy zaufaniu do zastosowanych operacji FLINT/Arb i implementacji wzorów. Nie jest to formalna weryfikacja całego programu ani model błędu pomiarowego.

## Kontrole poprawności i przypadki nierozstrzygnięte

- Sześć przedziałów dla dolnych granic reszty zawiera niezależnie obliczone dokładne wartości wymierne. Sprawdzono także zmienną należącą dokładnie do przestrzeni efektów losowych.
- Przykład z odpowiedzią należącą do przestrzeni `[X,Z]` poprawnie zwraca `unresolved_tail`. To brak tego wystarczającego certyfikatu; nie wniosek o nieistnieniu estymatora ML.
- Zerowy komponent kowariancji także daje jawny brak tego certyfikatu. Jego automatyczne usuwanie z modelu nie zostało wdrożone.
- Dokładny remis pozostał nierozstrzygnięty również w wersji obejmującej całą dziedzinę; żadna zmienna nie została dodana po wyczerpaniu budżetu 120 podziałów.
- Odrzucono pięć celowo uszkodzonych lub niewłaściwie opisanych certyfikatów, w tym prawidłowy certyfikat `[0,4]²` przedstawiony jako gwarancja na całej dziedzinie.
- Wszystkie 18 ścieżek zostało ponownie sprawdzonych z surowych danych; sprawdzono także odczyt zapisanych plików bez optymalizacji i zgodność obsługi starszego formatu.

## Znaczenie dla planowanej pracy CSDA

Potencjalny wkład można teraz opisać precyzyjniej: **wspólne, certyfikowane porównywanie rodzin rozszerzeń Gaussian LMM po ponownej optymalizacji kowariancji, z kontrolą zatrzymania, pełnymi ścieżkami dodawania pojedynczych zmiennych i par oraz jawną kontrolą całej nieograniczonej dziedziny przy sprawdzalnym warunku resztowym.**

Sam argument projekcyjny ma bliski precedens w [Zhou i in.](https://arxiv.org/pdf/1509.07426), a profilowanie i redukcja wymiaru są znane z [de Carvalho i in. (2012)](https://webhomes.maths.ed.ac.uk/~mdecarv/papers/decarvalho2012a.pdf). Dokumentacja [dokładnych macierzy wymiernych FLINT](https://python-flint.readthedocs.io/en/latest/fmpq_mat.html) i [arytmetyki Arb](https://python-flint.readthedocs.io/en/latest/arb.html) opisuje użyte operacje. Nowe wdrożenie nie uzasadnia przypisywania sobie pierwszeństwa dla tych składników. Zachowano szczegółowy `NOVELTY_AUDIT_PL.md`.

**Kolejnym priorytetem jest skalowanie liczby grup oraz porównanie metod na większym badaniu.** Obecny silnik tworzy pełną macierz Grama kandydatów, wylicza wszystkie pary i stosuje gęste operacje dla efektów losowych. Dziewięć kolumn efektów losowych nie wystarcza do wykazania skalowalności dla dużych danych skrzyżowanych. Dokładna eliminacja wymierna również może stać się kosztowna. Potrzebny jest rzadki silnik obliczeń z zachowaniem kontroli błędu oraz pomiar czasu i pamięci całej procedury.

Do zgłoszenia nadal potrzebne są: większe powtarzane symulacje, ocena fałszywych odkryć i mocy, porównania z ARSS-JPS oraz współczesnymi metodami na wspólnej klasie modeli, uzasadnienie kary statystycznej i zastosowania z selekcją wewnątrz podziałów treningowych. Procedura BIC nie odtwarza wszystkich reguł LRT z opublikowanego ARSS-JPS i nie gwarantuje globalnie najlepszego podzbioru ani wnioskowania po selekcji.

## Pliki i odtworzenie

- `certified_global_path.py`: nowy silnik kontroli ogonów i pełnej ścieżki.
- `run_global_experiments.py`: pilotaż 18 ścieżek i zapisane kontrole.
- `results/global_path/`: surowe dane, kompletne świadki, porównania dziedzin i wyniki diagnostyczne.
- `CSDA_covariance_screening_note.pdf` oraz `.tex`: zaktualizowana nota matematyczna po angielsku.
- `CSDA_stage2_summary_PL.md`: zachowane podsumowanie wcześniejszego etapu.

Po instalacji `requirements.txt`, w katalogu pakietu i przy jednym wątku BLAS:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 run_global_experiments.py
python3 replay_saved_path.py results/global_path/high_variance_915260.json
python3 summarize_global_results.py
```

Pierwszy skrypt ponawia i nadpisuje lokalne wyniki etapu 3, wykorzystując zachowane dane etapu 2. Drugi odtwarza certyfikat z zapisanych tablic bez dopasowywania modeli. Czasy zależą od środowiska; aktualizacja tabel po nowym uruchomieniu wymaga również przeglądu opisowych wartości w statycznej nocie.
'''
    (root/'CSDA_stage3_summary_PL.md').write_text(md)


if __name__=='__main__':main()
