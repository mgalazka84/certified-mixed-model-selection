"""Build stage-two tables, a scientific figure and its historical Polish summary."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent
    rows=json.loads((root/'results/path/rows.json').read_text())
    summary=json.loads((root/'results/path/summary.json').read_text())
    groups=['signals','suppressor','boundary','null','near_tie']
    labels=['Signals','Suppressor','Boundary','Null','Near tie']
    tex=[r'\begin{table}[H]\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule',
         r'Scenario & Paths & Stages & Pair rescues & Fits / reference & Median time ratio \\',r'\midrule']
    for s,label in zip(groups,labels):
        rr=[r for r in rows if r['scenario']==s]
        tex.append(f"{label} & {len(rr)} & {sum(r['stages'] for r in rr)} & {sum(r['pair_additions'] for r in rr)} & {sum(r['model_fits'] for r in rr)} / {sum(r['reference_model_fits'] for r in rr)} & {statistics.median(r['reference_over_verified_time'] for r in rr):.2f}"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',
            r'\caption{Second-stage BIC path pilot: $n=240$, $p=10$, two crossed components with 9 random-effect columns. Time ratio is all-model comparator / verified procedure, including certificate construction. Fits include current-model and candidate multi-start calls.}',r'\end{table}']
    (root/'results/path/path_table.tex').write_text('\n'.join(tex)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,3.1))
    for i,s in enumerate(groups):
        rr=[r for r in rows if r['scenario']==s]
        yy=[r['reference_over_verified_time'] for r in rr]
        ax.scatter([i-.11,i,i+.11],yy,color='#087f8c',s=42,zorder=3)
    ax.axhline(1,color='#a64730',ls='--',lw=1)
    ax.text(4.43,1.06,'Equal time',ha='right',color='#a64730',fontsize=9)
    ax.set_xticks(range(len(groups)),labels);ax.set_ylim(0,5);ax.set_xlim(-.5,4.5)
    ax.set_ylabel('Reference time / verified time')
    ax.set_title('Full single/pair BIC paths: all verification costs included',loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(root/'figures/path_pilot.png',dpi=200);plt.close(fig)
    speed=summary['median_reference_over_verified_time']
    md=f'''# Rozwinięcie artykułu Applied Sciences w kierunku CSDA — wyniki etapu 2

Materiał roboczy dla Marka Gałązki • 14 września 2026

**Rekomenduję artykuł o certyfikowanej selekcji przy ponownej estymacji kowariancji.** Mamy już działający prototyp pełnej ścieżki z pojedynczymi zmiennymi, ratowaniem przez pary i zatrzymaniem. Decyzje sprawdza arytmetyka przedziałowa Arb. Audyt literatury zawęża potencjalną nowość do połączenia tych elementów i wspólnych obliczeń dla rodzin modeli; nie uzasadnia deklaracji pierwszeństwa dla samych lemmatów.

Proponowany tytuł: **Verified covariance-aware screening for Gaussian mixed-model selection**.

## Co zostało wykonane

1. Rozszerzyłem silnik na wielokolumnowy model bieżący i bloki kandydatów, w tym pary.
2. Zaimplementowałem pełną, jednoznacznie określoną ścieżkę BIC: najpierw najlepsza poprawiająca zmienna; po zatrzymaniu pojedynczych dodatków sprawdzenie wszystkich pozostałych par; po dodaniu pary powrót do pojedynczych zmiennych.
3. Dodałem przedziałowe sprawdzanie ograniczeń, całego podziału dziedziny kowariancji oraz kompletności rodzin kandydatów. Świadek zapisany w JSON pozwala odtworzyć dowód decyzji bez ponownej optymalizacji.
4. Przeprowadziłem ukierunkowany audyt nowości i zachowałem szczegółowe porównanie źródeł w `NOVELTY_AUDIT_PL.md`.
5. Uruchomiłem pilotaż pełnych ścieżek, testy błędnych certyfikatów, dokładnego remisu, wyczerpania budżetu i problemu złego uwarunkowania.

## Nowe wyniki obliczeniowe

| Miara | Wynik |
|---|---:|
| Pełne ścieżki zakończone i odtworzone przez weryfikator | 15/15 |
| Zgodność wszystkich kolejnych decyzji z porównaniem obejmującym wszystkie modele | 15/15 |
| Decyzje potwierdzone przedziałowo | 63 |
| Dodania par po zatrzymaniu pojedynczych dodatków | 3 |
| Dopasowania wielostartowe, łącznie z modelami bieżącymi | 93 wobec 972 |
| Mediana czasu porównania / czasu metody z certyfikacją | {speed:.2f}× |
| Zakres zaobserwowanych ilorazów czasu | 0,79–4,44× |

Każdy zbiór ma `n=240`, `p=10`, dwa skrzyżowane komponenty efektów losowych i łącznie 9 kolumn ich macierzy projektu. Zastosowano 128 bitów precyzji w Arb i dziedzinę ilorazów wariancji `[0,4]²`. Jest po trzy powtórzenia w scenariuszach: kilka sygnałów, supresja, zerowa składowa wariancji, brak sygnału i prawie równe wyniki kandydatów. To pilotaż wykonalności, nie duże badanie skuteczności statystycznej.

**Dwa przypadki z niemal równymi kandydatami były wolniejsze od pełnego dopasowania wszystkich modeli.** Koszt weryfikacji jest uwzględniony w czasach; dodatkowe odtworzenie zapisanych świadków oraz zapis plików są poza benchmarkiem. Zmniejszenie liczby osobnych dopasowań nie oznacza proporcjonalnego skrócenia czasu.

W trzech przykładach supresji metoda najpierw dodała `x3`, potem stwierdziła brak poprawy przez pojedyncze zmienne i dodała wspólnie `x1, x2`. W jednym z tych przykładów później dodała również zmienną bez sygnału. To konkretnie pokazuje, że poprawność obliczeń nie gwarantuje identyfikacji wyłącznie prawdziwych efektów.

## Co dokładnie zostało udowodnione i sprawdzone

Dla każdego modelu funkcję profilowanego ML ograniczamy od dołu na całej zadanej dziedzinie. Dla bieżącego najlepszego rozwiązania mamy wartość osiągalną, więc ograniczenie od góry na minimum. Jeśli górna granica zwycięzcy jest mniejsza od dolnych granic wszystkich konkurentów, decyzja jest rozstrzygnięta. Model bez żadnego dodatku należy do porównywanej rodziny, dlatego ta sama reguła sprawdza również zatrzymanie.

Arb wylicza od nowa iloczyny danych, rozwiązania układów, wyznaczniki, logarytmy i gradienty. Nie ufa dolnym granicom ani statusom optymalizatora z float64. Dane są traktowane jako dokładne wartości binarne zapisane w plikach, a nie jako nieznane liczby rzeczywiste sprzed zaokrąglenia. Gwarancja nie dotyczy niepewności pomiarowej.

Przy braku rozdzielenia wynik brzmi `unresolved_budget` lub `unresolved_numeric`. Dokładny remis i zerowy budżet podziałów poprawnie pozostały nierozstrzygnięte. Weryfikator odrzucił pięć celowo błędnych przypadków: niewłaściwe dane, brak połowy dziedziny, niedopuszczalny punkt, błędnego zwycięzcę i niekompletną rodzinę kandydatów. Dla bardzo źle uwarunkowanego modelu obliczenie punktowe zawiodło jawnie przy 64 bitach i powiodło się przy 192; **nie jest to test całej ścieżki dla takiego modelu**.

## Związek z opublikowanym ARSS-JPS

Nowa wersja używa jawnego kryterium `F_M* + |M| log(n)`, z ustaloną strukturą kowariancji. Jest to przykład certyfikowanej ścieżki z karą typu BIC. **Nie jest wiernym wdrożeniem wszystkich reguł LRT z ARSS-JPS.** Zmieniono regułę akceptacji, a pary sprawdza się w pełnej rodzinie, bez bramki korelacyjnej i shortlisty top-L. W szczególności nie należy nazywać wyniku „15/15 identycznych ścieżek ARSS-JPS”.

Dla tej reguły BIC zaakceptowana para po zatrzymaniu pojedynczych dodatków automatycznie poprawia kryterium również względem każdego ze swoich modeli jednozmiennowych. Jest to warunek warunkowej poprawy BIC, nie test dwóch wartości p. Uzasadnienie statystyczne konkretnej kary w wybranym asymptotycznym modelu skrzyżowanym pozostaje odrębną sprawą.

W notatce pozostaje ogólna formuła ograniczenia LR: jeżeli minima dwóch modeli leżą w `[L_M,U_M]` i `[L_C,U_C]`, to statystyka leży w `[L_M-U_C, U_M-L_C]`. Umożliwia to późniejsze wierne odtworzenie ustalonych progów LRT, ale kalibracja po wielokrotnej selekcji wymaga osobnej teorii.

## Co wniósł audyt literatury

Najbardziej istotne doprecyzowanie: profilowanie i redukcja ML do dziedziny o małym wymiarze są znane już z pracy [de Carvalho i in. (2012)](https://webhomes.maths.ed.ac.uk/~mdecarv/papers/decarvalho2012a.pdf). Również argument z dolną granicą reszty ma bliski odpowiednik w Lemma 3 u [Zhou i in.](https://arxiv.org/pdf/1509.07426). Nie należy przedstawiać tych elementów jako samodzielnych odkryć.

Potencjalny wkład brzmi: **wspólne, certyfikowane porównywanie rodzin rozszerzeń LMM po ponownej optymalizacji kowariancji, z kontrolą decyzji o zatrzymaniu i możliwością odtworzenia certyfikatu.** „Nie znalazłem odpowiednika w sprawdzonych tekstach” nie znaczy „udowodniłem pierwszeństwo”. Szczegóły i ograniczenia przeglądu są w osobnym audycie.

## Zakres gwarancji i dalsza praca do CSDA

- Obecne certyfikaty ścieżek dotyczą `[0,4]²`. Poprzednie twierdzenie o wykluczeniu nieograniczonych ogonów pozostaje w notatce, ale nie zostało połączone z przedziałową weryfikacją pełnych ścieżek. Nie rozszerzamy automatycznie nowych wyników na wszystkie nieujemne wariancje.
- Wynik dotyczy dokładnej decyzji w kolejnej rodzinie dodatków, nie globalnie najlepszego podzbioru wszystkich zmiennych.
- Implementacja par tworzy macierz Grama kandydatów i wylicza wszystkie pary. Wersja produkcyjna dla dużego `p` i wielu grup wymaga dalszych zmian; obecne wyniki nie dowodzą takiego skalowania.
- Potrzebne są porównania z oryginalnym ARSS-JPS, dobrym algorytmem shortlistowym i metodami MSR3/ℓ₀ na wspólnych, rzeczywiście obsługiwanych modelach. Dzisiejszy benchmark porównuje dwa sposoby wykonania tej samej ścieżki BIC.
- Następny etap badania powinien obejmować wiele grup, powtarzane symulacje z oceną mocy i fałszywych odkryć, pamięć i czas całej procedury oraz co najmniej dwa zastosowania z selekcją wykonywaną wewnątrz podziałów treningowych.

Proponowany układ artykułu: problem zmiany rankingu po reficie → ograniczenia funkcji profilowanej → algorytm i świadek decyzji → twierdzenie o zgodności ścieżki → badanie obliczeniowe → zastosowania → ograniczenia. Gotowy jest rdzeń metodologiczny i prototyp; gotowość do zgłoszenia wymaga przede wszystkim mocniejszej walidacji empirycznej.

## Jak odtworzyć wynik

W katalogu pakietu, po zainstalowaniu `requirements.txt` i ustawieniu wątków BLAS na 1:

```bash
python3 run_path_experiments.py
python3 replay_saved_path.py results/path/suppressor_914360.json
python3 summarize_path_results.py
```

Pierwszy skrypt ponawia pilotaż; drugi sprawdza zapisany świadek bez dopasowywania modeli. Surowe dane symulowane, 15 kompletnych świadków i wyniki kontrolne są w `results/path/`. Zachowano także wyniki wcześniejszego, nieprzedziałowego pilotażu pojedynczego kroku (35 zbiorów); nie należy mieszać obu protokołów ani ich miar czasu.
'''
    (root/'CSDA_stage2_summary_PL.md').write_text(md)


if __name__=='__main__':main()
