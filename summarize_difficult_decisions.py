"""Summarize stage-eight frozen outputs; never runs an optimizer."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'results/difficult_decisions'
protocol = json.loads((DEST/'protocol.json').read_text())
rows = [json.loads((DEST/(c['id']+'.json')).read_text()) for c in protocol['cases']]
assert len(rows) == 55
assert not list(DEST.glob('*_failure.json'))
for r in rows:
    assert set(r['methods']) == {'verified','multistart','single_starts'}
    assert all('error' not in r['methods'][m] for m in ['verified','multistart'])
    assert isinstance(r['methods']['single_starts'], list)
    if r['methods']['verified']['status'] == 'verified':
        assert r['offline_replay']['valid']
    oracle = r.get('oracle_recheck')
    if oracle and not oracle['exact_tie'] and r['offline_replay']['valid']:
        assert r['methods']['verified']['winner_index'] == oracle['winner_index']

controlled = [r for r in rows if r['case']['kind'] == 'controlled']
crossed = [r for r in rows if r['case']['kind'] == 'crossed']
tie = next(r for r in rows if r['case']['kind'] == 'exact_tie')

def brief(group):
    ver = [r['methods']['verified'] for r in group]
    multi = [r['methods']['multistart'] for r in group]
    return {'n': len(group), 'online_status': dict(Counter(v['status'] for v in ver)),
            'online_verified': sum(v['status'] == 'verified' for v in ver),
            'offline_verified': sum(r['offline_replay']['valid'] for r in group),
            'median_splits': float(np.median([v['splits'] for v in ver])),
            'split_range': [min(v['splits'] for v in ver), max(v['splits'] for v in ver)],
            'median_online_seconds': float(np.median([v['total_seconds'] for v in ver])),
            'median_multistart_seconds': float(np.median([v['seconds'] for v in multi])),
            'median_single_start_seconds': float(np.median([s['seconds'] for r in group for s in r['methods']['single_starts']])),
            'median_offline_seconds': float(np.median([r['offline_replay']['seconds'] for r in group])),
            'single_start_winner_disagreements': sum(len(set(s['winner_index'] for s in r['methods']['single_starts'])) > 1 for r in group),
            'all_starts_match_multistart': sum(all(s['winner_index'] == r['methods']['multistart']['winner_index'] for s in r['methods']['single_starts']) for r in group),
            'online_agrees_multistart': sum(v['status'] == 'verified' and v['winner_index'] == m['winner_index'] for v, m in zip(ver, multi))}

gaps = []
for magnitude in [1., .01, 1e-4, 1e-6, 1e-8]:
    rr = [r for r in controlled if abs(r['case']['target_gap']) == magnitude]
    gaps.append({'target_absolute_gap': magnitude, **brief(rr)})

cross_summary = brief(crossed)
spreads = [max(s['fits'][j]['f'] for s in r['methods']['single_starts'])-
           min(s['fits'][j]['f'] for s in r['methods']['single_starts'])
           for r in crossed for j in range(4)]
bad = []
for r in crossed:
    for s in r['methods']['single_starts']:
        for j, fit in enumerate(s['fits']):
            if not fit['stationary'] or not fit['optimizer_success'] or fit['final_upper_active']:
                bad.append({'id': r['case']['id'], 'start': s['start'], 'support_index': j, **fit})
cross_summary.update(maximum_same_support_objective_spread=float(max(spreads)),
                     support_families_spread_above_1e_6=sum(v>1e-6 for v in spreads),
                     single_start_fits=24*9*4,
                     diagnostic_failures=bad)
dense_diagnostics = json.loads((DEST/'dense_local_diagnostics.json').read_text())
cross_summary['single_start_decisions_disagreeing_with_certificate'] = len(dense_diagnostics['decision_differences'])

oracle_errors = []
for r in controlled:
    win = r['oracle_recheck']['winner_index']
    for s in r['methods']['single_starts']:
        if s['winner_index'] != win:
            oracle_errors.append({'id': r['case']['id'], 'method': 'single_start', 'start': s['start']})
    if r['methods']['multistart']['winner_index'] != win:
        oracle_errors.append({'id': r['case']['id'], 'method': 'multistart'})
max_oracle_point_gap = max(c['absolute_difference_upper'] for r in [*controlled, tie]
                           for c in r['oracle_point_checks'])
summary = {'design': {'controlled_perturbations': 30, 'independent_controlled_draws': 3,
                       'crossed_independent_datasets': 24, 'exact_tie_controls': 1,
                       'excluded_pilot_cases': 8},
           'all': brief(rows), 'controlled': brief(controlled), 'gaps': gaps,
           'crossed': cross_summary, 'dense_diagnostics': dense_diagnostics,
           'controlled_local_disagreements_with_oracle': oracle_errors,
           'maximum_oracle_point_absolute_difference_upper': max_oracle_point_gap,
           'tie': {'online_status': tie['methods']['verified']['status'],
                   'offline_valid': tie['offline_replay']['valid'],
                   'single_start_winners': [s['winner_index'] for s in tie['methods']['single_starts']],
                   'multistart_winner': tie['methods']['multistart']['winner_index']},
           'limitations': ['Designed gap perturbations are dependent; no population failure rates or confidence intervals.',
                           'Crossed initializations are repeated analyses of each dataset; they are not independent replicates.',
                           'No claim that a positive-gap local decision was corrected unless observed.',
                           'Offline replay is additional work, not an online algorithm completion.',
                           '500-split, 8-second diagnostic budget differs from earlier full-path studies.']}
(DEST/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

tex = [r'\begin{table}[!ht]\centering\small',
       r'\begin{tabular}{rrrrrr}\toprule',
       r'$|\Delta|$ & Online & Replay & Splits & Online (s) & Local (s)\\\midrule']
for g in gaps:
    exp = int(round(np.log10(g['target_absolute_gap'])))
    label = '$1$' if exp == 0 else '$10^{'+str(exp)+'}$'
    tex.append(f"{label} & {g['online_verified']}/6 & {g['offline_verified']}/6 & {g['median_splits']:g} & {g['median_online_seconds']:.3f} & {g['median_multistart_seconds']:.3f}"+r'\\')
tex += [r'\bottomrule\end{tabular}',
        r'\caption{Controlled decision gaps: three residual draws and both signs per magnitude. Online and Replay count verified decisions in the original search and in separate subsequent checking of its saved witness. Splits and times are medians; Local denotes the unchanged multistart comparator. Replay time is additional and excluded from Online. Perturbations within a draw are dependent.}\label{tab:difficult}',
        r'\end{table}']
(DEST/'gap_table.tex').write_text('\n'.join(tex)+'\n')

findings = rf'''The original search verifies {summary['controlled']['online_verified']}/30 controlled decisions,
all with the analytic winner. All six cases at each of $|\Delta|=1$, $10^{{-2}}$
and $10^{{-4}}$ complete; none at $10^{{-6}}$ or $10^{{-8}}$ completes under
the diagnostic budget. Across the 30 nonzero-gap inputs, the seven single-start
decisions and the multistart decision all agree with the analytic winner.
The experiment therefore does not demonstrate an incorrect local selection
on these nonzero-gap inputs.

Separate replay verifies {summary['controlled']['offline_verified']}/30 saved witnesses, including nine of the
twelve unresolved nonzero-gap cases. The other three propose the wrong support
at $\Delta=-10^{{-8}}$; both the online procedure and subsequent checker decline
to certify them. These are unaccepted proposals, not erroneous certified outputs.
At the exact tie, single-start choices vary and the multistart fit proposes an
addition; online checking remains unresolved and replay rejects strict separation.
The stipulated reference policy retains the current model on a stopping tie.

On all 24 crossed datasets the online decision completes, passes replay and
agrees with multistart. Four of the 216 single-start family decisions disagree
with the certificate, on two ordinary-design datasets. Each affected family
contains a nonstationary fit despite the optimizer's success flag. Independent
dense covariance calculations reproduce the recorded objectives to
$1.43\times10^{{-13}}$ and the projected-gradient norms to $5.12\times10^{{-13}}$.
Across all 864 single-start model fits, 23 have a diagnostic warning; restarting
these fits with the dense objective lowers 21 objective values by more than
$10^{{-6}}$. Thus the changed choices demonstrate premature numerical stopping,
not distinct stationary local minima. A gradient diagnostic flags all four
affected decisions, and multistart resolves them without interval arithmetic.
Median family times are {cross_summary['median_online_seconds']:.3f} seconds for online certification,
{cross_summary['median_multistart_seconds']:.3f} for multistart, and
{cross_summary['median_single_start_seconds']:.3f} per single start.
The largest same-support objective spread across starts is
${cross_summary['maximum_same_support_objective_spread']:.2g}$.
These small diagnostic families establish observed agreement, not global
reliability of local optimization across other designs.
'''
assert not oracle_errors, 'Narrative requires revision: local disagreement was found'
assert cross_summary['all_starts_match_multistart'] == 22
assert cross_summary['online_agrees_multistart'] == 24
assert len(dense_diagnostics['decision_differences']) == 4
(DEST/'findings.tex').write_text(findings)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
x = np.array([g['target_absolute_gap'] for g in gaps])
axes[0].plot(x, [g['median_online_seconds'] for g in gaps], 'o-', label='Online search + checking', color='#1f5f99')
axes[0].plot(x, [g['median_multistart_seconds'] for g in gaps], 's--', label='Local multistart', color='#9a6617')
axes[0].plot(x, [g['median_offline_seconds'] for g in gaps], '^:', label='Additional replay', color='#488667')
axes[0].set(xscale='log', yscale='log', xlabel='Absolute penalized gap', ylabel='Median seconds', title='A  Cost of one family decision')
axes[0].invert_xaxis(); axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(x, [g['online_verified'] for g in gaps], 'o-', label='Online certified', color='#1f5f99')
axes[1].plot(x, [g['offline_verified'] for g in gaps], '^--', label='Witness passes later replay', color='#488667')
axes[1].set(xscale='log', xlabel='Absolute penalized gap', ylabel='Verified cases out of 6',
            title='B  Completion and independent replay', ylim=(-.25, 6.5), yticks=[0, 3, 6])
axes[1].invert_xaxis(); axes[1].legend(frameon=False, fontsize=8, loc='lower left')
for ax in axes:
    ax.grid(axis='y', alpha=.2)
fig.savefig(ROOT/'figures/difficult_decisions.png', dpi=220)
fig.savefig(ROOT/'figures/difficult_decisions.svg')
plt.close(fig)

table = '\n'.join(f"| {g['target_absolute_gap']:g} | {g['online_verified']}/6 | {g['offline_verified']}/6 | {g['median_splits']:g} | {g['median_online_seconds']:.3f} | {g['median_multistart_seconds']:.3f} |" for g in gaps)
report = f'''# CSDA: wartość certyfikacji przy trudnych decyzjach

Wykonano punkt 2: wyostrzono uzasadnienie wkładu i przeprowadzono celowane badanie. Stan: 14 września 2026.

## Co jest wkładem pracy

Wkładem jest konstrukcja sprawdzalnej decyzji dla całej rodziny rozszerzeń LMM po ponownej estymacji wariancji: dolne granice dla konkurentów, wykluczenie nieograniczonych ogonów i zapis umożliwiający ponowne sprawdzenie wyboru bez optymalizacji. Certyfikat dotyczy ustalonej reguły BIC z dodawaniem pojedynczych zmiennych i par. Nie gwarantuje najlepszego podzbioru ani poprawnej interpretacji statystycznej wybranych efektów.

Nie deklarujemy nowości samego profilowania ML, metody podziału dziedziny ani arytmetyki przedziałowej. Konstrukcja łączy je w sprawdzanie decyzji przy zmieniającej się kowariancji. Nowy eksperyment konkretyzuje koszt i ograniczenia tej gwarancji; nie jest dowodem pierwszeństwa w literaturze.

## Projekt eksperymentu

- 30 kontrolowanych wejść: trzy niezależne losowania reszt, po dziesięć zależnych perturbacji. Model ma 96 obserwacji, 12 grup po 8 i jeden predyktor o zerowej sumie wewnątrz każdej grupy. Dokładne minimum profilu jest dostępne analitycznie.
- Kontrolowano różnicę Δ = J(model z predyktorem) − J(model bazowy), dla obu znaków i modułów 1, 0.01, 0.0001, 0.000001, 0.00000001. Dodatnia Δ oznacza zatrzymanie, ujemna dodanie.
- Jeden osobny, symbolicznie dokładny remis: n=4, dwie grupy, x=(1,−1,1,−1), y=(4,0,−2,−2). Reguła referencyjna zachowuje model bazowy.
- 24 niezależne dane z dwoma komponentami losowymi: po sześć w warunkach zwykłych, z zerową drugą wariancją, prawie pokrywającymi się grupowaniami oraz dodatkowo predyktorami związanymi z grupami. n=120, p=3; badano rodzinę wszystkich pojedynczych dodatków do modelu bazowego.
- Siedem startów lokalnych w jednym wymiarze i dziewięć w dwóch, plus niezmieniony wcześniejszy multistart. Wykorzystano te same tolerancje L-BFGS-B co w silniku głównym; pojedyncze starty miały dziedzinę początkową [0,128]^q z rozszerzaniem aktywnej granicy.
- Niezmieniony certyfikator: 128 bitów, 500 podziałów, miękki limit 8 s na rodzinę. To osobny budżet diagnostyczny, krótszy od wcześniejszych badań ścieżek.
- Każdy zapis poddano osobnemu sprawdzeniu od oryginalnych danych, także gdy wyszukiwanie zgłosiło brak rozstrzygnięcia. Nie przepisano tych późniejszych sukcesów jako sukcesów algorytmu online.

Osiem wejść technicznego pilota zachowano osobno i wyłączono z powyższych liczb. Protokoły, źródła i wejścia zamrożono przed serią główną. Nie była to prerejestracja ani zewnętrzny eksperyment.

## Wyniki kontrolowanych luk

| Moduł Δ | Certyfikat online | Późniejsze sprawdzenie | Mediana podziałów | Online [s] | Multistart [s] |
|---:|---:|---:|---:|---:|---:|
{table}

Wszystkie 18 certyfikowanych decyzji są zgodne z analitycznym minimum. Wszystkie 210 decyzji pojedynczego startu i 30 decyzji multistartu na wejściach z niezerową luką również są zgodne z nim. Nie znaleźliśmy tu przykładu błędnej decyzji lokalnej przy niezerowej luce.

Przy bardzo małych lukach 12 wyszukiwań pozostało nierozstrzygniętych. Dziewięć zapisanych świadectw przechodzi jednak późniejsze, osobne sprawdzenie. W trzech przypadkach Δ≈−10⁻⁸ wyszukiwanie pozostawiło niewłaściwego kandydata; poprawnie odmówiło jego certyfikacji, a późniejsze sprawdzenie również go odrzuciło. Nie były to zaakceptowane błędne decyzje.

Przy dokładnym remisie wybory lokalne zależały od startu: {summary['tie']['single_start_winners']} (0=model bazowy, 1=dodanie). Multistart wskazał {summary['tie']['multistart_winner']}. Certyfikator nie ogłosił ścisłego zwycięzcy.

## Dlaczego najmniejsze luki są trudne

W obecnym kodzie weryfikacja uruchamia się po przekroczeniu progu 10⁻⁶(1+|U|), gdzie U jest bieżącą wartością kryterium. Dolne granice propozycji mają dodatkowy zapas 2·10⁻⁸ zależny od skali celu, a aktualizacja kandydata wymaga poprawy przekraczającej 10⁻⁷. Te progi sterują wyszukiwaniem i mogą zapobiec wywołaniu dokładnego sprawdzania albo zmianie kandydata przy bardzo małej luce.

Wynik odróżnia istnienie matematycznego certyfikatu od jego znalezienia przez program z ustalonymi progami i budżetem. Dla dziewięciu nierozstrzygniętych przypadków problemem nie był brak możliwości potwierdzenia nierówności w arytmetyce 128-bitowej: potwierdziło je późniejsze sprawdzenie. Nie dowodzi to skuteczności dowolnego zwiększania precyzji; wskazuje konkretny kierunek przyszłej zmiany wyszukiwania. W tym badaniu nie zmieniano progów ani silnika.

## Wrażliwość na inicjalizację przy dwóch komponentach

Wszystkie 24 decyzje zostały certyfikowane, ponownie sprawdzone i były zgodne z multistartem. Na dwóch zbiorach w zwykłym scenariuszu wybór pojedynczego startu zależał od inicjalizacji: 4 z 216 takich decyzji różniły się od certyfikowanej. Każda z tych czterech rodzin zawierała niestacjonarne dopasowanie, mimo zwróconego przez solver statusu sukcesu. Nie są to niezależne powtórzenia pozwalające oszacować ogólną częstość błędów.

Niezależne obliczenia z pełnymi macierzami kowariancji odtworzyły wartości celu do 1.43·10⁻¹³, a normy gradientu z warunkami brzegowymi do 5.12·10⁻¹³. Ponowne uruchomienie optymalizacji z tych punktów na pełnych macierzach poprawiło 21 z 23 oznaczonych dopasowań o ponad 10⁻⁶. Rozbieżności dowodzą przedwczesnego zatrzymania numerycznego; nie dowodzą istnienia odrębnych stacjonarnych minimów lokalnych. Zwykła kontrola gradientu ostrzegała we wszystkich czterech przypadkach, a multistart poprawnie je rozstrzygał bez arytmetyki przedziałowej. To wtórna diagnostyka, która nie zastępuje pierwotnych wyników.

Mediana czasu: certyfikacja {cross_summary['median_online_seconds']:.3f} s, multistart {cross_summary['median_multistart_seconds']:.3f} s, pojedynczy start {cross_summary['median_single_start_seconds']:.3f} s. Największy rozrzut wartości celu tego samego modelu między startami wyniósł {cross_summary['maximum_same_support_objective_spread']:.3g}.

Zachowano diagnostykę wszystkich 864 dopasowań pojedynczych startów; {len(bad)} miało co najmniej jeden sygnał ostrzegawczy: warunek stacjonarności, status optymalizatora lub aktywną końcową granicę. Taki sygnał nie jest automatycznie błędnym wyborem modelu. Uzgodnienie modeli nie daje tym lokalnym metodom ogólnej gwarancji globalnej optymalności.

Niezależne wzory analityczne na zapisanych danych sprawdzono arytmetyką 256-bitową; ocena profilu w punktach analitycznych zgadza się z ogólnym silnikiem przedziałowym z maksymalnym górnym ograniczeniem różnicy {max_oracle_point_gap:.3g}.

## Konsekwencja dla manuskryptu

Praca ma teraz bezpośrednie uzasadnienie gwarancji obliczeniowej: prawidłowo zakończona certyfikacja pozwala sprawdzić decyzję bez ponownego uruchamiania optymalizatora, a przy niewystarczającym rozdzieleniu program jawnie odmawia rozstrzygnięcia. Nowe dane nie uzasadniają tezy o częstych błędach dobrego multistartu ani powszechnej przewadze predykcyjnej certyfikacji.

Dodano sekcję o trudnych decyzjach, tabelę kosztu i ukończenia, wykres, analityczny punkt odniesienia w dodatku oraz precyzyjniejszy opis wkładu i ograniczeń. Wcześniejsze badania, porównanie glmmsel i ich wyniki pozostają oddzielnymi seriami. Przegląd dowodów przez autorów i końcowe sprawy redakcyjne nadal wymagają domknięcia.
'''
(ROOT/'CSDA_trudne_decyzje_PL.md').write_text(report)
old = (ROOT/'history/stage7_before_difficult_decisions/CSDA_wyniki_i_plan_PL.md').read_text()
(ROOT/'CSDA_wyniki_i_plan_PL.md').write_text(report+'\n\n---\n\n'+old)
print(json.dumps({'all': summary['all'], 'controlled_oracle_errors': oracle_errors,
                  'crossed_max_objective_spread': max(spreads), 'crossed_diagnostic_failures': len(bad),
                  'oracle_point_difference_upper': max_oracle_point_gap}, indent=2))
