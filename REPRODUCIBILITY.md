# Reproducibility protocols

## Mathematical scope

- Gaussian ML with `R(lambda) = I + sum(lambda[k] Z[k] Z[k].T)`.
- A fixed covariance-component structure. The stage-two engine supports arbitrary
  fixed-effect supports; the completed path uses single additions and pair rescue.
- Box bounds hold on the supplied compact ratio domain. The global extension
  verifies candidate-specific projection floors and determinant tail bounds to
  cover all nonnegative ratios, with a different finite search box at each stage.
- Stage one uses float64 cushions and has not been retroactively interval-verified.
  Stage two recomputes every certificate ingredient from binary input data in Arb.
- `certified_path.py` retains its compact-box contract. The global guarantee
  requires `certified_global_path.py` and its explicit checked tail witnesses.
- New decision status is `verified`; complete path status is `complete`. Budget
  exhaustion (`unresolved_budget`) and insufficient numerical separation
  (`unresolved_numeric`) do not authorize a new selection. Invalid/rank-deficient
  input models can raise an explicit exception.
- Missing positive floors, structurally zero covariance components and failed
  tail construction give `unresolved_tail` in the global implementation. This
  does not prove nonexistence of ML; the implemented condition is sufficient.
- The exact reference is observation-count BIC, `F_M* + |M| log(n)`: prefer a best
  improving single addition; otherwise try all remaining pairs; return to singles
  after accepting a pair. No correlation gate or top-L pair shortlist is used.
- The computational guarantee is not best-subset optimality, FWER control or
  selection-adjusted inference. The penalty's statistical justification is separate.

## Files

| File | Purpose |
|---|---|
| `covariance_screening.py` | Compressed Gaussian engine, box bounds, candidate exclusion and tail bounds |
| `certified_path.py` | Block engine, complete BIC path, Arb decision verifier and path replay |
| `certified_global_path.py` | Exact spanning bases, Arb projection floors, global path and replay |
| `certified_crossed_backend.py` | Smaller Schur solves, shared gradient products and bounded large-object caches |
| `run_scaling_experiments.py` | Matched fresh-process time and RSS study, 18 input datasets |
| `verify_crossed_backend.py` | Exact rational covariance/gradient checks and independent cross-replay |
| `summarize_scaling_results.py` | Stage-four results, scientific figures and historical Polish report |
| `comparison_methods.py` | Local BIC comparators, source-checked ARSS adapter and EN-CV + ML |
| `reference_arss_policy.py` | Unchanged archived ARSS search function; see `ARSS_POLICY_LICENSE` |
| `run_comparison_experiments.py` | Frozen repeated comparison protocol, training/test data and workers |
| `verify_comparison_methods.py` | Independent score/coefficient checks, ARSS policy and split isolation |
| `summarize_comparison_results.py` | Stage-five report, Monte Carlo uncertainty, paired differences and figure |
| `run_enet_sensitivity.py` | Wider penalty grid on the same inputs, including an intercept-only candidate |
| `summarize_enet_sensitivity.py` | Secondary grid effects on selected supports and prediction |
| `run_csda_validation.py` | Frozen stage-six real-data and simultaneous-dimension validation |
| `verify_csda_native.py` | Independent native statsmodels likelihood and local-refit checks |
| `summarize_csda_validation.py` | Stage-six tables, figure, uncertainty and current Polish report |
| `build_csda_manuscript.py` | Assemble the separate journal manuscript from the maintained theory and results |
| `run_penalized_lmm_comparison.py` | Frozen native glmmsel comparison and 40 new one-group datasets |
| `run_glmmsel_native.R` | Native R fitting and tuning on explicit whole-group folds |
| `summarize_penalized_lmm.py` | Main native comparison tables and Polish report |
| `run_glmmsel_grid_sensitivity.py` | Secondary low-penalty extension on all 46 original inputs |
| `run_glmmsel_extended.R` | Native extended-grid wrapper, with the package unchanged |
| `summarize_glmmsel_sensitivity.py` | Secondary narrative from saved outputs |
| `results/penalized_lmm/` | All 46 analyses, exact data, R inputs/objects, native diagnostics and certificates |
| `results/penalized_lmm_grid_sensitivity/` | All secondary native fits on the same inputs |
| `vendor/glmmsel/` | Original CRAN source, GPL-3 license, provenance and installation log |
| `CSDA_manuscript.tex` | Editable main manuscript, original Elsevier article class |
| `data/real/` | Unchanged public R datasets, CSV exports, original documentation and provenance |
| `results/csda_validation/` | Exact folds/data, all new method outputs, predictions and native checks |
| `run_global_experiments.py` | 18 full-orthant paths, domain comparison and tail-specific checks |
| `summarize_global_results.py` | Rebuild stage-three tables, figure and historical Polish report |
| `run_path_experiments.py` | 15 full paths, numerical checks and deliberately invalid certificates |
| `replay_saved_path.py` | Check a saved JSON witness from its original NPZ arrays without optimization |
| `summarize_path_results.py` | Rebuild stage-two table/figure and its historical Polish summary |
| `run_experiments.py` | The 35-dataset main pilot |
| `run_diagnostics.py` | Scaling, domain sensitivity, exact tie, rank reversal and tail examples |
| `verify_math.py` | Dense objective, derivative, convexity and bound diagnostics |
| `verify_extensions.py` | Multiple current fixed effects and tail-region checks |
| `run_shortlist_diagnostic.py` | Separate top-three heuristic comparison |
| `summarize_results.py` | Rebuild stage-one tables, plots and a separate historical summary |
| `results/` | Raw recorded outputs, protocol environment and generated tables |
| `results/path/` | 15 raw simulation datasets, complete witnesses, summaries and failure checks |
| `results/global_path/` | 18 full-orthant datasets/witnesses, larger-variance examples and checks |
| `results/scaling_path/` | Paired backend runs with exact witness hashes, measured RSS and diagnostics |
| `results/comparison/` | 300 full train/test datasets, five-method results, certificates and uncertainty |
| `figures/` | Exportable scientific figures |
| `CSDA_covariance_screening_note.tex` | Editable mathematical note |

The original uploaded Applied Sciences manuscript and the existing ARSS-JPS
repository were not modified. No original empirical commuting data are included.

## Reproduce

Run from this directory in a Python environment with the packages in `requirements.txt`.
The exact versions used for the recorded pilot are in `results/environment.json`.
Fix BLAS threads **before** Python starts:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 verify_math.py
python3 run_experiments.py --replicates 5 --max-splits 1200 --time-budget 20 --out results
python3 run_diagnostics.py
python3 verify_extensions.py
python3 run_shortlist_diagnostic.py
python3 summarize_results.py
python3 run_path_experiments.py
python3 replay_saved_path.py results/path/suppressor_914360.json
python3 summarize_path_results.py
python3 run_global_experiments.py
python3 replay_saved_path.py results/global_path/high_variance_915260.json
python3 summarize_global_results.py
python3 run_scaling_experiments.py
python3 verify_crossed_backend.py
python3 summarize_scaling_results.py
python3 verify_comparison_methods.py
python3 run_comparison_experiments.py
python3 run_enet_sensitivity.py
python3 summarize_enet_sensitivity.py
python3 summarize_comparison_results.py
pdflatex -interaction=nonstopmode -halt-on-error CSDA_covariance_screening_note.tex
pdflatex -interaction=nonstopmode -halt-on-error CSDA_covariance_screening_note.tex
python3 run_csda_validation.py prepare
python3 run_csda_validation.py run
python3 verify_csda_native.py
python3 summarize_csda_validation.py
```

Stage-six runs resume completed records and check the frozen source hashes.
Use a separate copy/output directory for a new protocol; do not mix partial
results from different code revisions. Native validation is performed after
timed selection experiments. To rebuild the journal artifacts from the saved
results without rerunning model selection, use the stage-seven summary and
manuscript-builder commands below, then compile the manuscript twice.

The real-data outer test partitions cover every retained observation once.
Carrots holds out both consumers and products (nine cells); cake holds out
experimental batches (five folds) and uses one variance component. Raw outcomes
are never used for the selected support outside their training fold. The given
carrot sensory PCs are fixed external descriptors, not a PCA refitted here.
The ordinal preference outcome is treated with a Gaussian working model.
Real-data folds overlap in training and do not support naive iid fold-level
standard errors. Dictionaries are column based and do not enforce hierarchy.

Rerunning overwrites the local result files. The supplied raw outputs preserve the
recorded run. Runtime depends on the machine, BLAS implementation and load; it is
not expected to reproduce exactly. The static note describes the recorded run;
review numeric narrative values when replacing its tables with a fresh run.

For a short exploratory run, use `--quick --replicates 1 --out exploratory`.
`run_diagnostics.py` expects the main result files to exist.

## Interpretation of counts and comparisons

A separate candidate fit is a multi-start optimization call for one candidate,
not a single likelihood evaluation. The method evaluates many candidate profiles
at shared covariance points while creating bounds. Timings include that work,
the base fit and construction of compressed products. The all-candidate comparator
uses the same numerical engine and initial starts. It is a multi-start local-ML
comparator, not an independently certified global optimizer.

The top-three diagnostic is explicitly not the complete original ARSS algorithm:
it does not expand after failed significance tests or perform pair rescue. Neither
it nor the current one-step prototype has been assessed here for FDR or FWER.

## Stage-two evidence and certificate contract

The recorded pilot completed 15/15 paths with 63 verified decisions and three pair
rescues. Every saved path replayed successfully and agreed with the all-model
multi-start comparator. There were 93 model fits including current-model fits,
versus 972; median reference/verified time was 2.88, range 0.79–4.44. Two near-tie
cases were slower with verification. Timing includes all Arb construction and
decision checks, but not the additional replay or disk writing.

Arb imports original binary data, recomputes cross products and encloses point
objectives and gradients. The verifier checks the complete partition and candidate
families before declaring a strict separation. Stored float lower bounds and
optimizer status are not trusted. The trusted components are the proved analytic
formulas, Python implementation and Arb enclosing operations. This is not a formal
verification of all software and is not a measurement-error model.

The new engine forms a full predictor Gram matrix, enumerates all pairs and uses
dense random-effect algebra. The old scalar engine has different storage behavior.
Neither the new pair implementation nor the 9-column random-effects pilot proves
scalability to large crossed grouping systems.

## Open research tasks

Tighten the conservative tail cover, handle more grouping levels with sparse
linear algebra and a scalable pair strategy, and compare additional native
penalized selectors that retain a fixed covariance structure. The stage-five adapter preserves the
ARSS-JPS search rule, while our certified BIC procedure still has a different
criterion. The focused audit found substantial prior foundations and does not
prove historical priority. Stage six adds real-data applications and simultaneous
growth of predictors and groups. Larger repeated studies are still needed for
precise rare-error estimates and broader claims about scaling.

## Stage-three evidence and additional contract

All 18 full-orthant paths completed and replayed, with 76 verified decisions and
three pair additions. All agreed with the all-model multistart comparator on
`[0,1024]^2`. The three new large-variance examples also agreed with its
`[0,2048]^2` runs. Compared with independently verified `[0,4]^2` paths, all three
changed addition sequences and two changed terminal supports. The previous 15
datasets retained their decisions. This is still an exploratory study with
`n=240`, `p=10` and nine random-effect columns.

Recorded fits were 137 versus 1150; median comparator/verified time was 5.52
(range 1.01–9.81). All exact-rank and Arb work is timed, with separate replay and
disk writes excluded. The global procedure uses adaptive pilot expansion and
data-dependent bounds; the comparator uses a fixed finite domain and different
starts. These ratios are descriptive, not a matched-domain speedup estimate.

Every tail floor is recalculated from raw input. Rational row reduction proves
that a selected basis spans the complete base/random design, including its
structural dependencies. Arb projection calculations are cached across supports.
If additional fixed columns lie in the projected space, an exact augmented-basis
fallback is used. Saved rank/floor claims are never trusted during replay.

The maximum automatic cutoff was 524288. These conservative cutoffs are proof
regions, not estimates or confidence limits. The logarithmic subdivision was
developed using large-variance seed 915260: one single-stopping family verified
in 98 splits while linear subdivision remained unresolved after 2500. This is
a development diagnostic, not an independent validation of a tuned algorithm.
Exact rational elimination, dense Gram products and all-pair enumeration may
become expensive as the problem grows.

## Using the structured backend

The default global API still uses the generic verifier. To opt into the
specialized two-component random-intercept backend:

```python
from certified_global_path import global_selection_path
from certified_crossed_backend import CrossedArbGaussian

result = global_selection_path(y, base, x, zs,
                               verifier_factory=CrossedArbGaussian)
```

Each row of each of the two random designs must have exactly one 1 and zeros
elsewhere. This condition is checked on raw data. Unsupported weighted or
multiple memberships and other component counts raise an explicit error;
use the generic backend for those model structures.

The new backend leaves the float search, criterion, full-orthant domain and
witness format unchanged. It uses a dense Schur complement with dimension
`min(r1,r2)`, avoids predictor solves at determinant-only vertices, computes
gradient Gram products once per anchor, and caps large covariance/gradient
caches at 32 entries and the profile cache at 256. It still builds dense input
products, uses exact rational tail-floor calculations, and enumerates all pairs.
For balanced group counts its principal dense arithmetic remains cubic in the
total number of group columns. This is not a general sparse implementation.

The scaling benchmark requires Linux for the documented process high-water RSS
measurement. Each backend runs in a fresh subprocess with BLAS set to one thread.
Time includes the complete path and verification; RSS includes the interpreter,
imports, loaded data and all path work. Extra replay and serialization are not
included in the measurements. The benchmark pairs the same input and verifies
equality of serialized decision witnesses as well as selected paths.

Both backends completed all 18 recorded inputs, independently verifying the same
85 decisions, including six pair additions. Both made 128 model-fit calls and
produced identical serialized witnesses. The median paired speedup was 1.77
(range 1.08–4.97), and median paired process peak-RSS reduction was 37.4%.
These are development results from three datasets per configuration at `n=800`.
Group-count and candidate-count sweeps were separate: the maximum 200 group
columns and 60 candidate variables were not tested together.

To replay a scaling record using the generic, independent covariance formulas:

```bash
python3 replay_saved_path.py results/scaling_path/groups100_917200_crossed.json --data results/scaling_path/groups100_917200_data.npz
```

Add `--backend crossed` to check it with the structured backend. Run additional
verification only after the timed benchmark to avoid CPU contention.

## Stage-five comparison contract

All 300 primary datasets completed for every method. The verified procedure
checked 1186 decisions and accepted 51 pairs; all complete paths replayed and
agreed with the local all-pair BIC path. Fifty pair additions occurred in the
suppressor scenario; one occurred under the correlated complete null.
These are computational guarantees for the stated BIC policy, not guarantees
that every selected variable is a true signal.

The repeated grid has six scenarios and 50 seeds each: signals, weak signals,
a suppressor pair, independent and correlated nulls, and group-aligned predictors.
Training dimensions are `n=300`, `p=12`, with 6 and 8 crossed groups. Test samples
have 1200 observations and entirely new random effects for 24 and 32 groups.
The seed grid, coefficients, tuning values and budgets are in
`results/comparison/protocol.json`; model code hashes freeze the recorded run.
Six development cases with different seeds were excluded from this grid.

Five methods are compared: verified BIC with all pairs, local multistart BIC
with all pairs, local BIC singles, the published ARSS-JPS search rule, and a
cross-validated elastic-net screen followed by an unpenalized ML mixed-model
refit. ARSS uses alpha 0.01, K=3, gate=0.60, L=5. Its archived search function
is unchanged, but its ML and score adapters use the common profile engine;
this is not the original statsmodels numerical benchmark. Source commit and
hashes are recorded in `arss_source.json`.

EN-CV tunes 39 penalty/mixing combinations using three inner splits with groups
held out on BOTH margins. The scaler is fitted inside each split. Off-diagonal
cells touching only one held-out margin are omitted from that split; validation
sets do not cover the whole sample. Selection functions receive training arrays
only. No test responses are used for selection or tuning. The EN screen is a
marginal linear estimator, not a penalized-LMM likelihood implementation.

Every completed selected support receives the same type of local ML refit for
fixed-effect predictions in new groups. Both population-mean MSE and response
MSE are retained, as are native shrunken EN prediction errors. A final local
coefficient refit does not extend the computational certificate for support
decisions. FDR is the mean dataset FDP; TPR under the complete null is undefined.
Wilson intervals and Monte Carlo SEs use datasets as the independent units.

One fresh process handles each dataset, with independent model objects per
method, rotating order and one BLAS thread. Time includes search, tuning,
certificate checks and final refitting, but excludes imports, independent
replay, held-out scoring and file I/O. No per-method peak RSS is estimated here.
Local domains expand on an active upper boundary; they have no global guarantee.
The verified search has a 30-second soft iteration budget per family and a
2500-split cap. Workers have a separate 360-second timeout. Failure denominators
are retained; partial supports are never promoted to completed selections.

`run_comparison_experiments.py` resumes existing records only when the protocol,
data hashes and frozen source hashes agree. Use a different output directory
for a changed protocol. Exact elapsed times are machine/load dependent; raw
outliers are kept. Do not equate the different stopping rules to the same
statistical error level or interpret these results as a comparison with glmmsel.

A separate EN grid sensitivity retains the primary 13 penalties, extends to 10,
and adds 1e6 as an intercept-only candidate; its zero-coefficient KKT condition
is checked on all training splits. This uses the same 300 inputs and inner
splits after the primary timed run. It does not overwrite primary results or
contribute timings to the five-method comparison. Results and paired differences
are in `results/enet_sensitivity/`.


## Native penalized-LMM comparison (stage seven)

Install R >= 4.1 and R packages Rcpp, RcppArmadillo, ggplot2 and jsonlite.
Then install the archived, unchanged `vendor/glmmsel/glmmsel_1.0.3.tar.gz`
with `R CMD INSTALL`. The recorded R version is 4.5.3; package versions and
BLAS information are in `R_requirements.json` and each native `R_sessionInfo.txt`.
The exact conda package list used here is archived under `vendor/glmmsel/`.
No installed R environment or compiled platform-specific package is bundled.

With BLAS threads fixed as above, run:

```bash
# Optional when Rscript is not on PATH:
# export CSDA_RSCRIPT=/absolute/path/to/Rscript
python3 run_penalized_lmm_comparison.py prepare
python3 run_penalized_lmm_comparison.py run
python3 run_glmmsel_grid_sensitivity.py
python3 summarize_penalized_lmm.py
python3 summarize_glmmsel_sensitivity.py
python3 build_csda_manuscript.py
pdflatex -interaction=nonstopmode -halt-on-error CSDA_manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error CSDA_manuscript.tex
```

To regenerate documents from the stored results, start at
`summarize_penalized_lmm.py` in that sequence. The sensitivity summarizer appends
its section after the primary report has been regenerated. Earlier stage
summarizers write their own historical reports; run stage-seven summaries last
to restore the current report. Frozen protocols and input hashes are checked
before fitting. Completed results are reused; retain an untouched copy of this
archive before running a different protocol.

The 46 primary analyses comprise 40 new independent simulations and six reused
cake input partitions (five outer folds and one full-data fit). All native and
verified analyses completed; the latter verify 223 decisions and all replay.
All 46 extended-grid analyses also completed. The second grid adds the last
penalty divided by 10 and 100, and zero, to every original grid. It changes
three selected supports. These are repeated fits on the same inputs, not 46
additional independent data sets.

Native glmmsel uses a broader diagonal random-slope covariance class than the
reference random-intercept model. Its post-selection ML row removes those
random slopes and refits the selected predictor set, so it is a different
estimator. Native fits are not certified global optima. New-group predictions
and profile values at the native selected parameters were checked independently
with grouped covariance matrices. The separate API pilot in
`results/penalized_lmm_diagnostics/` is excluded from all reported comparisons.


## Stage eight: difficult decision diagnostics

The main manuscript contains decision-gap and initialization diagnostics, an analytic balanced-group reference, and a scientific figure.

The unchanged verifier uses a separate diagnostic budget of 500 splits and eight
soft seconds per family at 128 bits. Three controlled residual draws generate
ten dependent gap perturbations each. The exact-tie input is a separate symbolic
control. Twenty-four independent crossed datasets each compare three single
additions with the base support under nine individual starts and the previous
multistart rule. The eight development inputs are excluded from reported counts.

All 42 accepted decisions pass independent replay. Twelve nonzero-gap inputs
and the exact tie are unresolved online; later replay verifies nine of those
nonzero-gap witnesses. The other three contain a wrong proposal, which is not
accepted by either checker. This does not alter their original online status.

Four of 216 single-start family choices disagree with the certified choice,
on two datasets. All contain nonstationary fits despite success flags; a dense
post-hoc check confirms their objectives and gradients. The previous multistart
agrees with all 24 crossed certificates. No distinct stationary local minima
are established by this experiment.

Scientific runner and input hashes are frozen under `results/difficult_decisions/`.
The post-hoc dense diagnostic has its own source and primary-result hashes.
Summary construction never fits a model. Reproduction from this directory:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 run_difficult_decisions.py prepare
python3 run_difficult_decisions.py run
python3 check_difficult_local_fits.py
python3 replay_difficult_decisions.py
python3 summarize_difficult_decisions.py
python3 build_csda_manuscript.py
```

The runner resumes existing results. To reproduce primary timings in a clean
checkout, preserve the archived results and run with a fresh results directory.
The `--pilot` flag selects only the excluded development cases.
`figures/difficult_decisions.svg` is the exportable vector figure; PNG is used
in the manuscript. The aggregate does not pool dependent gap perturbations
with independent simulation replications or with earlier full-path experiments.
