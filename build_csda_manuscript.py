"""Assemble the journal manuscript while preserving the historical research note."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
note=(ROOT/'CSDA_covariance_screening_note.tex').read_text()

def between(start,end):
    return note[note.index(start):note.index(end)]

preamble=r'''\documentclass[preprint,11pt,numbers]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,tabularx,array,graphicx,float,enumitem}
\usepackage[colorlinks=true,urlcolor=blue!50!black,citecolor=blue!50!black,linkcolor=blue!50!black]{hyperref}
\usepackage{xcolor}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{remark}\newtheorem{remark}[theorem]{Remark}
\newcommand{\R}{\mathbb R}
\newcommand{\tr}{\operatorname{tr}}
\newcommand{\Span}{\operatorname{span}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\dist}{\operatorname{dist}}
\newcommand{\vertx}{\operatorname{vert}}
\newcommand{\lam}{\lambda}
\newcommand{\calC}{\mathcal C}
\newcommand{\calP}{\mathcal P}
\input{results/scaling_path/scaling_macros.tex}
\input{results/comparison/comparison_macros.tex}
\input{results/csda_validation/manuscript_macros.tex}
\setlist{itemsep=2pt,topsep=4pt}
\setlength{\parskip}{3pt}
\journal{Computational Statistics \& Data Analysis}
\hypersetup{pdftitle={Certifying variable selection decisions in Gaussian mixed models},pdfauthor={Marek Galazka and Hanna Wdowicka}}
\begin{document}
\begin{frontmatter}
\title{Certifying variable selection decisions in Gaussian mixed models}
\author[amu]{Marek Ga\l{}\k{a}zka\corref{cor}}
\ead{galazka@amu.edu.pl}
\author[pue]{Hanna Wdowicka}
\address[amu]{Department of Algorithm Theory and Data Security, Faculty of Mathematics and Computer Science, Adam Mickiewicz University, Pozna\'n, Poland}
\address[pue]{Department of Statistics, Pozna\'n University of Economics and Business, Pozna\'n, Poland}
\cortext[cor]{Corresponding author.}
\begin{abstract}
Changing the explanatory variables in a model for grouped data can also change
the estimated variation between groups. A variable that appears unimportant
may therefore become useful after the model is refitted. We develop checkable
certificates for fixed-effect selection in Gaussian mixed models. Each
certificate proves that the chosen model beats every declared rival while
allowing variance components to be re-estimated, and can be checked without
rerunning optimization. The construction combines shared candidate-family
lower bounds with projection residual floors that exclude unbounded
variance-ratio tails. Interval arithmetic verifies the inequalities; the
procedure reports an unresolved outcome when it cannot certify a decision.
Under stated positive-floor and separation conditions, the bounds cover the
entire nonnegative variance domain. They certify single-variable additions,
joint additions of two variables and stopping under a specified BIC-type policy.
All 1186 path decisions are verified across 300 independent simulated datasets.
A structured implementation for two crossed random intercepts achieves a median
\ScalingRatio{}-fold speedup over the generic verified backend in a matched study.
Two applications and a native $\ell_0$-penalized mixed-model comparator assess
prediction separately from certification cost. Controlled decision gaps expose
completion limits, and varied initializations distinguish premature numerical
stopping from certified decisions. The guarantee concerns execution of the
specified selection policy, rather than best-subset optimality or
post-selection error control.
\end{abstract}
\begin{keyword}
Linear mixed models \sep Fixed-effect selection \sep Safe screening \sep
Variance components \sep Interval arithmetic \sep Reproducible computation
\end{keyword}
\end{frontmatter}
'''

intro=r'''\section{Introduction}
Suppose consumers rate several products and the aim is to identify which
product characteristics explain their preferences. Ratings by the same
consumer, or of the same product, are related. Mixed models account for these
two sources of variation, but changing the explanatory variables can also
change the estimated group variation. A variable that looks unimportant in
the current fit can therefore become useful when a competing model is refitted.
Deciding which variable to add---and when to stop---depends on this numerical
comparison. The consumer-preference application below illustrates this setting.

This paper develops a method that supplies a checkable certificate for such
decisions in Gaussian mixed models. A certificate establishes that the chosen
model is better under a specified penalized likelihood than every member of a
declared family of rivals, allowing each model to re-estimate its variance
components. It can be checked from the stored data without rerunning the
optimization. The key is to bound how well each rival could possibly fit:
if none can beat a feasible fit of the proposed winner, the decision is proved.
This avoids requiring an accurately optimized fit of every candidate and gives
the analyst independently checkable evidence for an addition or a stopping
decision. When the available bounds do not separate the models, the procedure
reports that the decision remains unresolved.

The published adaptive re-screening and pair-screened joint rescue procedure
(ARSS-JPS)~\cite{arss} reduces search cost through repeated predictor ranking,
shortlist refitting and joint rescue of correlated pairs. The new contribution
is a certificate for a complete candidate-family comparison across the entire
nonnegative variance-ratio domain. We make this distinction explicit because
the selection policy also changes: the present method uses an observation-count
BIC-type criterion and complete single or pair families, whereas ARSS-JPS uses
likelihood-ratio thresholds, a correlation gate and conditional pair tests.
The earlier procedure remains a comparator in the experiments.

Three developments make the certificate possible. First, we construct lower
bounds for the profiled Gaussian likelihood that share covariance calculations
across candidate models. Second, candidate-specific projection residual floors
give checkable exclusions of the unbounded variance-ratio tails, extending a
finite search partition to the full domain. Third, interval arithmetic checks
both these bounds and a feasible fit, and records a witness binding the
inequalities, the complete model family and domain coverage to the stored data.
We prove conditional correctness of the family decisions and the resulting
single/pair path, together with existence of a finite strict certificate under
the stated rank, positive-floor and positive-gap conditions. A structured
implementation for two random-intercept components reduces the cost of the
verified linear algebra.

These developments build on established ingredients. Profile and
low-dimensional shape optimization for mixed models are developed by
de Carvalho et al.~\cite{carvalho}; projection arguments enter coercivity
results for variance components~\cite{zhou}. Deterministic global
optimization~\cite{alphabb} and safe screening for penalized linear
regression~\cite{atamturk} provide precedents for excluding candidates through
bounds. Modern mixed-model subset selection includes coordinate descent and
local search with statistical analysis~\cite{thompson}, as well as
relaxation-based methods~\cite{sholokhov}. Methods for large crossed random
effects exploit structure at a scale beyond the dense implementation considered
here~\cite{backfit}. Our contribution is the construction and verification of
the family decision; profiling, supporting planes and matrix identities are
not themselves new.

The empirical evaluation separates the reliability of a numerical decision
from the statistical quality of the selected model. All 1186 decisions in the
main 300-dataset study are verified. Further studies compare computational
backends, increase predictor and group dimensions simultaneously, and evaluate
two applications with selection inside group-held-out training samples.
A one-group benchmark includes a native modern penalized mixed-model selector.
Controlled gaps between models and varied local starts reveal both unresolved
cases and premature numerical stopping. Gradient checks and multistart also
detect or resolve the observed single-start discrepancies; certification adds
a replayable proof over the full variance domain. It guarantees execution of
the declared single/pair policy, not global best-subset selection or better
prediction. More accurate likelihood calculations do not by themselves confer
selective-inference validity~\cite{rugamer}.

\begin{table}[!ht]\centering\small
\begin{tabularx}{\linewidth}{p{.22\linewidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}\toprule
Aspect & Published ARSS-JPS & Present verified procedure\\\midrule
Candidate search & Re-screened shortlist and gated pair rescue & Complete single family; complete pair family when singles stop\\
Criterion & Published joint and conditional likelihood-ratio rules & Explicit observation-count BIC-type penalty\\
Variance refits & Local fits for shortlisted candidates & Feasible proposals and checked bounds over all nonnegative variance ratios\\
Output & Selected support and numerical diagnostics & Selected support with replayable witnesses, or unresolved status\\
Claim & Efficient heuristic search & Correct execution of the specified family decisions\\\bottomrule
\end{tabularx}
\caption{The methodological extension changes both the reference policy and the
computational guarantee. It is not a claim that the two procedures select the
same variables.}
\end{table}
'''

theory=between(r'\section{Model, profile objective and the score discrepancy}',r'\section{Candidate exclusion, implementation and stopping}')
theory=theory.replace(r'\section{Model, profile objective and the score discrepancy}',r'\section{Profile likelihood and candidate bounds}')
theory=theory.replace(r'\section{A convex part of the profiled Gaussian objective}',r'\subsection{Convexity of the log-profiled residual}')
theory=theory.replace(r'\section{A box bound and an explicit correction \texorpdfstring{$B_C$}{B-C}}',r'\subsection{A candidate-family bound on a box}')
theory=theory.replace(r'\subsection{How the bound tightens}',r'\subsubsection{Tightness under subdivision}')
theory=theory.replace(r'\section{Excluding the unbounded variance-ratio tails}',r'\subsection{Excluding unbounded variance-ratio tails}')
theory=theory.replace('The requested GLS-to-refit bound','A GLS-to-refit bound')
theory=theory.replace('The following ingredients are elementary matrix-convexity facts; we include full\nproofs rather than treating them as new statistical phenomena. General background\non supporting planes and log-determinant concavity is in~\\cite{boyd}.',
                      'The following matrix identities establish the bound used below. Background on supporting planes and log-determinant concavity is given in~\\cite{boyd}.')
theory=theory.replace('This bound is implemented\nas an optional ablation. All reported main results use~\\eqref{eq:box}.','Our experiments use~\\eqref{eq:box}.')

exclude=between(r'\begin{corollary}[Correct exclusion]',r'The stage-one, scalar-only prototype')
paththm=between(r'\begin{theorem}[Conditional correctness of a complete path]',r'\begin{corollary}[Conditional improvement of a rescued pair]')
globalpart=between(r'Let $d_C=|C|$ and suppose a verified lower bound',r'\textbf{Computing a floor without a numerical rank assumption.}')
globalpart=globalpart.replace('below.','above.')

algorithm=r'''\section{Verified single-addition and pair-rescue paths}
For a support $M$ of selectable fixed-effect columns, define
\begin{equation}\label{eq:bic}
J(M)=\inf_{\lambda\in\R_+^q}F_M(\lambda)+|M|\log n.
\end{equation}
Always-present base columns, the residual scale and the number of allowed
covariance components contribute common parameter-count terms. We use the
observation-count BIC-type penalty as a precisely specified reference rule.
Its selection consistency in a crossed-effects asymptotic regime is not assumed.

Starting at the base support, the reference rule compares the current model
with all remaining single additions. If a single strictly improves $J$, the
best is added and this phase repeats. If none improves $J$, the rule compares
the current support with all remaining pairs. An improving pair is added and
the search returns to singles; otherwise it stops. The reference retains the
current model on a stopping tie and uses a fixed lexicographic convention for
equal improving candidates. Our strict certificate can leave such ties unresolved.
''' + exclude + paththm + r'''
\subsection{Search, witnesses and stopping}
The implementation separates a fast proposal/search routine from verified
decision checking. Float64 local fits and midpoint evaluations propose feasible
incumbents and partitions. They need not establish global optimization. The
checker starts from the binary input arrays, recomputes the relevant quantities
with Arb ball arithmetic~\cite{arb}, and uses outward enclosures for logarithms,
matrix solves, gradients and penalized objective values.

\begin{enumerate}
\item Construct the current model and the complete candidate family. Obtain a
feasible incumbent from multistart local fitting.
\item Enclose candidate-specific projection residual floors. Find finite
coordinate cutoffs whose verified tail bounds exceed the incumbent.
\item On the finite box, share covariance calculations across candidates.
Maintain supporting-plane bounds at a geometric midpoint and the current-model
fit, retaining valid bounds inherited from parent boxes.
\item Refine the lowest remaining rival bound by binary subdivision, updating
the incumbent whenever a better feasible point is found.
\item Verify full box coverage, every rival comparison and the tail exclusion.
Accept a decision only after strict verified separation; otherwise return an
explicit unresolved result when the search budget is exhausted.
\end{enumerate}

For the finite region, minima over vertices and partition leaves, and maxima
over anchors and parent bounds, preserve the lower-bound property. An upward
enclosure at an incumbent gives a feasible upper bound. Each internal tree node
must have two children covering its parent exactly. A data fingerprint binds the
witness to the stored arrays. A separate replay reconstructs all single/pair
families and their phase order, so removing a rival or a half-box cannot leave
a valid full-path witness.

\subsection{Verified coverage of the full variance domain}
''' + globalpart + r'''
\begin{corollary}[Existence of a finite strict certificate]\label{cor:finite}
Consider a finite candidate family satisfying the positive-floor and positive-rank
conditions of Theorem~\ref{thm:tail}. Suppose its optimal penalized values have a
unique winner and a positive gap $\delta$ to every rival. There exists a finite
box partition and, under arithmetic enclosures converging with precision, a
finite precision at which a strict family certificate can be verified.
\end{corollary}
\begin{proof}
Choose a feasible binary variance point for the winning support whose penalized
value $U$ is less than its optimum plus $\delta/4$. Such points exist by
continuity and density, including limits at zero coordinates. Tail exclusion
gives a finite box containing this point with all outside rival values greater
than $U$. On that box each $h_X$ has a finite Hessian bound. Subdivide until
Theorem~\ref{thm:rate} bounds every box error by less than $\delta/4$.
Each rival's partition lower bound then exceeds the winning optimum plus
$3\delta/4$, and hence exceeds $U$. Only finitely many strictly positive
projection floors, positive-definite solves and strict inequalities remain to
be checked. Increasing enclosing-arithmetic precision sufficiently preserves
all of their strict signs.
\end{proof}
This is an existence result, not a termination or complexity guarantee for
our budgeted best-first search. It assumes a positive gap and access to a
sufficiently good feasible incumbent. The implemented fixed precision and
finite budgets can remain unresolved even when such a certificate exists.

For random intercepts, the base and random-effect columns contain exact linear
dependencies. We select an exact rational spanning basis for their stored binary
values using FLINT row reduction, and recompute projection products in Arb.
Candidate-specific residual floors follow from small Schur complements; a full
exact augmented-rank calculation handles candidates already in the projected
space. Neither a numerical rank threshold nor a saved floor is accepted as a
proof of positive residual distance.

Pilot multistart fitting begins on $[0,4]^q$, with up to six expansions by four
when an upper boundary is active. This only proposes a feasible value. Verified
tail construction then tests powers of two, starting at 4, with at most 50
doublings, until every candidate's bound exceeds that value by 1. The resulting
cutoffs are sufficient proof regions, not parameter estimates. The search
bisects in $\log(1+\lambda_k)$ coordinates, while all supporting planes remain
in the original affine $\lambda$ coordinates.

The numerical contract treats the stored floating-point data as exact binary
numbers; it concerns rounding error, not measurement error. Failed positive-floor
checks, invalid solves, exact ties and exhausted budgets produce an unresolved
outcome. The verifier relies on the stated analytic assumptions and enclosing
arithmetic; this is not formal verification of the complete Python program.
'''

structured=between(r'\section{A structured verifier for two crossed intercepts}',r'\section{Stage-one pilot protocol and observed results}')
structured=structured.replace('The float64 search, BIC policy, variance domain and witness format can remain\nunchanged while the verified numerical backend is specialized.',
                              'The float64 search, BIC policy, variance domain and witness format remain unchanged when the numerical backend is specialized.')

experiments=r'''\section{Simulation studies}
\subsection{Design and reproducibility}\label{sec:researchworkflow}
The main repeated comparison uses 300 independent simulated training datasets:
50 for each of six scenarios. Training samples have $n=300$, $p=12$ candidate
columns and two crossed random intercepts with 6 and 8 groups. Independent test
samples have 1200 observations and entirely new random effects for 24 and 32
groups. Group counts differ by at most one and labels are independently
permuted on the two margins. Independent standard normal predictors are
centered and scaled using training statistics, which also transform the test
predictors. Fixed coefficients are specified on that training-standardized scale.
Residual standard deviation is 1; random-intercept standard deviations are
0.7 and 0.8 unless stated otherwise.

\begin{enumerate}
\item Ordinary signals: $(\beta_1,\beta_3,\beta_5)=(0.8,0.5,0.3)$.
\item Weak signals: $(\beta_1,\beta_3,\beta_5)=(0.35,0.25,0.15)$.
\item Suppression: $\operatorname{Corr}(x_1,x_2)=0.995$ and
$(\beta_1,\beta_2,\beta_3)=(6,-6,0.45)$.
\item Complete null with independent predictors.
\item Complete null with predictor equicorrelation 0.8.
\item Group-aligned predictors: $x_1$ and $x_2$ are respectively origin- and
destination-level normal values plus independent noise of standard deviation
0.1. Random-intercept standard deviations are 2 and 3, and
$(\beta_3,\beta_5)=(0.65,0.4)$.
\end{enumerate}
All unspecified coefficients are zero. Seeds are $918000+1000s+b$, with
scenario index $s=0,\ldots,5$ and replicate $b=0,\ldots,49$. Six separate
technical development cases were excluded. Protocol and source hashes were
frozen before the first main run; this was not a preregistered or external study.

The methods are verified BIC with pairs, local multistart BIC with the same
single/pair rule, local BIC with singles only, the archived ARSS-JPS search policy,
and an elastic-net screen followed by unpenalized ML refitting. The unchanged
\texttt{arss\_family} source at commit \texttt{0cdcf976} uses
$\alpha=0.01$, shortlist size 3, correlation gate 0.60 and pair shortlist size 5.
Adapters connect it to the common profile-ML engine. This is a source-checked
policy comparison, not execution of the original statsmodels backend.

Local fits start from zero, half the box width, and 0.1 and 0.9 times the width;
candidate fits also use the current-model point. The initial upper bound 4
expands by four if active, at most six times. A final active cap is a failure.
Projected-gradient diagnostics are recorded without treating them as a proof
of global optimality. Verified paths use 128-bit Arb arithmetic, a 2500-split
budget and a soft 30-second family search budget. A separate worker timeout is
360 seconds. All completed supports receive a common local ML coefficient
refit for prediction, which is not covered by the support-selection certificate.

Elastic net~\cite{sklearnenet} uses three internal group-held-out splits,
13 log-spaced penalties from $10^{-3}$ to 1, and mixing parameters 0.5, 0.9
and 1. Scaling is fitted inside each split. Training omits both sets of held-out
groups and validation uses their intersection; cells touching just one held-out
margin are omitted from that split. Thus these three validation sets do not
cover all observations. Test outcomes never enter tuning. The elastic-net
screen is a marginal linear model, not a penalized mixed-model estimator.
Native shrunken predictions are also retained.

\subsection{Measurements}
For true support $S_0$ and selected support $\widehat S$, true-positive rate is
$|\widehat S\cap S_0|/|S_0|$ when $S_0\ne\varnothing$. False-discovery proportion
is $|\widehat S\setminus S_0|/\max(1,|\widehat S|)$; its Monte Carlo mean estimates
FDR. We also measure exact support recovery and any false-selection event.
Population-mean MSE compares marginal predictions with the generating fixed
mean, while response MSE includes new random effects and residual noise.
Dataset-level Monte Carlo standard errors are $s/\sqrt B$; binary events receive
95\% Wilson intervals. Comparisons on the same datasets use paired differences.

A fresh process handles each dataset, with one BLAS thread, separate method
objects and a rotating execution order. Time includes preparation, selection,
tuning, certificate checking and final coefficient refitting. Importing,
held-out scoring, additional replay and writing files are excluded. Wall-clock
scheduling variability is retained. Medians and 90th percentiles are reported;
per-method peak memory is not inferred from the shared process. Failed or
unfinished selections are retained and excluded from completed-support metrics
with explicit denominators.

\subsection{Selection and prediction}
All five methods completed all 300 main datasets. The verified method produced
1186 checked decisions, and every completed path passed replay. Its complete
decision sequence agreed with local BIC with pairs in all 300 cases. This
observed agreement does not make the local comparator a global-ML oracle.
There were 51 accepted pairs: 50 in the suppression scenario and one under
the correlated complete null.
\input{results/comparison/comparison_selection_tables.tex}
\begin{figure}[!ht]\centering
\includegraphics[width=\linewidth]{figures/comparison_methods.png}
\caption{Main repeated comparison. Panels A and C show means and one Monte
Carlo standard error; panel B shows event proportions with 95\% Wilson intervals.
Panel D reports median complete-pipeline times. The local-pair series is omitted
from the plot for readability and included in the tables. Replicates are datasets.}
\end{figure}
In the suppression scenario, both active members were recovered in 50/50
verified and ARSS-JPS runs, compared with 0/50 local-single runs. The paired
population-mean MSE difference, verified minus singles, was $-0.360$ with Monte
Carlo standard error 0.007. This measures the benefit of allowing pair additions,
not a statistical benefit from interval arithmetic. ARSS-JPS had lower observed
FDR in this scenario (0.046 versus 0.090) and a similar prediction error.

Under the independent complete null, any false selection occurred in 7/50
verified runs (95\% Wilson interval 7.0--26.2\%) and 5/50 ARSS-JPS runs
(4.3--21.4\%). Under the correlated null the corresponding counts were 3/50
and 2/50. These are scenario-specific empirical error rates, not nominal error
control. The methods use different stopping rules and elastic net is tuned
for prediction; they do not have a matched familywise-error level.
\input{results/comparison/comparison_time_table.tex}
Median paired local-pair/verified elapsed time was 1.64, while median
verified/ARSS-JPS time was 2.60. These are comparisons of complete implementations
with different proposal and domain-expansion mechanisms, not an isolated cost
of Arb arithmetic. A secondary elastic-net sensitivity analysis retained all
original penalties, extended the grid to 10 and added an intercept-only penalty
$10^6$. It changed 13/300 supports, all in the group-aligned scenario; the paired
post-refit mean-MSE difference there was 0.005 with Monte Carlo standard error
0.052. Larger-grid tuning did not establish improved test prediction.

\subsection{Matched numerical-backend comparison}
An earlier matched study compares the generic and structured verified backends
on 18 datasets with $n=800$. One sweep fixes $p=20$ and varies
$(r_1,r_2)$ over $(10,15),(20,30),(40,60),(80,120)$; another fixes
$(r_1,r_2)=(20,30)$ and uses $p=20,40,60$. Each configuration has two ordinary
signal datasets and one suppressor dataset. Both backends use the same search,
domain, starts, precision and budget, and run in fresh single-thread processes
with alternating order. Time includes all verification; peak memory is the
Linux process high-water RSS, including interpreter and imported libraries.

All 18 paired paths completed with 85 verified decisions and six pair additions
per backend. Their complete serialized witnesses were identical. Median paired
speedup was \ScalingRatio{}, and median process peak-RSS reduction was
\ScalingMemoryReduction{}\%. Three datasets per configuration do not estimate
reliable performance distributions. The changes were measured jointly and
are not assigned separate causal benefits.
\begin{figure}[!ht]\centering
\includegraphics[width=.84\linewidth]{figures/scaling_path_pilot.png}
\caption{Matched backend comparison. Points are medians; bars show the range
over three datasets, not confidence intervals. Upper panels vary group counts
at $p=20$; lower panels vary candidate counts with 20 and 30 groups.}
\end{figure}

\subsection{Simultaneous growth of candidate and group dimensions}
The matched study varies the two dimensions separately. A new, fixed extension
uses $(n,p,r_1,r_2)=(600,24,12,18)$ and $(1200,40,40,60)$, with eight independent
ordinary-signal and eight suppressor datasets in each configuration. The
coefficient patterns, random-effect scales and independent new-group test design
match the main study. Seeds, exact data and source hashes are archived.
All five methods are rerun; the certificate has a 45-second soft family budget
and 2500 splits. Elastic net now uses the extended penalty grid and all nine
combinations of three held-out group partitions, so each inner observation is
validated once. This later extension is reported separately from the frozen
300-dataset benchmark.
\input{results/csda_validation/stress_table.tex}
\input{results/csda_validation/stress_findings.tex}
'''

applications=r'''\section{Prediction in two real-data applications}
\subsection{Design and evaluation}
Public teaching datasets provide reproducible applications with different
group structures. We specify the predictor dictionaries and group partitions
before fitting any selection procedure. All transformations use training data;
all selection and tuning are repeated inside each outer training split. Alongside
the five selectors, we fit intercept-only and full-dictionary mixed-model
baselines. Predictions are marginal for unseen groups, without fitted random
effects for those groups. Native elastic-net predictions are reported separately
from its post-selection ML refit.

Every observed outcome enters exactly one outer test cell. We report pooled
response MSE and mean absolute error, with completed-fold and evaluated-row
counts. Training sets overlap and observations share groups, so the folds are
not independent replicates and no naive fold-based standard errors or
significance tests are attached to their differences. Full-data fits describe
selected columns and are never used to generate the out-of-fold predictions.

\subsection{Consumer preferences with crossed grouping}
The \texttt{carrots} dataset distributed with \texttt{lmerTest} contains 103
consumers evaluating 12 products~\cite{lmertestdata}. Three missing preference
scores are excluded, leaving 1233 observations. We use random intercepts for
consumer and product. The outcome is a seven-point rating, modeled here by a
Gaussian working likelihood; the numerical certificate does not validate this
distributional approximation or an ordinal-model interpretation.

The eight candidate columns are the two supplied sensory principal-component
scores, indicators for Gender category 2 and Homesize category 3, and the four
interactions between each sensory score and each indicator. The sensory PCs
come from an independent trained-panel product-characteristic analysis and
are treated as published, externally supplied descriptors. We do not claim to
have retrained PCA inside the folds; the prediction task conditions on these
available descriptors. Other consumer taste ratings and the income variable
are not predictors.

Consumers and products are separately allocated to three balanced random
partitions. For each of their nine combinations, training contains neither
held-out consumer nor held-out product, and testing uses their intersection.
Each test cell contains 135--140 observed scores. The inner elastic-net
validation repeats the complete nine-cell construction within training, with
scaling fitted inside each inner split and the extended penalty grid. The
dictionary permits column selection without interaction hierarchy; this is
predictive feature selection, not a coding-invariant test of whole factors.

\subsection{A split-plot experiment with a continuous response}
The \texttt{cake} dataset distributed with \texttt{lme4} contains 270 cake
breakage angles from three recipes, 15 replicates per recipe and six baking
temperatures~\cite{cakedata}. We use the documented experimental batch
\texttt{recipe:replicate} as a single random-intercept grouping factor with
45 levels. We do not create an additional crossed effect from the replicate
numbering. This application uses the generic one-component verifier.

The five candidates are linear, quadratic and cubic powers of
$(\mathrm{temperature}-200)/25$, and indicators for recipes B and C. Five outer
folds hold out complete batches, with three batches per recipe in each fold;
each test fold has 54 observations. Inner tuning uses three group-held-out
folds. Selection of polynomial columns is not constrained by a hierarchy and
the exercise is predictive rather than a formal test of a complete recipe
or temperature factor. Randomly holding out batches does not establish
performance under a future temporal shift.

\subsection{Observed prediction and numerical checks}
\input{results/csda_validation/real_findings.tex}
\input{results/csda_validation/real_table.tex}
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/csda_real_prediction.png}
\caption{Pooled out-of-fold response MSE relative to the intercept-only mixed
model within each dataset. Each observation is tested once; all selection
occurs within training. These are descriptive comparisons from dependent
group-held-out folds, without confidence bars.}
\end{figure}
\input{results/csda_validation/native_findings.tex}
'''

penalized=r'''\widowpenalty=10000\clubpenalty=10000
\section{Comparison with native penalized mixed-model selection}
\subsection{Model scope and training-only tuning}
We run the unmodified CRAN source of \texttt{glmmsel} 1.0.3, the implementation
associated with Thompson et al.~\cite{thompson,glmmselmanual}. It performs
$\ell_0$-penalized selection using coordinate descent and optional local search.
The package specifies one clustering vector and can select independent random
slopes as well as fixed effects, under its fixed-before-random hierarchy.
It does not expose a separate crossed consumer/product covariance structure.
We therefore use the cake application and new one-group simulations. The
native selector searches a broader covariance class than our fixed
random-intercept reference; its results are a comparison of complete
statistical procedures, not identical-objective optimization timings.

We set the hierarchy parameter to its default $\alpha=0.8$, enable local search,
and use convergence tolerance $10^{-6}$, at most 10000 coordinate iterations
and 100 local-search iterations. The native adaptive path requests up to 100
penalties. Its full-training penalty values are evaluated without midpoint
interpolation in the same three whole-group inner folds used by elastic net.
Each native fit scales the original predictor dictionary and response using
only its training observations. The first eligible minimum of the unweighted
mean validation MSE selects the penalty. Eligibility requires finite full and
inner fits and no returned coordinate/local-search iteration limit at that
penalty. Warnings, all evaluated paths and endpoint selections are retained.
Outer-test outcomes are never supplied to R.

Predictions for unseen groups use the package's native fixed effects. We also
refit the reference random-intercept ML model to its selected predictor set,
reported as \texttt{glmmsel + ML}; this changes the estimator and removes
selected random slopes. A predictor is recorded as selected if its absolute fixed
coefficient exceeds $10^{-7}$ on the training-standardized scale or its
random-slope variance ratio exceeds $10^{-10}$ on that scale. Fixed and random
supports are also archived separately. This post-selection refit is not a
certificate for the native penalized fit.

Each case runs all selectors sequentially in a fresh worker, rotating method
order and fixing BLAS thread counts at one. Native times include full path
fitting and inner tuning, but exclude R startup and package loading; subprocess
wall times are also retained. Outer-test prediction, serialization and additional
independent checks are outside the fitting timers. These are descriptive
implementation measurements on a shared host, not dedicated-CPU benchmarks.

\subsection{Cake application and one-group simulations}
The five cake outer folds and their inner splits are exactly those used above;
all selectors are rerun, with a separate full-data fit for description.
The additional simulations use $n=300$, $p=12$, 30 training groups and 1200
test observations from 60 new groups, with random-intercept standard deviation
0.9 and residual standard deviation 1. Twenty independent datasets use the
ordinary-signal pattern above and twenty use its suppressor pattern with
correlation 0.995. Seeds start at 965000 and 966000, respectively. This is a
separate 40-dataset study, not an enlargement of the crossed-effects study
under the same protocol.
\input{results/penalized_lmm/cake_table.tex}
\input{results/penalized_lmm/simulation_table.tex}
\input{results/penalized_lmm/findings.tex}
\input{results/penalized_lmm_grid_sensitivity/findings.tex}
'''

difficult=r'''\section{Decision gaps and local initialization}\label{sec:difficult}
\subsection{Controlled gaps and an analytic reference}
The preceding studies measure selection and prediction. We now isolate the
numerical decision problem, keeping the existing search and verifier unchanged.
For a balanced random-intercept model with a within-group predictor, the
profile minima have a closed form (\ref{app:balanced}). This gives an
independent reference for the signed penalized gap
$\Delta=J(\{1\})-J(\varnothing)$: positive values prescribe stopping and negative
values prescribe addition. This diagnostic concerns one complete single-addition
family, not a full path or a population recovery rate.

We generate three independent residual draws with $n=96$ and 12 groups of
eight observations. Within each draw the predictor amplitude is calibrated
to target gaps $\Delta=\pm10^{-2k}$ for $k=0,\ldots,4$.
The 30 inputs therefore contain dependent perturbations within each draw.
The closed-form gap is enclosed at 256 bits from the stored binary arrays;
its sign, rather than the requested target alone, defines the reference.
One additional $n=4$ example has a symbolically exact stopping tie.

The diagnostic budget is 500 splits and a soft eight seconds per family,
with 128-bit checking. It is distinct from the larger budgets used in the
full-path studies. After every search we independently replay its saved witness,
including unresolved outputs. This extra calculation is reported separately;
it cannot retroactively count as an online algorithm completion. Single-start
L-BFGS-B fits use starts $0,0.01,0.1,1,4,16,64$, the existing convergence
tolerances, and an initial upper bound 128 with up to six expansions by four.
The earlier adaptive multistart comparator is also retained unchanged.

\input{results/difficult_decisions/gap_table.tex}
\begin{figure}[!ht]\centering
\includegraphics[width=\linewidth]{figures/difficult_decisions.png}
\caption{Designed gap perturbations. Panel A separates online cost, local
multistart cost and additional replay cost. Panel B counts certificates among
the six inputs at each gap magnitude. No population confidence intervals are
assigned to these dependent perturbations.}\label{fig:difficult}
\end{figure}

\subsection{Initialization study with two variance components}
A separate 24-dataset experiment uses $n=120$, three candidate predictors and
two random-intercept components. Six datasets have eight groups on one margin
and ten on the other; six have the same design with the second generating variance
zero. In six further datasets both margins have eight groups and differ in
only six row assignments. The last six use this nearly coincident grouping
with two group-aligned predictors plus independent noise of standard deviation
0.05. The third predictor has coefficient 0.35; the other coefficients are
zero. Residual standard deviation is one and the random-intercept standard
deviations are 0.7 and 0.8 except for the stated zero component.

Each dataset compares the intercept model with all three single additions.
Nine local starts are $(0,0)$, $(0.01,0.01)$, $(1,1)$, $(4,4)$,
$(0.1,10)$, $(10,0.1)$, $(64,0)$, $(0,64)$ and $(64,64)$.
We retain optimizer status, projected-gradient norm, all objective values,
chosen models, online witnesses and replay results. Methods use fresh models,
one BLAS thread and cyclic execution order in a fresh process per dataset.
Times include construction and all in-search checking, but exclude imports,
file operations and additional replay. The full protocol, seeds and source
hashes are archived; eight technical pilot inputs are excluded.

\input{results/difficult_decisions/findings.tex}

\subsection{What prevents a difficult decision from completing?}
The diagnostic separates the analytic guarantee from the implementation's
proposal logic. In the existing search, checking is triggered only when the
floating-point lower-bound margin exceeds $10^{-6}(1+|U|)$, where $U$ is the
incumbent value. Floating bounds additionally carry a scale-dependent
$2\times10^{-8}$ cushion, and an incumbent update requires an improvement
greater than $10^{-7}$ at a sampled point. These conservative search thresholds
can block a check or an update at a smaller genuine gap. Subdivision can then
exhaust its budget or stagnate at floating-point resolution.

The nine successful replays of unresolved nonzero-gap cases show that the
stored partition and 128-bit checker can already establish some decisions
that the online search does not report. This supports a specific future change
to checker scheduling and incumbent updates; it is not evidence that merely
raising precision or the split budget solves every difficult case. No search
threshold was changed in this experiment. The practical guarantee remains
conditional on successful verification, with an explicit refusal to certify
inadequately separated or incorrect proposals.
'''

discussion=r'''\section{Discussion}
The method addresses a reproducibility problem in computational statistics:
whether the realized selection path follows a stated criterion after covariance
re-estimation. A family-level certificate can answer that question without
requiring a globally optimized value for every candidate. A completed path has
no improving single or pair addition under the specified rule. It may still
miss a better subset reachable by deletion, swapping or a larger joint addition,
and it may select noise variables. These distinctions are substantive: pair
rescue helps recover suppressed effects, whereas checking arithmetic changes
the strength of the computational claim.

The experiments make the cost-benefit tradeoff concrete. Shared lower bounds
can reduce the cost of exhaustive local candidate refits, and the structured
backend improves matched time and memory. The archived ARSS-JPS policy remains
faster in the main repeated study and often has fewer false discoveries. Native
shrunken elastic-net predictions can also outperform a post-selection ML refit
in a real-data task. A numerical certificate is therefore not a reason to
prefer a selected model independently of its predictive objective.

The additional diagnostics identify a concrete use and a concrete limitation.
Single-start success flags can accompany a wrong family choice after premature
stopping; ordinary gradient checks and multistart also expose or resolve the
observed cases. The certificate supplies the stronger statement that all
rivals are excluded over the full variance domain, and this statement can be
replayed without optimization. At very small gaps the current search can fail
to request a check that would succeed, or retain a proposal that must be
rejected. Thus the value is an auditable decision with explicit abstention
when verification fails. The experiments do not establish frequent errors of
a well-diagnosed multistart implementation.

The present implementation targets few variance components. Vertex enumeration
scales as $2^q$, pair enumeration as $O(p^2)$, and the core linear algebra is
dense in the random-effect dimension. Positive projection residuals are a
sufficient, not necessary, condition for the tail certificate; high-dimensional
or exactly interpolating augmented designs can invalidate it. Fixed 128-bit
precision and finite search budgets can leave close decisions unresolved.
Sparse covariance algebra, adaptive precision and alternative tail conditions
are meaningful extensions, but are not established by these experiments.

The statistical regime also matters. Observation-count BIC is an explicit
reference penalty, not a proved crossed-effects calibration. The native
\texttt{glmmsel} comparison supplies evidence for one-group data and a broader
random-slope class. It does not establish a ranking for crossed covariance
structures or under identical statistical penalties. Further work should
include additional penalized selectors with a fixed covariance structure and
larger rare-event simulation designs. Extending our certificate to selected
random slopes, ordinal responses and selection-aware inference requires its
own analysis.

\section*{Data and computational materials}
The implementation, complete protocols, exact simulated arrays, public
real-data source files with hashes and attribution, raw results, figures and
path witnesses are available from the
\href{https://github.com/mgalazka84/certified-mixed-model-selection}{project repository on GitHub},
release \href{https://github.com/mgalazka84/certified-mixed-model-selection/releases/tag/v1.0.1}{\texttt{v1.0.1}}.
The complete reproducibility archive is supplied as a release asset; the
repository tree contains the source code, documentation and manuscript tables.
The source-checked
ARSS-JPS comparison uses the public
\href{https://github.com/mgalazka84/arss-jps-mixed-model-selection}{ARSS-JPS GitHub repository}
at commit
\href{https://github.com/mgalazka84/arss-jps-mixed-model-selection/tree/0cdcf976178be7ed00af4b0737a7ed97540786a4}{\texttt{0cdcf976}};
its source retains the MIT license. Public real-data sources and their original
documentation are identified in the data manifest. Commands regenerate the
reported summaries and replay stored paths independently of optimization.

\appendix
\section{Verification diagnostics and unsuccessful cases}
Independent small-matrix checks compare the verified algebra with exact
rational reference computations, covering zero variance components, either
ordering of group sizes, unused groups and disconnected group-incidence
structures. Eighty structured-backend combinations were checked at 128 bits
against exact matrices and 256-bit logarithmic enclosures. Fresh-file replay
also cross-checks witnesses with the generic and structured backends.

Deliberately corrupted data fingerprints, incomplete box partitions, missing
candidates and incorrect winners were rejected. Re-labelling a compact-box
witness as a full-orthant witness failed tail checking. Exact ties and zero
projection-floor examples remained unresolved. A nearly collinear point solve
failed at 64 bits and succeeded at 192 bits; this is not evidence that every
ill-conditioned full path can be certified. The full mathematical development,
development pilots and detailed negative tests are retained in the accompanying
research note. Its exploratory timing ratios are not pooled with the matched
or repeated studies reported here.

\section{Balanced-group analytic reference}\label{app:balanced}
Let $n=gm$ observations form $g$ groups of equal size $m$, with an always-present
intercept and a predictor $x$ satisfying $Z^\top x=0$. Let $\bar y_k$ and
$\bar y$ denote the group and overall means, and define
\[
B=m\sum_{k=1}^g(\bar y_k-\bar y)^2,\qquad
W_0=\sum_i(y_i-\bar y_{k(i)})^2,\qquad
W_1=W_0-\frac{(x^\top y)^2}{x^\top x}.
\]
Assume $W_0,W_1>0$. On the group subspace $R=I+\lambda ZZ^\top$ has eigenvalue
$r=1+m\lambda$; on its orthogonal complement it is the identity.
Because $x$ is a within-group contrast, fitting it changes only the within-group
residual. For $j=0,1$ the profiled objective is therefore
\[
F_j(r)=g\log r+n\log\{(W_j+B/r)/n\},\qquad r\geq1.
\]
Differentiation shows that its minimizing ratio is
$r_j^*=\max\{1,(n-g)B/(gW_j)\}$. The determinant exponent is $g$, since the
criterion is ML rather than REML. When both minima are interior,
\[
F_j^*=g\log(B/g)+(n-g)\log\{W_j/(n-g)\},\qquad
\Delta=(n-g)\log(W_1/W_0)+\log n.
\]
At the boundary, $F_j^*=n\log\{(W_j+B)/n\}$. These formulas are evaluated
directly from exact stored binary group sums using 256-bit enclosing arithmetic,
independently of the Woodbury implementation. Evaluating the generic Arb
profile at the rounded analytic minimizers agrees to within
$2.33\times10^{-32}$ in the recorded point checks.

For the exact-tie control take groups $(1,1,2,2)$,
$x=(1,-1,1,-1)^\top$ and $y=(4,0,-2,-2)^\top$.
Then $B=16$, $W_0=8$, $W_1=4$, and both minima are interior.
Consequently $\Delta=2\log(1/2)+\log4=0$ exactly, while the projection
floor for the augmented model remains positive.

\begin{samepage}
\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
OpenAI ChatGPT/Codex assisted with code development and manuscript organization,
drafting and editing. Responsibility for the scientific content rests with
the authors.
\end{samepage}

\clearpage
'''

bib=between(r'\begin{thebibliography}',r'\end{document}')
bib=bib.replace('Supplied revised manuscript,\n2026. Publication reported by the author; the supplied PDF contains placeholder\npublication metadata, so no DOI is inferred here.',
                '\\emph{Applied Sciences}, 16(20):10010, 2026. '
                '\\href{https://doi.org/10.3390/app162010010}{doi:10.3390/app162010010}.')
bib=bib.replace('\\href{https://arxiv.org/html/2506.20425v3}{arXiv:2506.20425v3}, version dated May 2026.',
                '\\emph{Journal of Computational and Graphical Statistics}, 2026. '
                '\\href{https://doi.org/10.1080/10618600.2026.2701133}{doi:10.1080/10618600.2026.2701133}. '
                'Full-text version inspected: \\href{https://arxiv.org/html/2506.20425v3}{arXiv:2506.20425v3}.')
bib=bib.replace(r'\end{thebibliography}',r'''\bibitem{lmertestdata} A. Kuznetsova, P. B. Brockhoff and R. H. B. Christensen.
\emph{lmerTest: Tests in Linear Mixed Effects Models}. Package documentation,
version 3.2-1, 2026. \texttt{carrots}: Consumer Preference Mapping of Carrots.
\href{https://cran.r-project.org/web/packages/lmerTest/lmerTest.pdf}{Official manual and dataset description}.
\bibitem{cakedata} lme4 developers. \emph{Breakage Angle of Chocolate Cakes}.
\texttt{cake} data documentation; original experiment attributed to Cook (1938)
and reported in Cochran and Cox (1957).
\href{https://github.com/lme4/lme4/blob/master/man/cake.Rd}{Official source documentation}.
\bibitem{statsmodels} Statsmodels developers. \emph{Variance Component Analysis}.
\href{https://www.statsmodels.org/stable/examples/notebooks/generated/variance_components.html}{Official examples for nested and crossed models}.
The independent validation uses version 0.14.6.
\end{thebibliography}''')
# Remove the uncited abstract-only screening reference from the journal text.
bib=re.sub(r'\\bibitem\{ghosh\}.*?(?=\\bibitem)', '', bib, flags=re.S)
body=preamble+intro+theory+algorithm+structured+experiments+applications+penalized+difficult+discussion
def expanded_input(match):
    path=ROOT/match.group(1)
    return path.read_text() if path.is_file() else ''
citation_text=re.sub(r'\\input\{([^}]+)\}',expanded_input,body)
keys=[]
for group in re.findall(r'\\cite\{([^}]+)\}',citation_text):
    for key in group.split(','):
        if key not in keys:keys.append(key)
entries={key:content for key,content in re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',bib,flags=re.S)}
assert set(keys)<=set(entries),'A cited reference has no bibliography entry'
bib='\\begin{thebibliography}{99}\\small\n'+''.join('\\bibitem{'+k+'}'+entries[k] for k in keys)+'\\end{thebibliography}\n'
(ROOT/'CSDA_manuscript.tex').write_text(body+bib+'\n\\end{document}\n')
print('Wrote CSDA_manuscript.tex')
