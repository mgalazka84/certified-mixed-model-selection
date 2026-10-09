# Certifying variable selection decisions in Gaussian mixed models

Computational materials accompanying the manuscript by **Marek Gałązka and Hanna Wdowicka** prepared for *Computational Statistics & Data Analysis*.

The procedure constructs a checkable certificate for each declared single-addition, pair-addition or stopping decision while re-estimating variance components. It reports an unresolved outcome when it cannot obtain a certificate. The guarantee applies to the specified model family and selection policy; statistical support recovery and prediction are evaluated separately.

## Version 1.0.1 and complete download

The complete reproducibility archive is attached to [release v1.0.1](https://github.com/mgalazka84/certified-mixed-model-selection/releases/tag/v1.0.1) as **CSDA_GitHub_package.zip**. Download and extract this asset to obtain all exact inputs, stored results, binary package/data files and figures before running the examples below. The release also provides the manuscript PDF and an archive checksum file.

The repository tree provides browsable source code, documentation and manuscript tables. GitHub's automatic **Source code** downloads contain that source tree; use **CSDA_GitHub_package.zip** for the full recorded study.

## Start here

- `CSDA_manuscript.pdf`: current manuscript; `CSDA_manuscript.tex`: LaTeX source.
- `REPRODUCIBILITY.md`: full protocols, recorded comparisons, numerical contract and reproduction commands.
- `certified_global_path.py`: certification on the entire nonnegative variance-ratio domain.
- `certified_path.py`: compact-box engine and witness verification.
- `certified_crossed_backend.py`: structured verified algebra for two crossed random intercepts.
- `replay_saved_path.py`: check stored decisions without repeating model selection.
- `results/`: recorded outputs, exact simulation inputs, witnesses and frozen source hashes.
- `data/real/`: public source data, provenance and attribution.
- `SHA256SUMS.json`: checksums of the distributed files.

## Installation and a saved-witness replay

Use a Python environment compatible with `requirements.txt`:

```bash
python3 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python3 -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 replay_saved_path.py results/global_path/high_variance_915260.json
```

The example uses Linux/macOS syntax for the thread settings. Set the equivalent environment variables in other shells before starting Python. Recorded Python and R versions are stored in the environment and session files. The native penalized-LMM comparison additionally requires R and the package sources under `vendor/glmmsel/`; installation instructions are in `REPRODUCIBILITY.md`.

## Rebuild the manuscript

The saved tables and figures are already included. With LaTeX installed:

```bash
python3 build_csda_manuscript.py
pdflatex -interaction=nonstopmode -halt-on-error CSDA_manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error CSDA_manuscript.tex
```

The builder extracts mathematical sections from the retained historical source `CSDA_covariance_screening_note.tex`. The current journal manuscript contains the updated bibliography and declarations. Rerunning experiments is unnecessary for this build. Full experimental reproduction is documented separately and can take substantially longer than witness replay.

## Recorded evidence and scope

The main comparison contains 300 independent datasets and 1186 verified path decisions. Other directories hold the matched backend comparison, two real-data applications, native penalized-LMM benchmarks and difficult-decision diagnostics. Dependent gap perturbations and repeated fits are distinguished from independent datasets in the manuscript. Original online outcomes remain distinct from later witness replay.

The covariance structure is fixed. Positive projection residual floors and suitable separation are required by the stated guarantees. The budgeted implementation may leave decisions unresolved. The procedure does not establish global best-subset selection or post-selection error control.

## Related published method

The ARSS-JPS comparator is from [mgalazka84/arss-jps-mixed-model-selection](https://github.com/mgalazka84/arss-jps-mixed-model-selection), commit `0cdcf976178be7ed00af4b0737a7ed97540786a4`, accompanying [Applied Sciences 16(20), 10010 (2026)](https://doi.org/10.3390/app162010010). The present certificate procedure and reference selection rule are distinct from that publication's shortlist and likelihood-ratio policy.

## Attribution and research workflow

Third-party license notices and provenance are retained, including `ARSS_POLICY_LICENSE`, the original glmmsel source license and the real-data source records. OpenAI ChatGPT/Codex assisted with code, manuscript preparation and the research workflow as described in the manuscript. Numerical results and figures were generated by the archived Python/R/Matplotlib workflows from stored inputs.

Use `CITATION.cff` for the software authors and title. The cited Applied Sciences DOI identifies the related ARSS-JPS article, not this software or a CSDA publication.
