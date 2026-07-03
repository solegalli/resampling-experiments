# Hard datasets

These three datasets extend the analysis to cases where **standard ensembles
struggle**. The datasets in `functions/data.py` are mostly well separated (a
single feature or a plain ensemble already discriminates the classes), which
leaves little room for resampling to help. The datasets below were chosen for
the opposite reason, to test whether resampling-based "special" ensembles
(RUSBoost, EasyEnsemble, BalancedRandomForest) add value where ordinary
ensembles do not perform well.

They are loaded through `functions/hard_data.py`:

```python
from functions.hard_data import DATASETS_HARD, load_hard_dataset
X_train, X_test, y_train, y_test = load_hard_dataset("diabetes130")
```

## Selection criteria

- **Naturally binary** task (no arbitrary one-vs-rest binarisation of a
  multiclass problem), to avoid the pseudo-replica issue.
- **Imbalanced** minority class.
- **Hard**: no single feature separates the classes (max univariate AUC < 0.90,
  the screen from `notebooks/datasets-with-perfect-separation.ipynb`) and the
  best standard ensemble reaches only a modest ROC-AUC.
- **Permissive license**: all three are distributed under **CC BY 4.0**, which
  permits commercial use with attribution.

## Datasets

| Name | Source | License | Instances | Features | Positive class | Positive rate |
|------|--------|---------|-----------|----------|----------------|---------------|
| `diabetes130` | UCI id 296 — Diabetes 130-US hospitals (1999-2008) | CC BY 4.0 | 101,766 | 47 | Readmitted within 30 days (`<30`) | ~11% |
| `default_credit` | UCI id 350 — Default of Credit Card Clients | CC BY 4.0 | 30,000 | 23 | Defaults next month | ~22% |
| `secom` | UCI id 179 — SECOM | CC BY 4.0 | 1,567 | 590 | Manufacturing failure | ~6.6% |

### diabetes130 — hospital readmission (medical)
- **Target**: `readmitted` has three values (`NO`, `>30`, `<30`). We use the
  clinically meaningful, imbalanced binarisation: positive = readmitted within
  30 days (`<30`); negative = everything else. This is the single standard
  framing of the problem, not an arbitrary class split.
- **Preprocessing**: 36 categorical columns are arbitrary ordinal-encoded;
  missing values are kept as an explicit `"Missing"` category; 2 constant
  columns (`examide`, `citoglipton`) are dropped.
- **Why hard**: 30-day readmission is weakly predictable from administrative and
  diagnostic features; published models report ROC-AUC ~0.64-0.70.

### default_credit — credit-card default (finance)
- **Target**: binary default indicator (already 0/1).
- **Preprocessing**: all 23 features are numeric with no missing values; used
  as-is.
- **Why hard**: behavioural default has substantial class overlap; standard
  models plateau around ROC-AUC ~0.77.

### secom — semiconductor manufacturing yield (industrial)
- **Target**: pass (`-1`) / fail (`1`) re-encoded to 0/1 (same convention as the
  imbalanced-learn datasets in `functions/data.py`).
- **Preprocessing**: 590 numeric sensor signals; missing values are flagged out
  of sample (EndTailImputer at 3x the feature maximum) rather than imputed with a
  central value, so tree models can use "not measured" as a signal; constant
  signals are dropped. Missingness is mixed: most sensors are <5% missing
  (sporadic) but ~28 are >=50% missing (structural).
- **Why hard**: most sensors are uninformative noise and the failure rate is
  very low (~6.6%); achievable ROC-AUC is typically ~0.60-0.70.

## Provenance and reproducibility

`diabetes130` and `default_credit` are fetched with the
[`ucimlrepo`](https://pypi.org/project/ucimlrepo/) package by id. `secom` is not
available through the `ucimlrepo` API, so it is downloaded directly from the UCI
static file server (`https://archive.ics.uci.edu/static/public/179/secom.zip`).

Downloads are cached under `.data_cache/` (git-ignored) so the data is fetched
only once.

## Attribution

These datasets are released under CC BY 4.0; please cite the UCI Machine Learning
Repository and the original dataset creators when using them:

- Diabetes 130-US hospitals: Strack et al. (2014), *BioMed Research International*.
- Default of Credit Card Clients: Yeh, I-C. (2009), *Expert Systems with Applications*.
- SECOM: McCann, M. & Johnston, A. UCI Machine Learning Repository.
