# Strong-signal, severely imbalanced datasets

These datasets **complement** the hard datasets in
`docs/hard_datasets.md`. The hard datasets are challenging because the features
don't separate the classes well, so a machine learning models may struggle to discriminate among the classes. The datasets here are the
opposite: a standard ensemble already discriminates the classes well
(ROC-AUC > 0.9), but the positive class is extremely **rare**.

They are loaded through `functions/imbalanced_data.py`:

```python
from functions.imbalanced_data import DATASETS_IMBALANCED, load_imbalanced_dataset
X_train, X_test, y_train, y_test = load_imbalanced_dataset("creditcard")
```

## Datasets

| Name | Source | License | Instances | Features | Positive class | Positive rate |
|------|--------|---------|-----------|----------|----------------|---------------|
| `htru2` | UCI id 372 — HTRU2 | CC BY 4.0 | 17,898 | 8 | Pulsar | ~9.2% (1,639) |
| `creditcard` | OpenML id 1597 — credit-card fraud | Public / DbCL v1.0 | 284,807 | 29 | Fraud | ~0.17% (492) |

### htru2 — pulsar detection (astrophysics)
- **Target**: `class` is already binary (1 = pulsar). 8 numeric features
  summarising the integrated pulse profile and the DM-SNR curve. No missing
  values, no encoding required.
- **Why strong signal**: the eight statistics separate pulsars from
  radio-frequency interference well; standard ensembles reach ROC-AUC ~0.97.

### creditcard — credit-card fraud (finance)
- **Target**: `Class` (1 = fraud). 28 PCA components (`V1`–`V28`) plus the
  transaction `Amount`; the OpenML version already excludes the raw time column.
  No missing values.
- **Why strong signal but extreme**: only 0.17% of transactions are fraud (492
  of 284,807), yet the PCA features are highly discriminative and standard
  ensembles reach ROC-AUC ~0.97. This is the textbook tiny-minority case.
- **Full dataset**: the loader uses the entire 284,807-row dataset (~0.17%
  positive). Down-sampling only the majority (as an earlier version did) is itself
  a form of random undersampling -- exactly what these experiments set out to
  test -- so it would confound the comparison. The full comparison (all standard
  and special ensembles, the undersamplers and the IHT sweep) is run on the whole
  dataset; the successive-halving search on 199k rows makes this slow but
  tractable.

## Provenance, reproducibility and licenses

`htru2` is fetched with [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) by id;
`creditcard` is fetched from OpenML via `sklearn.datasets.fetch_openml`. Both are
cached under `.data_cache/` (git-ignored) so each is fetched only once.

- **htru2**: CC BY 4.0 — sharing and adaptation for any purpose (including
  commercial) with attribution. Cite Lyon et al. (2016), *MNRAS*, and the UCI ML
  Repository.
- **creditcard**: OpenML lists the license as "Public". The widely used Kaggle
  distribution (`mlg-ulb/creditcardfraud`) applies the Open Data Commons
  **Database Contents License (DbCL) v1.0**, which permits commercial use of the
  database contents. The data comes from a collaboration between **Worldline**
  and the **Machine Learning Group at Université Libre de Bruxelles**. Cite
  Dal Pozzolo et al. (2015), *Calibrating Probability with Undersampling for
  Unbalanced Classification*, IEEE CIDM.
  - Note: this is the one dataset in the project whose license is "Public/DbCL"
    rather than the cleaner CC BY 4.0 of the others. DbCL v1.0 does permit
    commercial use; if a strictly CC-licensed corpus is required, drop this
    dataset and keep `htru2`.
