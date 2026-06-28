# Datasets for the undersampling-for-speed experiment

The undersampling-for-speed study (`scripts/training_scripts/train-undersampling-speed.py`,
`notebooks/11-undersampling-for-speed.ipynb`) needs datasets that are large
enough for training time to matter and imbalanced enough that there is majority
data to drop. We use four, spanning the imbalance/size spectrum.

| dataset       | source            | n (total) | features | positive | role |
|---------------|-------------------|-----------|----------|----------|------|
| creditcard    | OpenML id 1597    | 284,807   | 30       | ~0.17%   | huge, extreme imbalance — strongest speed case |
| protein_homo  | KDD Cup 2004      | 145,751   | 74       | ~0.89%   | big, very imbalanced — strong speed case |
| diabetes130   | UCI id 296        | ~101,766  | (encoded)| ~11%     | medium, moderate imbalance — modest speed case |
| isolet        | UCI id 54         | 7,797     | 617      | ~7.7%    | small, high-dimensional — no speed case (contrast) |

`protein_homo` and `isolet` are loaded through `imblearn.datasets.fetch_datasets`,
which downloads the imbalanced-learn benchmark collection from Zenodo. `creditcard`
uses the project's full loader; `diabetes130` uses `functions.hard_data`.

## Licenses

- **creditcard** — OpenML lists it as "Public"; the Kaggle distribution (mlg-ulb)
  uses the Open Data Commons Database Contents License (DbCL) v1.0, which permits
  commercial use. Provenance: Worldline / ULB Machine Learning Group.
- **diabetes130** — UCI Machine Learning Repository (id 296), **CC BY 4.0**
  (commercial use permitted with attribution).
- **isolet** — UCI Machine Learning Repository (id 54), **CC BY 4.0**
  (commercial use permitted with attribution). Donated by Ron Cole and Mark Fanty.
- **protein_homo** — from the KDD Cup 2004 protein homology task, redistributed in
  the imbalanced-learn benchmark collection (Zenodo record 61452). **The data
  license is not explicitly stated.** The imbalanced-learn BSD-3 license covers the
  software, not the data, and the original KDD Cup data required registration to
  download. It is fine as a research benchmark, but its terms for commercial use
  are unclear and should be verified before any commercial use. If a clearly
  commercial-use-OK big dataset is required, it can be swapped out without
  affecting the method.
