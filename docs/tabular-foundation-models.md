# Tabular foundation models: problems encountered

We tried to evaluate open tabular foundation models on the 17 datasets where the best ensemble has a ROC-AUC below 0.95. This log lists every problem found along the way. The analysis is in `notebooks/10-analysis-of-foundation-models.ipynb`.

**Machine:** Apple M4, 16 GB of unified memory, no CUDA GPU, macOS, Python 3.14. PyTorch's MPS backend can use at most about 8 GB of that memory.

## Outcome per model

| Model | Result |
|---|---|
| TabPFN-2 | Ran on the 13 datasets within its limit of 10,000 training rows. It can't be used on the other 4. |
| TabICLv2 (`tabicl` 2.2.0, checkpoint `tabicl-classifier-v2-20260212.ckpt`) | Ran on the 4 smallest datasets, then ran out of memory on `secom` (1,096 rows × 478 features). Not completed. |
| TabDPT | Ran on 16 of 17 datasets at low CPU priority, without memory problems: 1.5–25 s per small dataset, and 4–9 minutes on `default_credit`, `bank-marketing` and `adult` (21k–34k training rows). Stopped after 10 minutes on `diabetes130` (71k training rows, 30k test rows). |
| LimiX-2M | Cannot run on macOS. |
| AutoGluon | Cannot be installed on Python 3.14. |

## Problems

### Limits on dataset size

- **TabPFN-2** is limited to 10,000 training rows, 500 features and 10 classes. So it can't be used on 4 of the 17 datasets: `default_credit` (21k training rows), `bank-marketing` (32k), `adult` (34k) and `diabetes130` (71k). TabArena applies the same rule: models are only run on datasets within their limits.
- The newer TabPFN versions raise these limits, but their weights are not open (see licenses below).

### Memory and hardware

- **TabICL on Apple MPS** ran out of memory on `secom`, a small dataset with many features:
  - With the default 8 ensemble members per batch, it needed more than the 8.3 GB MPS limit.
  - With 1 member per batch, Metal tried to allocate an 11.5 GB buffer and crashed the Python process (`failed assertion: Failed to allocate private MTLBuffer`).
  - On CPU, it slowed the whole computer down until it couldn't be used for other work, so the run was stopped.
- **Conclusion:** on a laptop with 16 GB of memory and no NVIDIA GPU, TabICL can't be run on datasets with hundreds of features. Of the large datasets (21k–71k rows), only TabDPT was tried, and it could not finish `diabetes130` within 10 minutes.
- **TabDPT** fails on MPS with float64 data ("MPS framework doesn't support float64"), so the data must be cast to float32. Flash attention and `torch.compile` are on by default and need CUDA, so both must be turned off.
- **LimiX-2M** cannot run on macOS:
  - it imports a `triton` kernel unconditionally (`model/v1_0/layer.py`), and triton has no macOS builds;
  - it always creates a cache folder under `/mnt/public/infe_cache`, even when caching is off, which fails on macOS;
  - the README example sets distributed-training environment variables (`RANK`, `WORLD_SIZE`, …) even on a single machine;
  - the README lists retrieval configs (`cls_default_retrieval.json`) that are not in the repository.

### Library compatibility

- **AutoGluon** supports Python 3.10–3.13. On Python 3.14, pip silently falls back to an AutoGluon release from around 2020, which needs scipy < 1.5, and the install fails. AutoGluon was dropped.
- **LimiX** is not on PyPI. It has to be cloned from GitHub and installed with `pip install -e`.
- **APIs differ between models:**
  - TabPFN and TabICL have a scikit-learn API.
  - TabDPT has one too, but its ensemble prediction is a separate method (`ensemble_predict_proba`), and it only accepts NumPy arrays.
  - LimiX has no scikit-learn API: a single `predict(x_train, y_train, x_test)` call does everything.
- **TabPFN:** the `tabpfn` package now uses TabPFN-3 by default, which needs a Prior Labs login and token. TabPFN-2 has to be requested explicitly (`create_default_for_version(ModelVersion.V2)`).

### Licenses

The licenses vary, and several are not open source. Please check them before publishing results.

| Model | Weights | Code |
|---|---|---|
| TabPFN-2 | [Prior Labs License](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt) (Apache 2.0 with attribution) | [Apache-2.0](https://github.com/PriorLabs/TabPFN/blob/main/LICENSE) |
| TabPFN-2.5, 2.6, 3, 3.5 | Non-commercial, gated behind a Prior Labs login (e.g. [3.5](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)). Not used. | as above |
| TabICLv2 | Not stated separately | [BSD-3](https://github.com/soda-inria/tabicl/blob/main/LICENSE) |
| TabDPT | [Apache-2.0](https://huggingface.co/Layer6/TabDPT) | [Apache-2.0](https://github.com/layer6ai-labs/TabDPT-inference/blob/main/LICENSE) |
| LimiX-2M | [LimiX-2M license](https://huggingface.co/stableai-org/LimiX-2M/blob/main/LICENSE.txt) (commercial use needs authorization) | [Stable AI License](https://github.com/limix-ldm-ai/LimiX/blob/main/LICENSE.txt) (derived from Apache 2.0) |
| AutoGluon | n/a | [Apache-2.0](https://github.com/autogluon/autogluon/blob/master/LICENSE) |

## What worked

- **Determinism:** TabPFN-2, TabICL and TabDPT return identical probabilities when run twice with the same seed. The TabDPT README says its runs can't be exactly replicated, but with a seed they were, at least on MPS.
- **Seed variation:** the ROC-AUC varies across seeds by 0.0002–0.017 (std). That is 5–20 times smaller than the bootstrap std of the test set.
- **Row independence:** each test row's prediction does not depend on the other test rows. So the test set can be predicted once and the probabilities bootstrapped.
- **Speed:** on MPS, TabPFN-2 takes 1–200 seconds per dataset, depending on its size. TabDPT takes 1.5–25 seconds on small datasets and 4–9 minutes on datasets with 21k–34k training rows, because it reduces features with PCA and subsamples the context to a fixed size.
