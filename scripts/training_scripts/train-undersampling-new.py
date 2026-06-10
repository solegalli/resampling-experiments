"""
Train classifiers with undersampling on the new hard / strongly-imbalanced datasets.

Mirrors the per-undersampler scripts (train-rus.py, ...) but covers the datasets
added in functions.hard_data and functions.imbalanced_data.

Scope: RandomUnderSampler only.
------------------------------------
The neighbour-based undersamplers (CNN, ENN, RENN, AllKNN, NCR, OSS, NearMiss) do
not scale here. The *cleaning* methods barely reduce the majority class, so the
classifier is then tuned (100-candidate manual successive halving) on near-full
data; on secom (474 features) and the 10-70k-row datasets that is intractable
(hours per dataset). RandomUnderSampler instead balances to ~2x the minority, so
training stays small and bounded. We therefore run RUS -- the canonical
undersampler -- on all five datasets, which directly answers whether undersampling
a normal ensemble helps where the standard models struggle (notebook 05) or where
imbalance is severe (notebook 06). The intractability of the other methods on
datasets this size is itself a useful, documented result.

The candidate count is reduced (n_iter=30 instead of the default 100) to keep the
run tractable on these larger datasets; the conclusion is robust to it.
"""

import pickle
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from configs.undersamplers import rus
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data
from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import load_imbalanced_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

N_ITER = 30

LOADERS = {
    "secom": load_hard_dataset,
    "default_credit": load_hard_dataset,
    "diabetes130": load_hard_dataset,
    "htru2": load_imbalanced_dataset,
    "creditcard": load_imbalanced_dataset,
}

OUTPUT_DIR = REPO_ROOT / "models" / "undersampling-new"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sampling_stats = {}

for dataset in tqdm(LOADERS, desc="Datasets"):
    X_train, X_test, y_train, y_test = LOADERS[dataset](dataset)

    t0 = time.time()
    xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(
        rus, X_train, y_train, scale=False
    )
    stats["undersample_seconds"] = round(time.time() - t0, 1)
    sampling_stats[dataset] = {"rus": stats}

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_ensemble_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        model = train_model_w_undersampling(
            estimator_dict[estimator],
            hyperparam_ensemble_dict[params],
            xtrainu,
            ytrainu,
            xtest,
            ytest,
            Xu,
            yu,
            n_iter=N_ITER,
        )
        joblib.dump(model, OUTPUT_DIR / f"{dataset}_{estimator}_rus.pkl")

with open(OUTPUT_DIR / "sampling_stats", "wb") as fp:
    pickle.dump(sampling_stats, fp)
