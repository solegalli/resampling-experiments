"""
Train classifiers with undersampling on the new hard / strongly-imbalanced datasets.

Mirrors the per-undersampler scripts (train-rus.py, train-cnn.py, ...) but covers
the datasets added in functions.hard_data and functions.imbalanced_data.

Scope and runtime
-----------------
The neighbour-based undersamplers (CNN, ENN, RENN, AllKNN, NCR, OSS, NearMiss) are
very slow on large data -- the same days-scale runtime seen on datasets like
protein_homo and isolet. They are not impossible, just expensive: the cleaning
methods in particular barely shrink the majority class, so the classifier is then
tuned (100-candidate manual successive halving) on near-full data, which on secom
(474 features) is the dominant cost.

We therefore run the full 10-method suite on the three datasets where it finishes
in a reasonable time (htru2, default_credit, secom) and the row-reducing methods
only (RandomUnderSampler, NearMiss) on the two largest datasets (diabetes130 ~71k
rows, creditcard ~199k), where the cleaning methods would need a dedicated
multi-day run. Datasets are ordered cheapest-first so the easy-vs-hard removal
comparison (htru2 vs default_credit) lands early.

Undersampling is applied once per (dataset, undersampler) and reused across all
classifiers; distance-based methods are guided by a MinMaxScaler (scale=True) but
models are trained on the original scale. Saved to models/undersampling-new/.
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
from configs.undersamplers import (
    allknn,
    cnn,
    enn,
    ncr,
    nm1,
    nm2,
    oss,
    renn,
    rus,
    tomek,
)
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data
from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import load_imbalanced_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

# (name, sampler, scale): scale=True for distance-based methods, as in the
# original per-undersampler training scripts.
SAMPLERS = {
    "rus": (rus, False),
    "cnn": (cnn, True),
    "tomek": (tomek, True),
    "oss": (oss, True),
    "enn": (enn, True),
    "renn": (renn, True),
    "allknn": (allknn, True),
    "ncr": (ncr, True),
    "nm1": (nm1, True),
    "nm2": (nm2, True),
}

FULL_SUITE = list(SAMPLERS)
ROW_REDUCERS = ["rus", "nm1", "nm2"]

# Loader + which undersamplers to run. Ordered cheapest-first, with secom (the
# slow 474-feature dataset) last, so an interruption only costs secom.
DATASETS = [
    ("htru2", load_imbalanced_dataset, FULL_SUITE),
    ("default_credit", load_hard_dataset, FULL_SUITE),
    ("diabetes130", load_hard_dataset, ROW_REDUCERS),
    ("creditcard", load_imbalanced_dataset, ROW_REDUCERS),
    ("secom", load_hard_dataset, FULL_SUITE),
]

OUTPUT_DIR = REPO_ROOT / "models" / "undersampling-new"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume: reuse any (dataset, undersampler) already trained in a previous run, so
# a fresh checkout trains everything while a re-run only fills what is missing.
sampling_stats = {}
if (OUTPUT_DIR / "sampling_stats").exists():
    with open(OUTPUT_DIR / "sampling_stats", "rb") as fp:
        sampling_stats = pickle.load(fp)

for dataset, loader, undersamplers in tqdm(DATASETS, desc="Datasets"):
    sampling_stats.setdefault(dataset, {})
    todo = [
        name
        for name in undersamplers
        if not all(
            (OUTPUT_DIR / f"{dataset}_{est}_{name}.pkl").exists()
            for est in estimator_dict
        )
    ]
    if not todo:
        continue
    X_train, X_test, y_train, y_test = loader(dataset)

    for name in tqdm(todo, desc=dataset, leave=False):
        sampler, scale = SAMPLERS[name]

        t0 = time.time()
        xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(
            sampler, X_train, y_train, scale=scale
        )
        stats["undersample_seconds"] = round(time.time() - t0, 1)
        sampling_stats[dataset][name] = stats

        for estimator, params in zip(estimator_dict, hyperparam_ensemble_dict):
            model = train_model_w_undersampling(
                estimator_dict[estimator],
                hyperparam_ensemble_dict[params],
                xtrainu,
                ytrainu,
                xtest,
                ytest,
                Xu,
                yu,
            )
            joblib.dump(model, OUTPUT_DIR / f"{dataset}_{estimator}_{name}.pkl")

        # checkpoint stats after each undersampler (long run)
        with open(OUTPUT_DIR / "sampling_stats", "wb") as fp:
            pickle.dump(sampling_stats, fp)
