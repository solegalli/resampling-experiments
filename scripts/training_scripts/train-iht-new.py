"""
Train classifiers with Instance Hardness Threshold (IHT) undersampling on the
new hard / strongly-imbalanced datasets.

Mirrors scripts/training_scripts/train-iht.py (the InstanceHardnessUnderSampler at
thresholds 0.3-0.7, tuned with the manual successive halving), but on the datasets
added in functions.hard_data and functions.imbalanced_data.

IHT removes the majority-class instances a cross-validated classifier scores as
"hard" (P(class=1) > threshold); it does not balance, so a high threshold leaves
the training set near its original size. Datasets are ordered so the small ones
run first and diabetes130 (~71k) last.
"""

import pickle
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from configs.iht import iht03, iht04, iht05, iht06, iht07
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data
from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import load_imbalanced_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

IHT_SAMPLERS = {
    "iht03": iht03,
    "iht04": iht04,
    "iht05": iht05,
    "iht06": iht06,
    "iht07": iht07,
}

# loader per dataset, ordered fastest-first; creditcard (full 284k) is by far the
# slowest so it runs last. Models already on disk are skipped, so the run resumes
# cleanly after a sleep or interruption.
LOADERS = {
    "htru2": load_imbalanced_dataset,
    "default_credit": load_hard_dataset,
    "diabetes130": load_hard_dataset,
    "secom": load_hard_dataset,
    "creditcard": load_imbalanced_dataset,
}

OUTPUT_DIR = REPO_ROOT / "models" / "iht-new"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATS_PATH = OUTPUT_DIR / "sampling_stats"
if STATS_PATH.exists():
    with open(STATS_PATH, "rb") as f:
        sampling_stats = pickle.load(f)
else:
    sampling_stats = {name: {} for name in IHT_SAMPLERS}

for dataset, loader in tqdm(LOADERS.items(), desc="Datasets"):
    X_train, X_test, y_train, y_test = loader(dataset)

    for name, sampler in tqdm(IHT_SAMPLERS.items(), desc=dataset, leave=False):
        out_paths = {
            est: OUTPUT_DIR / f"{dataset}_{est}_{name}.pkl" for est in estimator_dict
        }
        # resume: skip this (dataset, undersampler) entirely if it is already done
        if all(
            p.exists() for p in out_paths.values()
        ) and dataset in sampling_stats.get(name, {}):
            continue

        xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(
            sampler, X_train, y_train, scale=False
        )
        sampling_stats.setdefault(name, {})[dataset] = stats

        for estimator, params in zip(estimator_dict, hyperparam_ensemble_dict):
            # scikit-learn's GradientBoosting does not scale to the full
            # creditcard (199k rows); its 1000-tree halving search runs for hours
            # per IHT threshold. xgboost/lightgbm/catboost cover gradient boosting
            # here and fit in seconds, so skip it on this dataset only.
            if dataset == "creditcard" and estimator == "gbm":
                continue
            if out_paths[estimator].exists():
                continue  # resume: skip a model already trained
            search = train_model_w_undersampling(
                estimator_dict[estimator],
                hyperparam_ensemble_dict[params],
                xtrainu,
                ytrainu,
                xtest,
                ytest,
                Xu,
                yu,
            )
            joblib.dump(search, out_paths[estimator])

        with open(STATS_PATH, "wb") as fp:
            pickle.dump(sampling_stats, fp)
