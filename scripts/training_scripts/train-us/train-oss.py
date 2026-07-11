"""
Train classifiers with One-Sided Selection undersampling on various imbalanced datasets.

Undersampling is applied once before hyperparameter search to avoid resampling
on every candidate. Models are tuned with a manual successive halving approach
and saved to disk for later evaluation.
"""

import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pickle

import joblib
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from configs.undersamplers import oss
from functions.cv_undersamplers import train_model_w_undersampling, undersample_data
from functions.data import DATASETS_LS, load_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "oss"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sampling_stats = {}

for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    xtrainu, ytrainu, xtest, ytest, Xu, yu, stats = undersample_data(
        oss, X_train, y_train, scale=True
    )
    sampling_stats[dataset] = stats

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_ensemble_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        start_time = time.time()
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
        search.fit_time = time.time() - start_time
        joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}_oss.pkl")

with open(OUTPUT_DIR / "sampling_stats", "wb") as fp:
    pickle.dump(sampling_stats, fp)
