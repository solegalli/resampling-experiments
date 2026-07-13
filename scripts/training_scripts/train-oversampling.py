"""
Train classifiers with oversampling on various imbalanced datasets.

Oversampling is applied once before hyperparameter search to avoid resampling
on every candidate. Models are tuned with a manual successive halving approach
and saved to disk for later evaluation.
"""

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from configs.oversamplers import (
    adasyn_equal,
    adasyn_half,
    bsmote1_equal,
    bsmote1_half,
    bsmote2_equal,
    bsmote2_half,
    ros_equal,
    ros_half,
    ros_shrink01_equal,
    ros_shrink1_equal,
    ros_shrink01_half,
    ros_shrink1_half,
    smote_equal,
    smote_half,
)
from functions.cv_oversamplers import oversample_data
from functions.cv_undersamplers import train_model_w_undersampling
from functions.data import load_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "oversampling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

oversamplers = {
    "ros_equal": ros_equal,
    "ros_half": ros_half,
    "ros_shrink01_equal": ros_shrink01_equal,
    "ros_shrink01_half": ros_shrink01_half,
    "ros_shrink1_equal": ros_shrink1_equal,
    "ros_shrink1_half": ros_shrink1_half,
    "smote_equal": smote_equal,
    "smote_half": smote_half,
    "adasyn_equal": adasyn_equal,
    "adasyn_half": adasyn_half,
    "bsmote1_equal": bsmote1_equal,
    "bsmote1_half": bsmote1_half,
    "bsmote2_equal": bsmote2_equal,
    "bsmote2_half": bsmote2_half,
}

# candidate datasets: few samples, roc-auc < 0.9, continuous features
DATASETS = ["glass-0-1-4-6_vs_2", "pima", "ozone_level", "scene", "secom"]

for name, sampler in oversamplers.items():
    for dataset in tqdm(DATASETS, desc=f"Datasets - {name}"):
        X_train, X_test, y_train, y_test = load_dataset(dataset)

        # Skip _half oversamplers when the dataset is not imbalanced enough.
        # sampling_strategy=0.5 requires minority < 50% of majority; we use 40%
        # as a safety margin to avoid borderline failures during CV folds (eg pima).
        counts = np.bincount(y_train)
        minority_ratio = counts.min() / counts.max()
        if "_half" in name and minority_ratio >= 0.4:
            tqdm.write(
                f"Skipping {dataset} with {name}: minority ratio {minority_ratio:.2f} >= 0.4."
            )
            continue

        if "ros" in name:
            xtraino, ytraino, xtest, ytest, Xo, yo = oversample_data(
                sampler, X_train, y_train, scale=False
            )
        else:
            # for smote and adasyn we need to scale the data before oversampling
            xtraino, ytraino, xtest, ytest, Xo, yo = oversample_data(
                sampler, X_train, y_train, scale=True,
            )

        for estimator, params in tqdm(
            zip(estimator_dict, hyperparam_ensemble_dict),
            desc=dataset,
            total=len(estimator_dict),
            leave=False,
        ):
            model = train_model_w_undersampling(
                estimator_dict[estimator],
                hyperparam_ensemble_dict[params],
                xtraino,
                ytraino,
                xtest,
                ytest,
                Xo,
                yo,
            )
            joblib.dump(model, OUTPUT_DIR / f"{dataset}_{estimator}_{name}.pkl")
