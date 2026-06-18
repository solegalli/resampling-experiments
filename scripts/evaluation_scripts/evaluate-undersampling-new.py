"""
Evaluate the undersampling models trained on the new datasets.

Loads each model saved by train-undersampling-new.py, evaluates it on the test
set with bootstrapped samples (metrics at the optimal threshold), and stores a
merged results pickle in models/undersampling-new/, keyed by dataset then by
``{estimator}_{undersampler}`` (matching evaluate-undersampling.py).

The dataset/undersampler scope mirrors the training script: the full suite on
htru2/default_credit/secom and the row-reducing methods on diabetes130/creditcard.
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
from functions.evaluation import evaluate_model_on_test_set
from functions.hard_data import load_hard_dataset
from functions.imbalanced_data import load_imbalanced_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

FULL_SUITE = [
    "rus",
    "cnn",
    "tomek",
    "oss",
    "enn",
    "renn",
    "allknn",
    "ncr",
    "nm1",
    "nm2",
]
ROW_REDUCERS = ["rus", "nm1", "nm2"]

DATASETS = [
    ("htru2", load_imbalanced_dataset, FULL_SUITE),
    ("default_credit", load_hard_dataset, FULL_SUITE),
    ("diabetes130", load_hard_dataset, ROW_REDUCERS),
    ("creditcard", load_imbalanced_dataset, ROW_REDUCERS),
    ("secom", load_hard_dataset, FULL_SUITE),
]

MODELS_DIR = REPO_ROOT / "models" / "undersampling-new"

scores_dict = {}

for dataset, loader, undersamplers in tqdm(DATASETS, desc="Datasets"):
    _, X_test, _, y_test = loader(dataset)

    scores_dict[dataset] = {}
    for undersampler in undersamplers:
        for estimator in estimator_dict:
            model = joblib.load(
                MODELS_DIR / f"{dataset}_{estimator}_{undersampler}.pkl"
            )
            scores_dict[dataset][f"{estimator}_{undersampler}"] = (
                evaluate_model_on_test_set(model, X_test, y_test)
            )

with open(MODELS_DIR / "results", "wb") as fp:
    pickle.dump(scores_dict, fp)
