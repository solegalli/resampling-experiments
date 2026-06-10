"""
Evaluate the undersampling models trained on the new datasets.

Loads each model saved by train-undersampling-new.py (RandomUnderSampler on the
five new datasets), evaluates it on the test set with bootstrapped samples
(metrics at the optimal threshold), and stores a merged results pickle in
models/undersampling-new/, keyed by dataset then by ``{estimator}_rus``.
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

LOADERS = {
    "secom": load_hard_dataset,
    "default_credit": load_hard_dataset,
    "diabetes130": load_hard_dataset,
    "htru2": load_imbalanced_dataset,
    "creditcard": load_imbalanced_dataset,
}

MODELS_DIR = REPO_ROOT / "models" / "undersampling-new"

scores_dict = {}

for dataset in tqdm(LOADERS, desc="Datasets"):
    _, X_test, _, y_test = LOADERS[dataset](dataset)

    scores_dict[dataset] = {}
    for estimator in estimator_dict:
        model = joblib.load(MODELS_DIR / f"{dataset}_{estimator}_rus.pkl")
        scores_dict[dataset][f"{estimator}_rus"] = evaluate_model_on_test_set(
            model, X_test, y_test
        )

with open(MODELS_DIR / "results", "wb") as fp:
    pickle.dump(scores_dict, fp)
