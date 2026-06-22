"""
Evaluate IHT-undersampling models trained on the new datasets.

Loads each model saved by train-iht-new.py, evaluates it on the test set with
bootstrapped samples (metrics at the optimal threshold), and stores one results
pickle per IHT threshold in models/iht-new/ (matching evaluate-iht.py).
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

IHT_VERSIONS = ["iht03", "iht04", "iht05", "iht06", "iht07"]

LOADERS = {
    "secom": load_hard_dataset,
    "htru2": load_imbalanced_dataset,
    "default_credit": load_hard_dataset,
    "creditcard": load_imbalanced_dataset,
    "diabetes130": load_hard_dataset,
}

MODELS_DIR = REPO_ROOT / "models" / "iht-new"

for undersampler in tqdm(IHT_VERSIONS, desc="IHT versions"):
    scores_dict = {}

    for dataset, loader in LOADERS.items():
        _, X_test, _, y_test = loader(dataset)

        scores_dict[dataset] = {}
        for estimator in estimator_dict:
            model = joblib.load(
                MODELS_DIR / f"{dataset}_{estimator}_{undersampler}.pkl"
            )
            scores_dict[dataset][f"{estimator}_{undersampler}"] = (
                evaluate_model_on_test_set(model, X_test, y_test)
            )

    with open(MODELS_DIR / f"results_{undersampler}", "wb") as fp:
        pickle.dump(scores_dict, fp)
