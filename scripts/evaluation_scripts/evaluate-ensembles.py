"""
Evaluate pre-trained classical machine learning classifiers on various imbalanced datasets.

Loads each model saved by train-ensembles.py, evaluates it on the test set
using bootstrapped samples, and stores the results as a pickle file in the same folder
as the models.
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
from functions.data import DATASETS_LS, load_dataset
from functions.evaluation import evaluate_model_on_test_set

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

MODELS_DIR = REPO_ROOT / "models" / "ensembles"

scores_dict = {}
folds_dict = {}

for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    _, X_test, _, y_test = load_dataset(dataset)

    scores_dict[dataset] = {}
    folds_dict[dataset] = {}

    for estimator in tqdm(estimator_dict, desc=dataset, leave=False):
        search = joblib.load(MODELS_DIR / f"{dataset}_{estimator}.pkl")
        results, fold_metrics = evaluate_model_on_test_set(search, X_test, y_test)
        scores_dict[dataset][estimator] = results
        folds_dict[dataset][estimator] = fold_metrics

with open(MODELS_DIR / "results", "wb") as fp:
    pickle.dump(scores_dict, fp)

with open(MODELS_DIR / "results_folds", "wb") as fp:
    pickle.dump(folds_dict, fp)
