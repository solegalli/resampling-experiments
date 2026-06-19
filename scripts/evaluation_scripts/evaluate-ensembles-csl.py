"""
Evaluate pre-trained ensemble models trained on various imbalanced datasets with sample weights
for cost sensitive learning.

Loads each model saved by train-ensembles-csl.py, evaluates it on the test set
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
from functions.cv import get_sample_weights
from functions.data import DATASETS_LS, load_dataset
from functions.evaluation import evaluate_model_on_test_set

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)

MODELS_DIR = REPO_ROOT / "models" / "csl"

scores_dict = {}

for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    _, X_test, y_train, y_test = load_dataset(dataset)

    sample_weights = get_sample_weights(y_train)

    scores_dict[dataset] = {}

    for estimator in tqdm(estimator_dict, desc=dataset, leave=False):
        for weight in sample_weights:
            search = joblib.load(MODELS_DIR / f"{dataset}_{estimator}_sw{weight}.pkl")
            scores_dict[dataset][f"{estimator}_sw{weight}"] = evaluate_model_on_test_set(
                search, X_test, y_test
            )

with open(MODELS_DIR / "results.pkl", "wb") as fp:
    pickle.dump(scores_dict, fp)
