"""
Evaluate pre-trained classifiers trained with IHT undersampling on various imbalanced datasets.

Loads each model saved by the train-iht.py scripts, evaluates it on the
test set using bootstrapped samples, and stores one results pickle per undersampler
in its corresponding models folder.
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

IHT_VERSIONS = [
    "iht07",
    "iht06",
    "iht05",
    "iht04",
    "iht03",
]

models_dir = REPO_ROOT / "models" / "iht"

for undersampler in tqdm(IHT_VERSIONS, desc="IHT Versions"):
    scores_dict = {}
    folds_dict = {}

    for dataset in tqdm(DATASETS_LS, desc=undersampler, leave=False):
        _, X_test, _, y_test = load_dataset(dataset)

        scores_dict[dataset] = {}
        folds_dict[dataset] = {}

        for estimator in estimator_dict:
            model = joblib.load(
                models_dir / f"{dataset}_{estimator}_{undersampler}.pkl"
            )
            results, fold_metrics = evaluate_model_on_test_set(model, X_test, y_test)
            scores_dict[dataset][f"{estimator}_{undersampler}"] = results
            folds_dict[dataset][f"{estimator}_{undersampler}"] = fold_metrics

    with open(models_dir / f"results_{undersampler}", "wb") as fp:
        pickle.dump(scores_dict, fp)

    with open(models_dir / f"results_folds_{undersampler}", "wb") as fp:
        pickle.dump(folds_dict, fp)
