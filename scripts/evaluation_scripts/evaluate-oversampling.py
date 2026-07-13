"""
Evaluate pre-trained classifiers trained with oversampling on various imbalanced datasets.

Loads each model saved by train-oversampling.py, evaluates it on the
test set using bootstrapped samples, and stores one results pickle per oversampler
in the models/oversampling folder.
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
from functions.data import load_dataset
from functions.evaluation import evaluate_model_on_test_set

warnings.filterwarnings(
    "ignore", message=".*X does not have valid feature names.*", category=UserWarning
)
warnings.simplefilter(action="ignore", category=FutureWarning)

OVERSAMPLERS = [
    "ros_equal",
    "ros_half",
    "ros_shrink01_equal",
    "ros_shrink01_half",
    "ros_shrink1_equal",
    "ros_shrink1_half",
    "smote_equal",
    "smote_half",
    "adasyn_equal",
    "adasyn_half",
    "bsmote1_equal",
    "bsmote1_half",
    "bsmote2_equal",
    "bsmote2_half",
]

models_dir = REPO_ROOT / "models" / "oversampling"

# candidate datasets: few samples, roc-auc < 0.9, continuous features
DATASETS = ["glass-0-1-4-6_vs_2", "pima", "ozone_level", "scene", "secom"]

for oversampler in tqdm(OVERSAMPLERS, desc="Oversamplers"):
    scores_dict = {}
    folds_dict = {}

    for dataset in tqdm(DATASETS, desc=oversampler, leave=False):
        _, X_test, _, y_test = load_dataset(dataset)

        scores_dict[dataset] = {}
        folds_dict[dataset] = {}

        for estimator in estimator_dict:
            modeldir = f"{dataset}_{estimator}_{oversampler}.pkl"
            if "pima" in modeldir and "half" in modeldir:
                tqdm.write(f"Skipping modeldir: minority ratio in pima was >= 0.4.")
                continue
            model = joblib.load(models_dir / modeldir)
            results, fold_metrics = evaluate_model_on_test_set(model, X_test, y_test)
            scores_dict[dataset][estimator] = results
            folds_dict[dataset][estimator] = fold_metrics

    with open(models_dir / f"results_{oversampler}", "wb") as fp:
        pickle.dump(scores_dict, fp)

    with open(models_dir / f"results_folds_{oversampler}", "wb") as fp:
        pickle.dump(folds_dict, fp)
