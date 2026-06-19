"""
Train ensemble (tree-based) machine learning classifiers on various imbalanced datasets
using cost-sensitive learning. Because most models don't accept the parameter class_weight
we opted for passing the sample weights vector.

Models trained: random forests, XGBoost, LightGBM, CatBoost, AdaBoost, and
gradient boosting machines from scikit-learn. Hyperparameter tuning uses
successive halving with the number of trees as the limiting resource.
"""

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from tqdm import tqdm

from configs.ensemble_models import estimator_dict
from configs.hyperparams import hyperparam_ensemble_dict
from functions.cv import get_sample_weights, train_model
from functions.data import DATASETS_LS, load_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

OUTPUT_DIR = REPO_ROOT / "models" / "csl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    sample_weights = get_sample_weights(y_train)

    for estimator, params in tqdm(
        zip(estimator_dict, hyperparam_ensemble_dict),
        desc=dataset,
        total=len(estimator_dict),
        leave=False,
    ):
        for weight, sw in sample_weights.items():
            search = train_model(
                estimator_dict[estimator],
                hyperparam_ensemble_dict[params],
                X_train,
                y_train,
                scoring="roc_auc",
                sample_weight=sw,
            )
            joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}_sw{weight}.pkl")
