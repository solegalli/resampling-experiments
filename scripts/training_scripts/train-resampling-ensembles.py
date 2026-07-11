"""
Train special ensemble classifiers on various imbalanced datasets.

Models trained: Balanced Random Forest, Easy Ensemble, and RUSBoost —
ensemble methods that incorporate resampling into their design. Hyperparameter
tuning uses successive halving with the number of trees as the limiting resource.

Note: "poker-8-9_vs_5" requires a custom configuration (sampling_strategy=0.5)
because the default "auto" strategy fails for this dataset.
"""

import sys
import time
import warnings
from pathlib import Path

from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from configs.hyperparams import hyperparam_special_dict
from configs.resampling_ensembles import estimator_dict
from functions.cv import train_model
from functions.data import DATASETS_LS, load_dataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="The total space of parameters")
warnings.simplefilter(action="ignore", category=FutureWarning)

OUTPUT_DIR = REPO_ROOT / "models" / "special-ensembles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_models_for_poker():
    # poker-8-9_vs_5 fails with sampling_strategy="auto" because the minority class
    # is too small; sampling_strategy=0.5 is used instead.
    dataset = "poker-8-9_vs_5"
    X_train, X_test, y_train, y_test = load_dataset(dataset)

    rus = RUSBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=10),
        learning_rate=1.0,
        random_state=2909,
        sampling_strategy=0.5,
    )
    search = train_model(rus, {"learning_rate": stats.uniform(0, 10)}, X_train, y_train)
    joblib.dump(search, OUTPUT_DIR / f"{dataset}_rusboost.pkl")

    easy = EasyEnsembleClassifier(
        estimator=AdaBoostClassifier(random_state=10, n_estimators=10),
        random_state=2909,
        sampling_strategy=0.5,
    )
    search = train_model(easy, {"replacement": (True, False)}, X_train, y_train)
    joblib.dump(search, OUTPUT_DIR / f"{dataset}_easyEnsemble.pkl")

    search = train_model(
        estimator_dict["balancedRF"],
        hyperparam_special_dict["brf_params"],
        X_train,
        y_train,
    )
    joblib.dump(search, OUTPUT_DIR / f"{dataset}_balancedRF.pkl")


for dataset in tqdm(DATASETS_LS, desc="Datasets"):
    if dataset == "poker-8-9_vs_5":
        train_models_for_poker()
    else:
        X_train, X_test, y_train, y_test = load_dataset(dataset)

        for estimator, params in tqdm(
            zip(estimator_dict, hyperparam_special_dict),
            desc=dataset,
            total=len(estimator_dict),
            leave=False,
        ):
            start_time = time.time()
            search = train_model(
                estimator_dict[estimator],
                hyperparam_special_dict[params],
                X_train,
                y_train,
                scoring="roc_auc",
            )
            search.fit_time = time.time() - start_time
            joblib.dump(search, OUTPUT_DIR / f"{dataset}_{estimator}.pkl")
